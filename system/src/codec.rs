//! Stage 5 — "best-of" byte coder.
//!
//! Column grammar (and any other caller that has a short-to-medium byte
//! buffer to compress) used to route everything through the order-0
//! `range_coder`. That's optimal for *highly skewed* byte histograms like
//! a post-resonance residual, but it loses 25–40% on text or dictionary
//! payloads because it can't exploit inter-byte (order-1) structure.
//!
//! This module picks per-buffer between three candidates and emits a
//! 1-byte tag so the decoder can reverse the right one:
//!
//!   tag 0x00  raw              — smallest winner for very tiny / random
//!   tag 0x01  range coder      — order-0, great for skewed residuals
//!   tag 0x02  rANS order-1     — better on text, dict indices, sources
//!                                with inter-byte context
//!
//! Tags 0x10+ are reserved for preprocessors that forward to `best_encode`
//! after transforming the data (currently: u32 varint-delta).
//!
//! Zero external deps: both coders are native to this crate.

const TAG_RAW:   u8 = 0x00;
const TAG_RANGE: u8 = 0x01;
const TAG_RANS:  u8 = 0x02;

// A rANS order-1 model has O(n_contexts × n_symbols) header bytes. On
// very small payloads that header dwarfs any savings, so skip rANS.
const RANS_MIN_INPUT: usize = 64;

/// Encode `data` using whichever native coder produces the smallest
/// self-describing output. Output always begins with a 1-byte tag.
///
/// The only cost of the "try all" strategy is encoding time; we keep
/// this O(n) so the impact on large columns is a small constant factor.
pub fn best_encode(data: &[u8]) -> Vec<u8> {
    // RAW is always a valid candidate. Treat it as the floor.
    let raw_size = 1 + data.len();
    let mut best_tag  = TAG_RAW;
    let mut best_size = raw_size;

    // Order-0 range coder — cheap header (≤ 2 + 256·5 + 4 bytes).
    let rc = crate::range_coder::encode(data);
    let rc_size = 1 + rc.len();
    if rc_size < best_size {
        best_tag  = TAG_RANGE;
        best_size = rc_size;
    }

    // Order-1 rANS — heavier header, but wins on anything with inter-byte
    // structure (text, dictionary indices, source code, log lines). Skip
    // it on tiny inputs where the per-context header swamps any gain.
    let rn = if data.len() >= RANS_MIN_INPUT {
        let enc = crate::rans::encode(data);
        let rn_size = 1 + enc.len();
        if rn_size < best_size {
            best_tag  = TAG_RANS;
            best_size = rn_size;
        }
        Some(enc)
    } else {
        None
    };

    let mut out = Vec::with_capacity(best_size);
    out.push(best_tag);
    match best_tag {
        TAG_RAW   => out.extend_from_slice(data),
        TAG_RANGE => out.extend_from_slice(&rc),
        TAG_RANS  => out.extend_from_slice(rn.as_ref().unwrap()),
        _         => unreachable!(),
    }
    out
}

/// Inverse of `best_encode`. Returns `None` if the tag is unknown or the
/// inner coder fails.
pub fn best_decode(data: &[u8]) -> Option<Vec<u8>> {
    if data.is_empty() {
        // Legacy / idiomatic: empty payload decodes to empty data.
        return Some(vec![]);
    }
    let tag  = data[0];
    let body = &data[1..];
    match tag {
        TAG_RAW   => Some(body.to_vec()),
        TAG_RANGE => crate::range_coder::decode(body),
        TAG_RANS  => crate::rans::decode(body),
        _ => None,
    }
}

// ─── Integer-column helpers ─────────────────────────────────────────────
//
// Monotonic (or near-monotonic) integer columns — timestamps, auto-
// increment IDs, offsets — benefit dramatically from delta coding before
// entropy coding. A 32-bit timestamp incremented by ~1 per row stores as
// a single byte of varint per delta, and then rANS / range-coder crushes
// the result further. Absolute u32-LE encoding cannot exploit this.

/// Encode a `u32` slice as [first value LE][varint deltas (wrapping)].
/// The decoder reconstructs absolute values using wrapping_add.
pub fn encode_u32_delta(values: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2 + 4);
    if values.is_empty() {
        return out;
    }
    let first = values[0];
    out.extend_from_slice(&first.to_le_bytes());
    let mut prev = first;
    for &v in &values[1..] {
        let diff = v.wrapping_sub(prev);
        put_varint(&mut out, diff as u64);
        prev = v;
    }
    out
}

/// Inverse of `encode_u32_delta`. `count` is the expected number of values.
pub fn decode_u32_delta(data: &[u8], count: usize) -> Option<Vec<u32>> {
    if count == 0 {
        return Some(vec![]);
    }
    if data.len() < 4 {
        return None;
    }
    let mut out = Vec::with_capacity(count);
    let first = u32::from_le_bytes(data[0..4].try_into().ok()?);
    out.push(first);
    let mut prev = first;
    let mut p = 4usize;
    for _ in 1..count {
        let (d, n) = get_varint(data, p)?;
        let v = prev.wrapping_add(d as u32);
        out.push(v);
        prev = v;
        p += n;
    }
    Some(out)
}

/// Same shape for `u16` data (rarely needed in practice but symmetric).
pub fn encode_u16_delta(values: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() + 2);
    if values.is_empty() {
        return out;
    }
    let first = values[0];
    out.extend_from_slice(&first.to_le_bytes());
    let mut prev = first;
    for &v in &values[1..] {
        let diff = v.wrapping_sub(prev);
        put_varint(&mut out, diff as u64);
        prev = v;
    }
    out
}

pub fn decode_u16_delta(data: &[u8], count: usize) -> Option<Vec<u16>> {
    if count == 0 {
        return Some(vec![]);
    }
    if data.len() < 2 {
        return None;
    }
    let mut out = Vec::with_capacity(count);
    let first = u16::from_le_bytes(data[0..2].try_into().ok()?);
    out.push(first);
    let mut prev = first;
    let mut p = 2usize;
    for _ in 1..count {
        let (d, n) = get_varint(data, p)?;
        let v = prev.wrapping_add(d as u16);
        out.push(v);
        prev = v;
        p += n;
    }
    Some(out)
}

// ─── Little unsigned LEB128 (varint) ────────────────────────────────────

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn get_varint(data: &[u8], mut p: usize) -> Option<(u64, usize)> {
    let start = p;
    let mut v: u64 = 0;
    let mut shift = 0u32;
    loop {
        let b = *data.get(p)?;
        p += 1;
        v |= ((b & 0x7F) as u64) << shift;
        if b & 0x80 == 0 {
            return Some((v, p - start));
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_empty() {
        let enc = best_encode(&[]);
        assert_eq!(best_decode(&enc).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn roundtrip_tiny_picks_raw() {
        // Tiny data — header overhead > savings, raw must win.
        let data = b"abc";
        let enc = best_encode(data);
        assert_eq!(enc[0], TAG_RAW);
        assert_eq!(best_decode(&enc).unwrap(), data);
    }

    #[test]
    fn roundtrip_skewed_byte_histogram() {
        // Mostly zeros + a few rare bytes — range coder should win.
        let mut data = vec![0u8; 4000];
        for i in (0..4000).step_by(37) { data[i] = 7; }
        let enc = best_encode(&data);
        // Must be much smaller than input.
        assert!(enc.len() < data.len() / 4,
                "expected >4x, got {} -> {}", data.len(), enc.len());
        assert_eq!(best_decode(&enc).unwrap(), data);
    }

    #[test]
    fn roundtrip_text_rans_wins() {
        // Natural-text-like source — order-1 rANS should beat range coder.
        let mut data = Vec::new();
        for _ in 0..2000 {
            data.extend_from_slice(b"the quick brown fox jumps over the lazy dog\n");
        }
        let enc = best_encode(&data);
        // Order-1 rANS on repetitive English text: ~3–4× is typical.
        assert!(enc.len() * 3 < data.len(),
                "expected >3x on repetitive English, got {} -> {}",
                data.len(), enc.len());
        assert_eq!(best_decode(&enc).unwrap(), data);
    }

    #[test]
    fn roundtrip_random_raw_floor() {
        // Random bytes: neither coder can beat raw + 1 byte tag.
        let data: Vec<u8> = (0..200).map(|i| ((i * 131 + 17) ^ 0x55) as u8).collect();
        let enc = best_encode(&data);
        // Output should never be wildly larger than input+tag.
        assert!(enc.len() <= data.len() + 16,
                "{}B ciphertext for {}B plaintext is excessive", enc.len(), data.len());
        assert_eq!(best_decode(&enc).unwrap(), data);
    }

    #[test]
    fn varint_roundtrip_monotonic_u32s() {
        let values: Vec<u32> = (0..10_000).map(|i| 1_700_000_000u32 + i).collect();
        let enc = encode_u32_delta(&values);
        // Delta stream should be much smaller than 40000 bytes (absolute
        // u32-LE). In fact it's ~ 4 + 9999 one-byte varints ≈ 10kB.
        assert!(enc.len() < values.len() * 4);
        let dec = decode_u32_delta(&enc, values.len()).unwrap();
        assert_eq!(dec, values);
    }

    #[test]
    fn varint_roundtrip_nearly_sorted() {
        // "Nearly sorted": mostly ascending with occasional small jitter.
        let values: Vec<u32> = (0..5000)
            .map(|i| (i * 10 + ((i * 37) % 7)) as u32)
            .collect();
        let enc = encode_u32_delta(&values);
        let dec = decode_u32_delta(&enc, values.len()).unwrap();
        assert_eq!(dec, values);
    }

    #[test]
    fn varint_roundtrip_u16_deltas() {
        let values: Vec<u16> = (0..1000).map(|i| (i * 3) as u16).collect();
        let enc = encode_u16_delta(&values);
        let dec = decode_u16_delta(&enc, values.len()).unwrap();
        assert_eq!(dec, values);
    }

    #[test]
    fn delta_then_best_encode_hits_ultra_ratio() {
        // Classic timestamp column: hit the same "nearly-sorted u32s"
        // regime the top-level pipeline hits, to make sure the stacked
        // transform (delta → best_encode) does the heavy lifting.
        let values: Vec<u32> = (0..50_000)
            .map(|i| 1_700_000_000u32 + i)
            .collect();
        let d = encode_u32_delta(&values);
        let enc = best_encode(&d);
        // 50_000 × 4B raw = 200kB absolute. We expect ≪ 5kB here.
        assert!(enc.len() < 5_000,
                "expected ≪5kB for monotonic 50k-u32 timestamp column, got {}",
                enc.len());
        let d2 = best_decode(&enc).unwrap();
        let vals2 = decode_u32_delta(&d2, values.len()).unwrap();
        assert_eq!(vals2, values);
    }
}
