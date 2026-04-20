/// Stage 5a — Order-0 entropy coder.
///
/// Historically this was a true range coder (carry-propagating arithmetic
/// coder). That implementation had two bugs that only showed up at scale:
///
///   1. Raw byte counts were used directly as the probability denominator,
///      so on inputs above 2^16 bytes `range /= total` could collapse the
///      working range to zero and loop forever.
///   2. `low` was masked to 32 bits on every renormalisation, discarding
///      carries. That silently corrupted output when a symbol with a high
///      cumulative position (e.g. 0xFF in a skewed histogram) bumped `low`
///      past 2^32.
///
/// The zero-dependency, numerically robust replacement is order-0 rANS.
/// It has the same public API (`encode` / `decode` on `&[u8]`) and the
/// same semantic role (cheap order-0 coder for highly skewed byte streams,
/// e.g. post-resonance residuals or varint columns), but cannot get stuck
/// and cannot drop carries.
///
/// Format:
///   [u16-LE n_nonzero]
///   n_nonzero × ( [u8 sym] [u16-LE norm_freq] )   (freqs sum to TOTAL_SCALE)
///   [u32-LE original length]
///   [u32-LE encoded stream length]
///   [encoded stream bytes]
///
/// The encoded stream is a normal rANS forward stream (encode in reverse,
/// store reversed so the decoder reads forwards).

const TOTAL_BITS:  u32 = 14;
const TOTAL_SCALE: u32 = 1 << TOTAL_BITS;   // 16384
const RANS_LOWER:  u32 = 1 << 23;           // renormalisation threshold

#[derive(Clone)]
struct FreqTable {
    freq: [u32; 256],
    cum:  [u32; 257],
    total: u32,
}

impl FreqTable {
    fn from_data(data: &[u8]) -> Self {
        let mut raw = [0u32; 256];
        for &b in data {
            raw[b as usize] += 1;
        }
        Self::from_raw(&raw)
    }

    /// Normalise raw counts to `TOTAL_SCALE`. Every symbol with a non-zero
    /// raw count receives a normalised frequency of at least 1.
    fn from_raw(raw: &[u32; 256]) -> Self {
        let total_raw: u64 = raw.iter().map(|&f| f as u64).sum();
        let mut freq = [0u32; 256];

        if total_raw == 0 {
            // Never encoded; keep the invariant sum(freq) == TOTAL_SCALE
            // so `from_stored` works even if someone constructs an empty
            // table defensively.
            for f in freq.iter_mut() { *f = TOTAL_SCALE / 256; }
            let sum: u32 = freq.iter().sum();
            freq[255] += TOTAL_SCALE - sum;
        } else {
            let mut assigned = 0u32;
            let mut max_idx = 0usize;
            let mut max_val = 0u32;
            for i in 0..256 {
                if raw[i] > 0 {
                    let f = ((raw[i] as u64 * TOTAL_SCALE as u64) / total_raw).max(1) as u32;
                    freq[i] = f;
                    assigned += f;
                    if raw[i] > max_val {
                        max_val = raw[i];
                        max_idx = i;
                    }
                }
            }
            if assigned > TOTAL_SCALE {
                let excess = assigned - TOTAL_SCALE;
                let new_f = freq[max_idx].saturating_sub(excess).max(1);
                let delta = freq[max_idx] - new_f;
                freq[max_idx] = new_f;
                assigned -= delta;
                // Theoretical edge case: one symbol wasn't enough. Shave
                // the rest from any freq > 1 until we hit TOTAL_SCALE.
                let mut i = 0;
                while assigned > TOTAL_SCALE && i < 256 {
                    if i != max_idx && freq[i] > 1 {
                        freq[i] -= 1;
                        assigned -= 1;
                    }
                    i += 1;
                }
            } else if assigned < TOTAL_SCALE {
                freq[max_idx] += TOTAL_SCALE - assigned;
            }
        }

        let mut cum = [0u32; 257];
        for i in 0..256 {
            cum[i + 1] = cum[i] + freq[i];
        }
        debug_assert_eq!(cum[256], TOTAL_SCALE);
        Self { freq, cum, total: TOTAL_SCALE }
    }

    fn from_stored(freq: [u32; 256]) -> Self {
        let mut cum = [0u32; 257];
        for i in 0..256 {
            cum[i + 1] = cum[i] + freq[i];
        }
        let total = cum[256];
        Self { freq, cum, total }
    }

    fn symbol_for_cum(&self, value: u32) -> u8 {
        let mut lo = 0usize;
        let mut hi = 256usize;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if self.cum[mid] <= value {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo as u8
    }
}

// ─── Encoder ─────────────────────────────────────────────────────────────────

pub fn encode(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return vec![];
    }

    let table = FreqTable::from_data(data);

    // rANS encode in reverse order. State grows as we consume symbols; bytes
    // are flushed out whenever state exceeds the renormalisation band.
    let mut state: u32 = RANS_LOWER;
    let mut encoded: Vec<u8> = Vec::with_capacity(data.len() / 2 + 16);

    for i in (0..data.len()).rev() {
        let sym = data[i] as usize;
        let f = table.freq[sym];
        // Invariant: `f > 0` because every byte in `data` was counted and
        // `from_raw` gives non-zero symbols a floor of 1 after normalisation.
        debug_assert!(f > 0);

        // Renormalise: push the low byte out while state would overflow.
        // max_state = ((RANS_LOWER >> TOTAL_BITS) << 8) * f.  With
        // TOTAL_BITS = 14 and f >= 1, max_state >= 512, so state (u32) is
        // guaranteed to drop below it after at most four right-shifts.
        let max_state = ((RANS_LOWER >> TOTAL_BITS) << 8) * f;
        while state >= max_state {
            encoded.push((state & 0xFF) as u8);
            state >>= 8;
        }

        // rANS forward: x' = (x / f) * TOTAL + (x % f) + c
        state = ((state / f) << TOTAL_BITS) + (state % f) + table.cum[sym];
    }

    // Flush final 4 bytes of state.
    encoded.push((state & 0xFF) as u8); state >>= 8;
    encoded.push((state & 0xFF) as u8); state >>= 8;
    encoded.push((state & 0xFF) as u8); state >>= 8;
    encoded.push((state & 0xFF) as u8);

    // Decoder reads from front, so reverse now.
    encoded.reverse();

    // Assemble output.
    let mut out = Vec::with_capacity(3 + 256 * 3 + 4 + 4 + encoded.len());

    let mut non_zero: Vec<(u8, u16)> = Vec::new();
    for i in 0..256 {
        if table.freq[i] > 0 {
            non_zero.push((i as u8, table.freq[i] as u16));
        }
    }
    out.extend_from_slice(&(non_zero.len() as u16).to_le_bytes());
    for (sym, f) in &non_zero {
        out.push(*sym);
        out.extend_from_slice(&f.to_le_bytes());
    }

    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
    out.extend_from_slice(&encoded);

    out
}

// ─── Decoder ─────────────────────────────────────────────────────────────────

pub fn decode(compressed: &[u8]) -> Option<Vec<u8>> {
    if compressed.is_empty() {
        return Some(vec![]);
    }
    if compressed.len() < 10 {
        return None;
    }

    let mut p = 0usize;
    let count = u16::from_le_bytes(compressed[p..p + 2].try_into().ok()?) as usize;
    p += 2;

    if compressed.len() < p + count * 3 + 8 {
        return None;
    }

    let mut freq = [0u32; 256];
    for _ in 0..count {
        let sym = compressed[p];
        p += 1;
        let f = u16::from_le_bytes(compressed[p..p + 2].try_into().ok()?) as u32;
        p += 2;
        freq[sym as usize] = f;
    }
    let table = FreqTable::from_stored(freq);
    if table.total != TOTAL_SCALE {
        return None;
    }

    let orig_len = u32::from_le_bytes(compressed[p..p + 4].try_into().ok()?) as usize;
    p += 4;
    if orig_len == 0 {
        return Some(vec![]);
    }

    let enc_len = u32::from_le_bytes(compressed[p..p + 4].try_into().ok()?) as usize;
    p += 4;
    if compressed.len() < p + enc_len {
        return None;
    }
    let stream = &compressed[p..p + enc_len];

    let mut sp = 0usize;
    let mut state: u32 = 0;
    for _ in 0..4 {
        state = (state << 8) | (*stream.get(sp).unwrap_or(&0)) as u32;
        sp += 1;
    }

    let mut out = Vec::with_capacity(orig_len);
    for _ in 0..orig_len {
        // rANS decode: find symbol whose cumulative range contains the
        // bottom TOTAL_BITS of state.
        let cum_value = state & (TOTAL_SCALE - 1);
        let sym = table.symbol_for_cum(cum_value);
        let f = table.freq[sym as usize];
        let c = table.cum[sym as usize];

        if f == 0 {
            return None;
        }

        state = f * (state >> TOTAL_BITS) + (state & (TOTAL_SCALE - 1)) - c;

        while state < RANS_LOWER {
            state = (state << 8) | (*stream.get(sp).unwrap_or(&0)) as u32;
            sp += 1;
        }

        out.push(sym);
    }

    Some(out)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_simple() {
        let data = b"hello world hello world hello";
        let enc = encode(data);
        let dec = decode(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_single_byte() {
        let data = vec![42u8; 1000];
        let enc = encode(&data);
        let dec = decode(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_all_bytes() {
        let data: Vec<u8> = (0..=255).collect();
        let enc = encode(&data);
        let dec = decode(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_low_entropy_residual() {
        let mut data = vec![0u8; 5000];
        for i in (0..5000).step_by(7)  { data[i] = 1; }
        for i in (0..5000).step_by(13) { data[i] = 2; }
        for i in (0..5000).step_by(37) { data[i] = 255; }
        let enc = encode(&data);
        let dec = decode(&enc).unwrap();
        assert_eq!(dec, data);
        assert!(enc.len() < data.len() / 2,
            "expected good compression, got {} -> {}", data.len(), enc.len());
    }

    #[test]
    fn roundtrip_empty() {
        let enc = encode(&[]);
        let dec = decode(&enc).unwrap();
        assert!(dec.is_empty());
    }

    #[test]
    fn roundtrip_large_repeated_pattern() {
        let pattern = [0u8, 0, 0, 0, 1, 0, 0, 2];
        let data: Vec<u8> = pattern.iter().cycle().take(50_000).cloned().collect();
        let enc = encode(&data);
        let dec = decode(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn compression_beats_input_on_skewed() {
        let mut data = vec![0u8; 10_000];
        data[500]  = 1;
        data[3000] = 2;
        let enc = encode(&data);
        let payload = enc.len();
        assert!(payload < 100,
            "expected near-zero payload, got {} bytes", payload);
    }

    #[test]
    fn roundtrip_large_text_does_not_hang() {
        // Regression: old range coder hung when data size exceeded BOTTOM
        // (= 2^16) because raw counts drove `range /= total` to zero.
        let mut data = Vec::new();
        for _ in 0..4000 {
            data.extend_from_slice(b"the quick brown fox jumps over the lazy dog\n");
        }
        assert!(data.len() > (1 << 16));
        let enc = encode(&data);
        assert!(enc.len() < data.len(), "should compress English text");
        let dec = decode(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_high_byte_heavy() {
        // Regression: old range coder dropped carries when the active symbol
        // had a high cumulative position (close to total), so 0xFF-heavy
        // data decoded incorrectly. This test exercises that path.
        let mut data = vec![0u8; 2000];
        for i in (0..2000).step_by(3)  { data[i] = 0xFF; }
        for i in (0..2000).step_by(11) { data[i] = 0xFE; }
        let enc = encode(&data);
        let dec = decode(&enc).unwrap();
        assert_eq!(dec, data);
    }
}
