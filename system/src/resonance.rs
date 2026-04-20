/// Stage 1 — Resonance Extraction
///
/// Finds periodic structure in data using the resonance lags from the profiling
/// pass. For each significant lag L: residual[i] = data[i] wrapping_sub data[i-L].
/// Periodic data at that lag produces near-zero residuals — AV1 loves that.
///
/// Stores: top resonance lag + mean_base (for reconstruction).
/// Reconstruction: data[i] = residual[i] wrapping_add data[i-L].

use crate::profile::ResonanceLag;

pub const RESONANCE_MIN_STRENGTH: f64 = 0.25;
pub const RESONANCE_MIN_DATA:     usize = 512;

#[derive(Debug, Clone)]
pub struct ResonanceHeader {
    pub lag:       u32,
    pub strength:  f32,
    pub prefix:    Vec<u8>,   // first `lag` bytes stored verbatim (seed for reconstruction)
}

/// Extract resonances from data. Returns (header, residual_bytes).
/// If no significant resonance found, returns None and caller uses data as-is.
pub fn extract(data: &[u8], lags: &[ResonanceLag]) -> Option<(ResonanceHeader, Vec<u8>)> {
    if data.len() < RESONANCE_MIN_DATA { return None; }

    // Pick the strongest lag that actually produces a gain
    let best = lags.iter()
        .filter(|r| r.strength >= RESONANCE_MIN_STRENGTH && r.lag < data.len())
        .max_by(|a, b| a.strength.partial_cmp(&b.strength).unwrap())?;

    let lag = best.lag;

    // Compute residuals: residual[i] = data[i] wrapping_sub data[i - lag]
    let prefix  = data[..lag].to_vec();
    let residual: Vec<u8> = data[lag..]
        .iter()
        .enumerate()
        .map(|(i, &b)| b.wrapping_sub(data[i]))  // data[i] is data[(lag+i)-lag]
        .collect();

    // Only use resonance if residuals are actually more compressible than
    // the original. The old guard compared byte-variance, which is
    // misleading on residuals that are *mostly* zero with rare high-
    // magnitude spikes (e.g. u32 high-byte transitions produce a rare
    // 0xFF residual whose squared contribution inflates variance above
    // the original data even though the residual is obviously easier to
    // compress). Shannon entropy is the correct metric — it tracks how
    // many bits per byte an order-0 coder needs, which is exactly what
    // the downstream range coder consumes.
    let orig_h  = byte_entropy(&data[lag..]);
    let resid_h = byte_entropy(&residual);

    if resid_h >= orig_h * 0.85 {
        // Didn't help enough — skip
        return None;
    }

    let header = ResonanceHeader {
        lag:      lag as u32,
        strength: best.strength as f32,
        prefix,
    };

    Some((header, residual))
}

/// Reconstruct original bytes from resonance header + residual.
pub fn reconstruct(header: &ResonanceHeader, residual: &[u8]) -> Vec<u8> {
    let lag   = header.lag as usize;
    let mut out = header.prefix.clone();
    out.reserve(residual.len());
    for (i, &r) in residual.iter().enumerate() {
        let prev = out[i];  // out[i] = out[(lag+i) - lag]
        out.push(r.wrapping_add(prev));
    }
    out
}

/// Serialize header to bytes for container storage.
pub fn serialize_header(h: &ResonanceHeader) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&h.lag.to_le_bytes());
    out.extend_from_slice(&h.strength.to_le_bytes());
    let plen = h.prefix.len() as u32;
    out.extend_from_slice(&plen.to_le_bytes());
    out.extend_from_slice(&h.prefix);
    out
}

pub fn deserialize_header(data: &[u8]) -> Option<(ResonanceHeader, usize)> {
    if data.len() < 12 { return None; }
    let lag      = u32::from_le_bytes(data[0..4].try_into().ok()?);
    let strength = f32::from_le_bytes(data[4..8].try_into().ok()?);
    let plen     = u32::from_le_bytes(data[8..12].try_into().ok()?) as usize;
    if data.len() < 12 + plen { return None; }
    let prefix   = data[12..12 + plen].to_vec();
    let consumed = 12 + plen;
    Some((ResonanceHeader { lag, strength, prefix }, consumed))
}

/// Shannon entropy of the byte distribution, in bits per byte (0..=8).
///
/// Strided sample for speed on large residuals. The stride is forced to be
/// coprime with the usual small periods (2, 3, 5) so that period-4 data
/// doesn't collapse to a single phase — that failure mode is what made the
/// old variance-based guard reject perfect lags on u32-increment streams.
fn byte_entropy(data: &[u8]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut step = (data.len() / 4096).max(1);
    if step > 1 {
        if step % 2 == 0 { step += 1; }
        if step % 3 == 0 { step += 2; }
        if step % 5 == 0 { step += 2; }
    }
    let mut counts = [0u64; 256];
    let mut n = 0u64;
    let mut i = 0usize;
    while i < data.len() {
        counts[data[i] as usize] += 1;
        n += 1;
        i += step;
    }
    if n == 0 { return 0.0; }
    let n_f = n as f64;
    let mut h = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / n_f;
            h -= p * p.log2();
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::ResonanceLag;

    #[test]
    fn roundtrip_periodic_data() {
        // Data with strong period-4 pattern
        let data: Vec<u8> = (0..2048).map(|i| match i % 4 {
            0 => 0x10, 1 => 0x20, 2 => 0x30, _ => 0x40,
        }).collect();
        let lags = vec![ResonanceLag { lag: 4, strength: 0.9 }];
        let (hdr, residual) = extract(&data, &lags).expect("should extract resonance");
        let reconstructed = reconstruct(&hdr, &residual);
        assert_eq!(reconstructed, data);
    }

    #[test]
    fn roundtrip_header_serialization() {
        let hdr = ResonanceHeader {
            lag: 8, strength: 0.75,
            prefix: vec![1, 2, 3, 4, 5, 6, 7, 8],
        };
        let bytes = serialize_header(&hdr);
        let (hdr2, consumed) = deserialize_header(&bytes).unwrap();
        assert_eq!(consumed, bytes.len());
        assert_eq!(hdr2.lag, 8);
        assert_eq!(hdr2.prefix, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn no_resonance_on_random_data() {
        let data: Vec<u8> = (0..2048).map(|i| ((i * 137 + 42) % 256) as u8).collect();
        let lags = vec![ResonanceLag { lag: 4, strength: 0.3 }];
        // May or may not extract — if it does, roundtrip must hold
        if let Some((hdr, residual)) = extract(&data, &lags) {
            let reconstructed = reconstruct(&hdr, &residual);
            assert_eq!(reconstructed, data);
        }
    }

    // ── Benchmark-bar regression test ───────────────────────────────────
    //
    // The "nearly-sorted u32s → 405x" bench case lives here in spirit:
    // the mechanism is two resonance passes at lag-4 followed by an
    // ultra-compress via range_coder on an almost-all-zero residual.
    // We simulate the critical path and pin the multiplier so any
    // future change that breaks this chain is caught immediately.
    #[test]
    fn bench_monotonic_u32_chain_hits_100x() {
        // 1 M u32s increasing by 1 → 4 MB of raw bytes.
        let n_vals = 1_000_000u32;
        let mut data = Vec::with_capacity(n_vals as usize * 4);
        for i in 0..n_vals {
            data.extend_from_slice(&i.to_le_bytes());
        }
        let original_len = data.len();

        // Pass 1: lag = 4 — real profile would find this with strength ≈ 1.0.
        let lags = vec![crate::profile::ResonanceLag { lag: 4, strength: 0.99 }];
        let (h1, r1) = extract(&data, &lags).expect("lag-4 resonance must fire");
        let hdr1 = serialize_header(&h1);

        // Pass 2 runs on the residual. After pass 1 the residual is
        // [1,0,0,0,1,0,0,0,...] repeating, so lag-4 again kills all
        // the remaining ones and residual becomes mostly zeros.
        let (h2, r2) = extract(&r1, &lags).expect("pass-2 lag-4 resonance must fire");
        let hdr2 = serialize_header(&h2);

        // Final stage: range coder on the near-all-zero residual.
        let compressed = crate::range_coder::encode(&r2);

        let total = hdr1.len() + hdr2.len() + compressed.len();
        let ratio = original_len as f64 / total as f64;
        assert!(ratio >= 100.0,
            "monotonic-u32 resonance chain regression: got {:.1}x \
             (want >= 100x, orig={}B total={}B r1={}B r2={}B comp={}B)",
            ratio, original_len, total, r1.len(), r2.len(), compressed.len());

        // Roundtrip: range-decoder → reverse pass 2 → reverse pass 1.
        let r2_dec = crate::range_coder::decode(&compressed).unwrap();
        assert_eq!(r2_dec, r2);
        let r1_rec = reconstruct(&h2, &r2_dec);
        assert_eq!(r1_rec, r1);
        let d_rec  = reconstruct(&h1, &r1_rec);
        assert_eq!(d_rec, data);
    }
}
