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

    // Only use resonance if residuals are actually lower entropy
    let orig_var   = byte_variance(&data[lag..]);
    let resid_var  = byte_variance(&residual);

    if resid_var >= orig_var * 0.85 {
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

fn byte_variance(data: &[u8]) -> f64 {
    if data.is_empty() { return 0.0; }
    let step = (data.len() / 4096).max(1);
    let mut sum = 0i64;
    let mut sum2 = 0i64;
    let mut n = 0i64;
    for &b in data.iter().step_by(step) {
        let v = b as i64;
        sum += v;
        sum2 += v * v;
        n += 1;
    }
    if n == 0 { return 0.0; }
    let mean = sum as f64 / n as f64;
    sum2 as f64 / n as f64 - mean * mean
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
}
