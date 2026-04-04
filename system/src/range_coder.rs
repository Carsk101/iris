/// Stage 5a — Native Range Coder
///
/// Replaces bzip2 on the ultra-compress path (H < 0.15).
/// After resonance extraction, the residual has very few unique byte values
/// with known frequencies. A range coder encodes to theoretical minimum.
///
/// Format:
///   [256 × u32-LE frequency table] (1024 bytes)
///   [u32-LE original length]
///   [encoded byte stream]
///
/// The frequency table is exact for this specific residual — no generic model.

const BOTTOM: u64 = 1 << 16;

// ─── Frequency Table ─────────────────────────────────────────────────────────

#[derive(Clone)]
struct FreqTable {
    freq: [u32; 256],
    cum:  [u32; 257],  // cumulative: cum[i] = sum of freq[0..i]
    total: u32,
}

impl FreqTable {
    fn from_data(data: &[u8]) -> Self {
        let mut freq = [0u32; 256];
        for &b in data {
            freq[b as usize] += 1;
        }
        // Ensure every symbol has at least frequency 1 for robustness
        // (range coder requires non-zero frequencies for all possible symbols
        //  that might appear during decode)
        for f in &mut freq {
            if *f == 0 { *f = 1; }
        }
        let mut cum = [0u32; 257];
        for i in 0..256 {
            cum[i + 1] = cum[i] + freq[i];
        }
        let total = cum[256];
        Self { freq, cum, total }
    }

    fn from_stored(freq: [u32; 256]) -> Self {
        let mut cum = [0u32; 257];
        for i in 0..256 {
            cum[i + 1] = cum[i] + freq[i];
        }
        let total = cum[256];
        Self { freq, cum, total }
    }

    /// Find symbol for a given cumulative count value
    fn symbol_for_count(&self, count: u32) -> u8 {
        // Binary search for the symbol whose cumulative range contains count
        let mut lo: usize = 0;
        let mut hi: usize = 256;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if self.cum[mid] <= count {
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
    let mut out = Vec::with_capacity(data.len() + 1028);

    // Write frequency table (256 × u32-LE = 1024 bytes)
    for i in 0..256 {
        out.extend_from_slice(&table.freq[i].to_le_bytes());
    }

    // Write original length
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());

    // Range encode
    let mut low: u64 = 0;
    let mut range: u64 = u32::MAX as u64;

    for &byte in data {
        let sym = byte as usize;
        let sym_low  = table.cum[sym] as u64;
        let sym_high = table.cum[sym + 1] as u64;
        let total    = table.total as u64;

        range /= total;
        low   += sym_low * range;
        range *= sym_high - sym_low;

        // Normalise: emit bytes when top bits are settled
        while range < BOTTOM {
            out.push((low >> 24) as u8);
            low = (low << 8) & 0xFFFF_FFFF;
            range <<= 8;
            // If range collapses, force it to BOTTOM
            if range < BOTTOM {
                // continue the loop
            }
        }
    }

    // Flush remaining state (4 bytes)
    out.push((low >> 24) as u8);
    out.push((low >> 16) as u8);
    out.push((low >> 8) as u8);
    out.push(low as u8);

    out
}

// ─── Decoder ─────────────────────────────────────────────────────────────────

pub fn decode(compressed: &[u8]) -> Option<Vec<u8>> {
    if compressed.is_empty() {
        return Some(vec![]);
    }
    if compressed.len() < 1028 {
        return None; // need at least freq table + length
    }

    // Read frequency table
    let mut freq = [0u32; 256];
    for i in 0..256 {
        let off = i * 4;
        freq[i] = u32::from_le_bytes(
            compressed[off..off + 4].try_into().ok()?
        );
    }
    let table = FreqTable::from_stored(freq);
    if table.total == 0 { return None; }

    // Read original length
    let orig_len = u32::from_le_bytes(
        compressed[1024..1028].try_into().ok()?
    ) as usize;

    if orig_len == 0 {
        return Some(vec![]);
    }

    let stream = &compressed[1028..];

    // Initialise decoder state
    let mut code: u64 = 0;
    let mut pos = 0usize;
    for _ in 0..4 {
        code = (code << 8) | (*stream.get(pos).unwrap_or(&0)) as u64;
        pos += 1;
    }

    let mut low: u64 = 0;
    let mut range: u64 = u32::MAX as u64;
    let mut out = Vec::with_capacity(orig_len);

    for _ in 0..orig_len {
        let total = table.total as u64;
        range /= total;

        let count = ((code - low) / range) as u32;
        let count = count.min(table.total - 1);
        let sym = table.symbol_for_count(count);

        let sym_low  = table.cum[sym as usize] as u64;
        let sym_high = table.cum[sym as usize + 1] as u64;

        low   += sym_low * range;
        range *= sym_high - sym_low;

        // Normalise
        while range < BOTTOM {
            code = (code << 8) | (*stream.get(pos).unwrap_or(&0)) as u64;
            code &= 0xFFFF_FFFF;
            pos += 1;
            low = (low << 8) & 0xFFFF_FFFF;
            range <<= 8;
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
        // Simulate post-resonance residual: mostly zeros with a few values
        let mut data = vec![0u8; 5000];
        for i in (0..5000).step_by(7) { data[i] = 1; }
        for i in (0..5000).step_by(13) { data[i] = 2; }
        for i in (0..5000).step_by(37) { data[i] = 255; }
        let enc = encode(&data);
        let dec = decode(&enc).unwrap();
        assert_eq!(dec, data);
        // Should compress well — much smaller than input
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
        // Very skewed distribution — range coder should approach entropy limit
        let mut data = vec![0u8; 10_000];
        data[500] = 1;
        data[3000] = 2;
        let enc = encode(&data);
        // 1024 header + 4 len + very few encoded bytes
        let payload = enc.len() - 1028;
        // Theoretical entropy ≈ 0.003 bits/byte → ~4 bytes for 10k input
        assert!(payload < 100,
            "expected near-zero payload, got {} bytes", payload);
    }
}
