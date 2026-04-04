/// Stage 5b — Context-Conditioned rANS Coder
///
/// Replaces zstd on the context-packed flat buffer.
/// After context grouping, LZ77 finds almost nothing useful — what's left is
/// per-symbol entropy coding. rANS (range Asymmetric Numeral Systems) is faster
/// to decode than arithmetic coding and hits the same compression ratio.
///
/// Architecture:
///   - Per-context frequency tables (up to 256 contexts × 256 symbols)
///   - Tables normalised to power-of-two totals for fast decode (shift instead of divide)
///   - Encode forward, decode backward (rANS natural order)
///   - Interleaved contexts: each byte uses prev byte as context (order-1 model)
///
/// Format:
///   [u32-LE original_len]
///   [u32-LE num_contexts]
///   For each context:
///     [u8 context_byte]
///     [256 × u16-LE normalised frequencies]  (512 bytes per context)
///   [u32-LE encoded_len]
///   [encoded bytes — rANS state flushed big-endian]

const RANS_SCALE_BITS: u32 = 12;
const RANS_SCALE: u32      = 1 << RANS_SCALE_BITS;  // 4096
const RANS_LOWER: u32      = 1 << 23;   // renorm threshold

// ─── Normalised frequency table for one context ─────────────────────────────

#[derive(Clone)]
struct ContextFreqs {
    freq: [u16; 256],      // normalised to sum = RANS_SCALE
    cum:  [u16; 257],      // cumulative frequencies
}

impl ContextFreqs {
    fn from_raw(raw: &[u32; 256]) -> Self {
        let total: u64 = raw.iter().map(|&f| f as u64).sum();
        let mut freq = [0u16; 256];

        if total == 0 {
            // Uniform — shouldn't happen but be safe
            freq.iter_mut().for_each(|f| *f = (RANS_SCALE / 256) as u16);
            // Fix rounding
            let sum: u32 = freq.iter().map(|&f| f as u32).sum();
            freq[0] = (freq[0] as u32 + RANS_SCALE - sum) as u16;
        } else {
            // Proportional normalisation with correction
            let mut assigned = 0u32;
            let mut max_idx = 0usize;
            let mut max_val = 0u32;

            for i in 0..256 {
                if raw[i] > 0 {
                    // At least 1 for any symbol that appeared
                    let f = ((raw[i] as u64 * RANS_SCALE as u64) / total).max(1) as u32;
                    freq[i] = f as u16;
                    assigned += f;
                    if raw[i] > max_val {
                        max_val = raw[i];
                        max_idx = i;
                    }
                }
            }

            // Correct rounding error on the most frequent symbol
            if assigned > RANS_SCALE {
                let excess = assigned - RANS_SCALE;
                if (freq[max_idx] as u32) > excess {
                    freq[max_idx] -= excess as u16;
                }
            } else if assigned < RANS_SCALE {
                freq[max_idx] += (RANS_SCALE - assigned) as u16;
            }
        }

        let mut cum = [0u16; 257];
        for i in 0..256 {
            cum[i + 1] = cum[i] + freq[i];
        }

        Self { freq, cum }
    }

    fn from_stored(freq: [u16; 256]) -> Self {
        let mut cum = [0u16; 257];
        for i in 0..256 {
            cum[i + 1] = cum[i] + freq[i];
        }
        Self { freq, cum }
    }

    /// Lookup: given a cumulative value, find the symbol
    fn symbol_for_cum(&self, value: u16) -> u8 {
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

// ─── Order-1 model (context = previous byte) ────────────────────────────────

struct Order1Model {
    contexts: Vec<(u8, ContextFreqs)>,  // only contexts that appear
    ctx_index: [u16; 256],              // context_byte → index into contexts (0xFFFF = absent)
}

impl Order1Model {
    fn build(data: &[u8]) -> Self {
        if data.is_empty() {
            return Self { contexts: vec![], ctx_index: [0xFFFF; 256] };
        }

        let mut raw_counts = [[0u32; 256]; 256];  // raw_counts[prev][cur]
        let mut prev = 0u8;  // context for first byte is 0
        for &b in data {
            raw_counts[prev as usize][b as usize] += 1;
            prev = b;
        }

        let mut contexts = Vec::new();
        let mut ctx_index = [0xFFFF_u16; 256];

        for ctx in 0..256 {
            let total: u32 = raw_counts[ctx].iter().sum();
            if total > 0 {
                ctx_index[ctx] = contexts.len() as u16;
                contexts.push((ctx as u8, ContextFreqs::from_raw(&raw_counts[ctx])));
            }
        }

        Self { contexts, ctx_index }
    }

    fn get(&self, ctx: u8) -> Option<&ContextFreqs> {
        let idx = self.ctx_index[ctx as usize];
        if idx == 0xFFFF { None }
        else { Some(&self.contexts[idx as usize].1) }
    }
}

// ─── Encoder ─────────────────────────────────────────────────────────────────

pub fn encode(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        let mut out = Vec::new();
        out.extend_from_slice(&0u32.to_le_bytes());
        return out;
    }

    let model = Order1Model::build(data);

    // rANS encode — process data in reverse, output will be reversed at end
    let mut rans_state: u32 = RANS_LOWER;
    let mut encoded_bytes: Vec<u8> = Vec::with_capacity(data.len());

    // Build (context, symbol) pairs forward, then encode backward
    let mut pairs: Vec<(u8, u8)> = Vec::with_capacity(data.len());
    let mut prev = 0u8;
    for &b in data {
        pairs.push((prev, b));
        prev = b;
    }

    // Encode in reverse order
    for &(ctx, sym) in pairs.iter().rev() {
        let freqs = model.get(ctx).unwrap();
        let f = freqs.freq[sym as usize] as u32;
        let c = freqs.cum[sym as usize] as u32;

        // Renormalise: push bytes out when state gets too large
        let max_state = ((RANS_LOWER >> RANS_SCALE_BITS) << 8) * f;
        while rans_state >= max_state {
            encoded_bytes.push((rans_state & 0xFF) as u8);
            rans_state >>= 8;
        }

        // rANS encode step: x' = (x / f) * M + (x % f) + c
        rans_state = ((rans_state / f) << RANS_SCALE_BITS)
                   + (rans_state % f) + c;
    }

    // Flush final state (4 bytes, big-endian for decode convenience)
    encoded_bytes.push((rans_state & 0xFF) as u8); rans_state >>= 8;
    encoded_bytes.push((rans_state & 0xFF) as u8); rans_state >>= 8;
    encoded_bytes.push((rans_state & 0xFF) as u8); rans_state >>= 8;
    encoded_bytes.push((rans_state & 0xFF) as u8);

    // Reverse — decode reads from front
    encoded_bytes.reverse();

    // Serialize
    let mut out = Vec::with_capacity(4 + 4 + model.contexts.len() * 513 + 4 + encoded_bytes.len());

    // Original length
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());

    // Context tables — sparse encoding: only store non-zero freq symbols
    // Format per context: [ctx_byte] [n_nonzero: u16-LE] [sym, freq_u16-LE]×n_nonzero
    out.extend_from_slice(&(model.contexts.len() as u32).to_le_bytes());
    for &(ctx_byte, ref freqs) in &model.contexts {
        out.push(ctx_byte);
        let nonzero: Vec<(u8, u16)> = (0..256)
            .filter(|&i| freqs.freq[i] > 0)
            .map(|i| (i as u8, freqs.freq[i]))
            .collect();
        out.extend_from_slice(&(nonzero.len() as u16).to_le_bytes());
        for &(sym, f) in &nonzero {
            out.push(sym);
            out.extend_from_slice(&f.to_le_bytes());
        }
    }

    // Encoded stream
    out.extend_from_slice(&(encoded_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&encoded_bytes);

    out
}

// ─── Decoder ─────────────────────────────────────────────────────────────────

pub fn decode(compressed: &[u8]) -> Option<Vec<u8>> {
    if compressed.len() < 4 { return None; }

    let mut p = 0usize;

    // Original length
    let orig_len = read_u32(compressed, &mut p)? as usize;
    if orig_len == 0 { return Some(vec![]); }

    // Context tables — sparse encoding
    let n_contexts = read_u32(compressed, &mut p)? as usize;
    let mut ctx_index = [0xFFFF_u16; 256];
    let mut contexts: Vec<(u8, ContextFreqs)> = Vec::with_capacity(n_contexts);

    for i in 0..n_contexts {
        if p >= compressed.len() { return None; }
        let ctx_byte = compressed[p]; p += 1;
        if p + 2 > compressed.len() { return None; }
        let n_nonzero = u16::from_le_bytes([compressed[p], compressed[p + 1]]) as usize;
        p += 2;
        let mut freq = [0u16; 256];
        for _ in 0..n_nonzero {
            if p + 3 > compressed.len() { return None; }
            let sym = compressed[p]; p += 1;
            let f = u16::from_le_bytes([compressed[p], compressed[p + 1]]);
            p += 2;
            freq[sym as usize] = f;
        }
        ctx_index[ctx_byte as usize] = i as u16;
        contexts.push((ctx_byte, ContextFreqs::from_stored(freq)));
    }

    // Encoded stream
    let enc_len = read_u32(compressed, &mut p)? as usize;
    if p + enc_len > compressed.len() { return None; }
    let stream = &compressed[p..p + enc_len];

    // Initialise rANS state from first 4 bytes
    let mut sp = 0usize;
    let mut rans_state: u32 = 0;
    for _ in 0..4 {
        rans_state = (rans_state << 8) | (*stream.get(sp).unwrap_or(&0)) as u32;
        sp += 1;
    }

    // Decode forward
    let mut out = Vec::with_capacity(orig_len);
    let mut prev = 0u8;

    for _ in 0..orig_len {
        let ci = ctx_index[prev as usize];
        if ci == 0xFFFF { return None; }
        let freqs = &contexts[ci as usize].1;

        // rANS decode step
        let cum_value = (rans_state & (RANS_SCALE - 1)) as u16;
        let sym = freqs.symbol_for_cum(cum_value);
        let f = freqs.freq[sym as usize] as u32;
        let c = freqs.cum[sym as usize] as u32;

        // Update state: x' = f * (x >> scale_bits) + (x & mask) - c
        rans_state = f * (rans_state >> RANS_SCALE_BITS)
                   + (rans_state & (RANS_SCALE - 1)) - c;

        // Renormalise
        while rans_state < RANS_LOWER {
            rans_state = (rans_state << 8) | (*stream.get(sp).unwrap_or(&0)) as u32;
            sp += 1;
        }

        out.push(sym);
        prev = sym;
    }

    Some(out)
}

fn read_u32(data: &[u8], p: &mut usize) -> Option<u32> {
    if *p + 4 > data.len() { return None; }
    let v = u32::from_le_bytes(data[*p..*p + 4].try_into().ok()?);
    *p += 4;
    Some(v)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_simple() {
        let data = b"hello world hello world hello world";
        let enc = encode(data);
        let dec = decode(&enc).unwrap();
        assert_eq!(dec, data.to_vec());
    }

    #[test]
    fn roundtrip_single_byte_repeated() {
        let data = vec![42u8; 5000];
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
    fn roundtrip_empty() {
        let enc = encode(&[]);
        let dec = decode(&enc).unwrap();
        assert!(dec.is_empty());
    }

    #[test]
    fn roundtrip_context_packed_like_data() {
        // Simulate what the context-packed flat buffer looks like:
        // runs of similar bytes grouped by context
        let mut data = Vec::with_capacity(10_000);
        // Context 0x20 (space): mostly lowercase letters
        for i in 0..2000 { data.push(b'a' + (i % 26) as u8); }
        // Context 0x0A (newline): mostly uppercase
        for i in 0..2000 { data.push(b'A' + (i % 26) as u8); }
        // Context 0x2C (comma): mostly digits
        for i in 0..2000 { data.push(b'0' + (i % 10) as u8); }
        // Some random
        for i in 0..1000 { data.push((i * 137 % 256) as u8); }

        let enc = encode(&data);
        let dec = decode(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_binary_pattern() {
        // Binary data with structure
        let data: Vec<u8> = (0..20_000)
            .map(|i| ((i * 7 + (i / 256) * 13) % 256) as u8)
            .collect();
        let enc = encode(&data);
        let dec = decode(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_two_bytes() {
        let data = vec![0xAA, 0xBB];
        let enc = encode(&data);
        let dec = decode(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_one_byte() {
        let data = vec![0xFF];
        let enc = encode(&data);
        let dec = decode(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_large_structured() {
        // Large file with strong context structure — the real use case
        let mut data = Vec::with_capacity(50_000);
        for _ in 0..5000 {
            data.extend_from_slice(b"GET /api/v1");
        }
        for _ in 0..5000 {
            data.extend_from_slice(b"200");
        }
        let enc = encode(&data);
        let dec = decode(&enc).unwrap();
        assert_eq!(dec, data);
    }
}
