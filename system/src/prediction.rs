/// Stage 3 — Non-Local Prediction Graph
///
/// Finds similar (not necessarily identical) blocks anywhere in the file,
/// regardless of distance. LZ77 max window is 8MB. iris has no window limit.
///
/// Mechanism: SimHash fingerprints from the profiling pass index every 4KB block.
/// For each block, find its nearest neighbor by Hamming distance on SimHash.
/// SimHash + popcount correctly approximates cosine similarity of shingle sets.
/// If similarity > threshold: store (ref_block_id, XOR_delta).
/// XOR delta of similar blocks is highly compressible.
///
/// Lookup uses a hash table on band signatures (LSH-lite) for O(n) expected time.

use std::collections::HashMap;

pub const PRED_MIN_BLOCKS:     usize = 2;
pub const PRED_SIMILARITY_MIN: f64   = 0.55;   // SimHash cosine threshold
pub const PRED_BLOCK_SIZE:     usize = 4096;
pub const PRED_MIN_GAIN:       f64   = 0.15;   // delta must be ≥15% smaller than raw

/// Number of bands for LSH lookup. Each band = 64/NUM_BANDS bits.
/// Blocks sharing any band signature are candidate neighbors.
const NUM_BANDS: usize = 4; // 4 bands × 16 bits each

/// Cap on candidates evaluated per block. Prevents a pathological LSH
/// bucket (e.g. a file with thousands of identical 4 KB runs) from
/// turning neighbor search into O(n²).
const MAX_CANDIDATES_PER_BLOCK: usize = 32;

#[derive(Debug, Clone)]
pub enum BlockEncoding {
    /// Stored as-is (incompressible or no good reference found)
    Raw { data: Vec<u8> },
    /// Encoded as XOR delta from a reference block
    Delta {
        ref_block: u32,
        delta:     Vec<u8>,   // XOR delta — zstd compressed downstream
        len:       u32,       // actual block length (last block may be short)
    },
}

/// Build prediction graph over blocks. Returns per-block encodings.
/// Uses LSH band signatures for O(n) expected-time neighbor lookup.
pub fn build(data: &[u8], fingerprints: &[u64]) -> Vec<BlockEncoding> {
    let n = data.len();
    let n_blocks = fingerprints.len();

    if n_blocks < PRED_MIN_BLOCKS {
        return split_raw(data);
    }

    // LSH index: for each band, map band_signature → list of block indices
    let mut band_tables: Vec<HashMap<u16, Vec<usize>>> =
        (0..NUM_BANDS).map(|_| HashMap::new()).collect();

    let mut encodings: Vec<BlockEncoding> = Vec::with_capacity(n_blocks);

    for block_idx in 0..n_blocks {
        let block_start = block_idx * PRED_BLOCK_SIZE;
        let block_end   = (block_start + PRED_BLOCK_SIZE).min(n);
        let block       = &data[block_start..block_end];
        let fp          = fingerprints[block_idx];

        // Collect candidate references from LSH bands. The candidate set
        // is capped so a pathological LSH bucket can't turn this loop
        // into O(n²). `seen` uses a small HashSet for O(1) dedup.
        let mut best_ref: Option<(usize, f64)> = None;
        let mut seen: std::collections::HashSet<usize> =
            std::collections::HashSet::with_capacity(MAX_CANDIDATES_PER_BLOCK);

        'cand: for band in 0..NUM_BANDS {
            let sig = band_signature(fp, band);
            if let Some(candidates) = band_tables[band].get(&sig) {
                // Iterate most-recently-inserted first — locality bias.
                for &ri in candidates.iter().rev() {
                    if !seen.insert(ri) { continue; }
                    let sim = fingerprint_similarity(fp, fingerprints[ri]);
                    if sim >= PRED_SIMILARITY_MIN {
                        if best_ref.map_or(true, |(_, s)| sim > s) {
                            best_ref = Some((ri, sim));
                        }
                    }
                    if seen.len() >= MAX_CANDIDATES_PER_BLOCK { break 'cand; }
                }
            }
        }

        if let Some((ref_idx, _sim)) = best_ref {
            let ref_start = ref_idx * PRED_BLOCK_SIZE;
            let ref_end   = (ref_start + PRED_BLOCK_SIZE).min(n);
            let ref_block = &data[ref_start..ref_end];

            // Compute XOR delta (pad shorter block with zeros)
            let delta_len = block.len().max(ref_block.len());
            let mut delta = Vec::with_capacity(delta_len);
            for i in 0..delta_len {
                let b = if i < block.len()     { block[i] }     else { 0 };
                let r = if i < ref_block.len() { ref_block[i] } else { 0 };
                delta.push(b ^ r);
            }

            // Only use delta if it's meaningfully more compressible
            let delta_entropy = crate::profile::byte_entropy(&delta);
            let block_entropy = crate::profile::byte_entropy(block);

            if block_entropy - delta_entropy >= PRED_MIN_GAIN {
                // Insert into LSH index BEFORE pushing (preceding blocks only)
                for band in 0..NUM_BANDS {
                    let sig = band_signature(fp, band);
                    band_tables[band].entry(sig).or_default().push(block_idx);
                }
                encodings.push(BlockEncoding::Delta {
                    ref_block: ref_idx as u32,
                    delta,
                    len: block.len() as u32,
                });
                continue;
            }
        }

        // Insert into LSH index for future blocks to find
        for band in 0..NUM_BANDS {
            let sig = band_signature(fp, band);
            band_tables[band].entry(sig).or_default().push(block_idx);
        }

        // No good reference — store raw
        encodings.push(BlockEncoding::Raw { data: block.to_vec() });
    }

    encodings
}

/// Reconstruct original data from block encodings.
pub fn reconstruct(encodings: &[BlockEncoding]) -> Vec<u8> {
    let mut decoded_blocks: Vec<Vec<u8>> = Vec::with_capacity(encodings.len());

    for enc in encodings {
        match enc {
            BlockEncoding::Raw { data } => {
                decoded_blocks.push(data.clone());
            }
            BlockEncoding::Delta { ref_block, delta, len } => {
                let ref_data = &decoded_blocks[*ref_block as usize];
                let out_len  = *len as usize;
                let mut out  = Vec::with_capacity(out_len);
                for i in 0..out_len {
                    let r = if i < ref_data.len() { ref_data[i] } else { 0 };
                    let d = if i < delta.len()    { delta[i] }    else { 0 };
                    out.push(r ^ d);
                }
                decoded_blocks.push(out);
            }
        }
    }

    decoded_blocks.into_iter().flatten().collect()
}

/// Flatten block encodings into a single byte stream for context packing.
/// Delta blocks contribute their (more compressible) deltas;
/// Raw blocks contribute raw bytes.
pub fn flatten_for_packing(encodings: &[BlockEncoding]) -> (Vec<u8>, Vec<BlockMeta>) {
    let mut payload = Vec::new();
    let mut metas   = Vec::new();

    for (i, enc) in encodings.iter().enumerate() {
        match enc {
            BlockEncoding::Raw { data } => {
                metas.push(BlockMeta { block_idx: i as u32, is_delta: false,
                    ref_block: 0, offset: payload.len() as u32, len: data.len() as u32 });
                payload.extend_from_slice(data);
            }
            BlockEncoding::Delta { ref_block, delta, len } => {
                metas.push(BlockMeta { block_idx: i as u32, is_delta: true,
                    ref_block: *ref_block, offset: payload.len() as u32, len: *len });
                payload.extend_from_slice(delta);
            }
        }
    }

    (payload, metas)
}

#[derive(Debug, Clone)]
pub struct BlockMeta {
    pub block_idx: u32,
    pub is_delta:  bool,
    pub ref_block: u32,
    pub offset:    u32,
    pub len:       u32,
}

impl BlockMeta {
    pub fn serialize(&self) -> [u8; 17] {
        let mut out = [0u8; 17];
        out[0..4].copy_from_slice(&self.block_idx.to_le_bytes());
        out[4] = self.is_delta as u8;
        out[5..9].copy_from_slice(&self.ref_block.to_le_bytes());
        out[9..13].copy_from_slice(&self.offset.to_le_bytes());
        out[13..17].copy_from_slice(&self.len.to_le_bytes());
        out
    }
    pub fn deserialize(data: &[u8]) -> Option<Self> {
        if data.len() < 17 { return None; }
        Some(Self {
            block_idx: u32::from_le_bytes(data[0..4].try_into().ok()?),
            is_delta:  data[4] != 0,
            ref_block: u32::from_le_bytes(data[5..9].try_into().ok()?),
            offset:    u32::from_le_bytes(data[9..13].try_into().ok()?),
            len:       u32::from_le_bytes(data[13..17].try_into().ok()?),
        })
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// SimHash cosine similarity: fraction of matching bits.
/// SimHash is designed so that popcount(a XOR b) / 64 approximates
/// (1 - cosine_similarity) / 2 for the underlying shingle feature vectors.
/// We return the fraction of matching bits as the similarity score.
fn fingerprint_similarity(a: u64, b: u64) -> f64 {
    let matching = 64 - (a ^ b).count_ones() as usize;
    matching as f64 / 64.0
}

/// Extract a 16-bit band signature from a 64-bit SimHash for LSH lookup.
fn band_signature(hash: u64, band: usize) -> u16 {
    ((hash >> (band * 16)) & 0xFFFF) as u16
}

fn split_raw(data: &[u8]) -> Vec<BlockEncoding> {
    data.chunks(PRED_BLOCK_SIZE)
        .map(|c| BlockEncoding::Raw { data: c.to_vec() })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_all_raw() {
        let data = vec![7u8; PRED_BLOCK_SIZE * 3 + 100];
        let fps: Vec<u64> = (0..4).map(|i| i as u64 * 0x1234567890).collect();
        let enc = build(&data, &fps);
        let out = reconstruct(&enc);
        assert_eq!(out, data);
    }

    #[test]
    fn roundtrip_with_deltas() {
        // Two identical blocks → delta should be all zeros
        let block: Vec<u8> = (0..PRED_BLOCK_SIZE).map(|i| (i % 251) as u8).collect();
        let mut data = block.clone();
        data.extend_from_slice(&block); // identical second block
        // Slightly different third block
        let mut block3 = block.clone();
        for b in block3.iter_mut().take(100) { *b = b.wrapping_add(1); }
        data.extend_from_slice(&block3);

        let fps: Vec<u64> = data.chunks(PRED_BLOCK_SIZE)
            .map(|c| crate::pipeline::simhash_block(c))
            .collect();
        let enc = build(&data, &fps);
        let out = reconstruct(&enc);
        assert_eq!(out, data);
    }

    #[test]
    fn roundtrip_too_few_blocks() {
        let data = vec![42u8; PRED_BLOCK_SIZE / 2];
        let fps = vec![0u64];
        let enc = build(&data, &fps);
        let out = reconstruct(&enc);
        assert_eq!(out, data);
    }

    #[test]
    fn flatten_roundtrip() {
        let data = vec![0xABu8; PRED_BLOCK_SIZE * 2];
        let fps = vec![0u64, 1u64];
        let enc = build(&data, &fps);
        let (payload, metas) = flatten_for_packing(&enc);
        // Verify metas can serialize/deserialize
        for m in &metas {
            let bytes = m.serialize();
            let m2 = BlockMeta::deserialize(&bytes).unwrap();
            assert_eq!(m.block_idx, m2.block_idx);
            assert_eq!(m.is_delta, m2.is_delta);
            assert_eq!(m.len, m2.len);
        }
        assert!(!payload.is_empty());
    }

    #[test]
    fn simhash_identical_blocks_match() {
        let block: Vec<u8> = (0..PRED_BLOCK_SIZE).map(|i| (i * 7 % 256) as u8).collect();
        let fp1 = crate::pipeline::simhash_block(&block);
        let fp2 = crate::pipeline::simhash_block(&block);
        assert_eq!(fp1, fp2);
        assert_eq!(fingerprint_similarity(fp1, fp2), 1.0);
    }

    #[test]
    fn simhash_different_blocks_diverge() {
        let block1: Vec<u8> = (0..PRED_BLOCK_SIZE).map(|i| (i % 256) as u8).collect();
        let block2: Vec<u8> = (0..PRED_BLOCK_SIZE).map(|i| ((i * 137 + 42) % 256) as u8).collect();
        let fp1 = crate::pipeline::simhash_block(&block1);
        let fp2 = crate::pipeline::simhash_block(&block2);
        let sim = fingerprint_similarity(fp1, fp2);
        assert!(sim < 0.9, "expected different blocks to have low similarity, got {}", sim);
    }
}
