/// Stage 3 — Non-Local Prediction Graph
///
/// Finds similar (not necessarily identical) blocks anywhere in the file,
/// regardless of distance. LZ77 max window is 8MB. iris has no window limit.
///
/// Mechanism: MinHash fingerprints from the profiling pass index every 4KB block.
/// For each block, find its nearest neighbor by Hamming distance on fingerprints.
/// If similarity > threshold: store (ref_block_id, XOR_delta).
/// XOR delta of similar blocks is highly compressible.
///
/// This is the "distance weapon" from the roadmap — the thing LZ77 physically
/// cannot do.

pub const PRED_MIN_BLOCKS:     usize = 2;
pub const PRED_SIMILARITY_MIN: f64   = 0.55;   // Jaccard-like threshold
pub const PRED_BLOCK_SIZE:     usize = 4096;
pub const PRED_MIN_GAIN:       f64   = 0.15;   // delta must be ≥15% smaller than raw

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
pub fn build(data: &[u8], fingerprints: &[u64]) -> Vec<BlockEncoding> {
    let n = data.len();
    let n_blocks = fingerprints.len();

    if n_blocks < PRED_MIN_BLOCKS {
        return split_raw(data);
    }

    let mut encodings: Vec<BlockEncoding> = Vec::with_capacity(n_blocks);

    for block_idx in 0..n_blocks {
        let block_start = block_idx * PRED_BLOCK_SIZE;
        let block_end   = (block_start + PRED_BLOCK_SIZE).min(n);
        let block       = &data[block_start..block_end];
        let fp          = fingerprints[block_idx];

        // Find best reference among all PRECEDING blocks
        let best_ref = (0..block_idx)
            .map(|ri| {
                let sim = fingerprint_similarity(fp, fingerprints[ri]);
                (ri, sim)
            })
            .filter(|(_, sim)| *sim >= PRED_SIMILARITY_MIN)
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());

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
                encodings.push(BlockEncoding::Delta {
                    ref_block: ref_idx as u32,
                    delta,
                    len: block.len() as u32,
                });
                continue;
            }
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

/// Estimate similarity between two MinHash fingerprints.
/// Uses bit-population similarity as a Jaccard proxy.
fn fingerprint_similarity(a: u64, b: u64) -> f64 {
    let xor      = a ^ b;
    let matching = 64 - xor.count_ones() as usize;
    matching as f64 / 64.0
}

fn split_raw(data: &[u8]) -> Vec<BlockEncoding> {
    data.chunks(PRED_BLOCK_SIZE)
        .map(|c| BlockEncoding::Raw { data: c.to_vec() })
        .collect()
}
