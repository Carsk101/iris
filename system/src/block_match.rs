/// Stage 5c — Native Hierarchical Block Matcher
///
/// Replaces ffmpeg/AV1 for the high-entropy route.
/// What iris needs from AV1 is two things: motion search (find reference blocks)
/// and residual coding (encode the difference). This module does both natively.
///
/// Architecture:
///   - Hierarchical block matching: try 4KB blocks first (like Stage 3),
///     then subdivide to 1KB and 256B if XOR residual entropy is still high
///   - Residual coding: XOR delta + rANS from Stage 5b
///   - No YUV, no frames, no ffmpeg — just block matching and entropy coding
///
/// Format:
///   [u32-LE original_len]
///   [u32-LE num_blocks]
///   For each block:
///     [u8 block_type]  — 0=raw, 1=delta_4k, 2=delta_1k, 3=delta_256
///     [u16-LE block_len]
///     type-specific payload
///   [rANS-encoded residual stream]

use std::collections::HashMap;

const BLOCK_4K:  usize = 4096;
const BLOCK_1K:  usize = 1024;
const BLOCK_256: usize = 256;

const MATCH_THRESHOLD: f64 = 0.55;   // SimHash similarity threshold
const SUBDIVIDE_ENTROPY: f64 = 0.70; // subdivide if delta entropy above this
const DELTA_MIN_GAIN: f64 = 0.10;    // delta must save at least 10% entropy

// Block types in the encoded stream
const BT_RAW:      u8 = 0;
const BT_DELTA_4K: u8 = 1;
const BT_DELTA_1K: u8 = 2;
const BT_DELTA_256: u8 = 3;

#[derive(Debug, Clone)]
struct EncodedBlock {
    block_type: u8,
    data: Vec<u8>,     // raw bytes or XOR delta
    ref_idx: u32,      // reference block index (for delta types)
    sub_blocks: Vec<EncodedSubBlock>, // subdivided blocks within this 4KB region
}

#[derive(Debug, Clone)]
struct EncodedSubBlock {
    block_type: u8,
    offset: u16,       // offset within the parent 4KB block
    len: u16,
    data: Vec<u8>,     // raw or XOR delta
    ref_idx: u32,      // reference: parent_block_idx * (4096/sub_size) + sub_idx
}

/// Compress data using hierarchical block matching + rANS.
/// This is the native replacement for the AV1 route.
pub fn compress(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return vec![];
    }

    let n = data.len();
    let n_blocks_4k = (n + BLOCK_4K - 1) / BLOCK_4K;

    // Compute SimHash fingerprints at all three resolutions
    let fps_4k = compute_fingerprints(data, BLOCK_4K);

    // LSH index for 4KB blocks
    let mut lsh_4k: Vec<HashMap<u16, Vec<usize>>> =
        (0..4).map(|_| HashMap::new()).collect();

    let mut all_blocks: Vec<EncodedBlock> = Vec::with_capacity(n_blocks_4k);

    for bi in 0..n_blocks_4k {
        let b_start = bi * BLOCK_4K;
        let b_end = (b_start + BLOCK_4K).min(n);
        let block = &data[b_start..b_end];
        let fp = fps_4k[bi];

        // Try to find a 4KB match
        let best_4k = find_match(&lsh_4k, fp, &fps_4k, bi);

        if let Some((ref_idx, _sim)) = best_4k {
            let ref_start = ref_idx * BLOCK_4K;
            let ref_end = (ref_start + BLOCK_4K).min(n);
            let ref_block = &data[ref_start..ref_end];

            let delta = xor_delta(block, ref_block);
            let delta_h = byte_entropy(&delta);
            let block_h = byte_entropy(block);

            if block_h - delta_h >= DELTA_MIN_GAIN {
                if delta_h > SUBDIVIDE_ENTROPY && block.len() >= BLOCK_1K * 2 {
                    // Subdivide: try 1KB sub-blocks
                    let sub_blocks = subdivide_block(
                        data, block, bi, BLOCK_1K, &lsh_4k, &fps_4k, n,
                    );
                    insert_lsh(&mut lsh_4k, fp, bi);
                    all_blocks.push(EncodedBlock {
                        block_type: BT_DELTA_4K,
                        data: delta,
                        ref_idx: ref_idx as u32,
                        sub_blocks,
                    });
                    continue;
                }

                // Good 4KB delta
                insert_lsh(&mut lsh_4k, fp, bi);
                all_blocks.push(EncodedBlock {
                    block_type: BT_DELTA_4K,
                    data: delta,
                    ref_idx: ref_idx as u32,
                    sub_blocks: vec![],
                });
                continue;
            }
        }

        // No good match — store raw
        insert_lsh(&mut lsh_4k, fp, bi);
        all_blocks.push(EncodedBlock {
            block_type: BT_RAW,
            data: block.to_vec(),
            ref_idx: 0,
            sub_blocks: vec![],
        });
    }

    // Serialize to byte stream
    serialize(&all_blocks, n)
}

/// Decompress data produced by `compress`.
pub fn decompress(compressed: &[u8]) -> Option<Vec<u8>> {
    if compressed.is_empty() {
        return Some(vec![]);
    }

    let mut p = 0usize;
    let orig_len = read_u32(compressed, &mut p)? as usize;
    if orig_len == 0 { return Some(vec![]); }

    let n_blocks = read_u32(compressed, &mut p)? as usize;

    // First pass: decode block metadata
    struct BlockInfo {
        block_type: u8,
        ref_idx: u32,
        data_offset: usize,
        data_len: usize,
        n_subs: usize,
        subs: Vec<SubInfo>,
    }
    struct SubInfo {
        block_type: u8,
        offset: u16,
        len: u16,
        ref_idx: u32,
        data_offset: usize,
        data_len: usize,
    }

    let raw_len = read_u32(compressed, &mut p)? as usize;
    let payload_raw = crate::rans::decode(&compressed[p..p + raw_len])?;
    p += raw_len;

    let delta_len = read_u32(compressed, &mut p)? as usize;
    let payload_delta = crate::rans::decode(&compressed[p..p + delta_len])?;
    p += delta_len;

    // Read block descriptors
    let mut blocks: Vec<BlockInfo> = Vec::with_capacity(n_blocks);
    let mut dp_raw = 0usize;
    let mut dp_delta = 0usize;
    let desc_data = &compressed[p..];
    let mut dp2 = 0usize; // position into desc_data

    for _ in 0..n_blocks {
        if dp2 >= desc_data.len() { return None; }
        let bt = desc_data[dp2]; dp2 += 1;
        let ref_idx = read_u32_slice(desc_data, &mut dp2)?;
        let data_len = read_u16_slice(desc_data, &mut dp2)? as usize;
        let n_subs = desc_data.get(dp2).copied()? as usize; dp2 += 1;

        let data_offset = if bt == BT_RAW { dp_raw } else { dp_delta };
        if bt == BT_RAW { dp_raw += data_len; } else { dp_delta += data_len; }

        let mut subs = Vec::with_capacity(n_subs);
        for _ in 0..n_subs {
            let sbt = desc_data.get(dp2).copied()?; dp2 += 1;
            let s_offset = read_u16_slice(desc_data, &mut dp2)?;
            let s_len = read_u16_slice(desc_data, &mut dp2)?;
            let s_ref = read_u32_slice(desc_data, &mut dp2)?;
            let s_data_len = read_u16_slice(desc_data, &mut dp2)? as usize;
            
            let s_data_offset = if sbt == BT_RAW { dp_raw } else { dp_delta };
            if sbt == BT_RAW { dp_raw += s_data_len; } else { dp_delta += s_data_len; }
            
            subs.push(SubInfo {
                block_type: sbt, offset: s_offset, len: s_len,
                ref_idx: s_ref, data_offset: s_data_offset, data_len: s_data_len,
            });
        }

        blocks.push(BlockInfo {
            block_type: bt, ref_idx, data_offset, data_len, n_subs, subs,
        });
    }

    // Second pass: reconstruct
    let mut decoded_blocks: Vec<Vec<u8>> = Vec::with_capacity(n_blocks);

    for bi in &blocks {
        let block_data = if bi.block_type == BT_RAW {
            payload_raw.get(bi.data_offset..bi.data_offset + bi.data_len).unwrap_or(&[])
        } else {
            payload_delta.get(bi.data_offset..bi.data_offset + bi.data_len).unwrap_or(&[])
        };

        match bi.block_type {
            BT_RAW => {
                decoded_blocks.push(block_data.to_vec());
            }
            BT_DELTA_4K | BT_DELTA_1K | BT_DELTA_256 => {
                let ref_data = decoded_blocks.get(bi.ref_idx as usize)
                    .map(|v| v.as_slice()).unwrap_or(&[]);
                let mut out = xor_delta(block_data, ref_data);

                // Apply sub-block refinements
                for sub in &bi.subs {
                    let sub_data = if sub.block_type == BT_RAW {
                        payload_raw.get(sub.data_offset..sub.data_offset + sub.data_len).unwrap_or(&[])
                    } else {
                        payload_delta.get(sub.data_offset..sub.data_offset + sub.data_len).unwrap_or(&[])
                    };
                    let s_off = sub.offset as usize;
                    let s_len = sub.len as usize;

                    if sub.block_type == BT_RAW {
                        // Replace region with raw data
                        for i in 0..s_len.min(sub_data.len()) {
                            if s_off + i < out.len() {
                                out[s_off + i] = sub_data[i];
                            }
                        }
                    } else {
                        // Sub-delta: XOR with reference sub-block
                        let sub_ref_block_idx = (sub.ref_idx / (BLOCK_4K as u32 / s_len as u32)) as usize;
                        let sub_ref_offset = (sub.ref_idx % (BLOCK_4K as u32 / s_len as u32)) as usize * s_len;
                        if let Some(rb) = decoded_blocks.get(sub_ref_block_idx) {
                            for i in 0..s_len.min(sub_data.len()) {
                                let r = rb.get(sub_ref_offset + i).copied().unwrap_or(0);
                                if s_off + i < out.len() {
                                    out[s_off + i] = sub_data[i] ^ r;
                                }
                            }
                        }
                    }
                }

                decoded_blocks.push(out);
            }
            _ => { decoded_blocks.push(block_data.to_vec()); }
        }
    }

    let mut result: Vec<u8> = decoded_blocks.into_iter().flatten().collect();
    result.truncate(orig_len);
    Some(result)
}

// ─── Serialization ───────────────────────────────────────────────────────────

fn serialize(blocks: &[EncodedBlock], orig_len: usize) -> Vec<u8> {
    // Collect block data into two separate payloads for rANS encoding: raw and delta
    let mut raw_payload = Vec::new();
    let mut delta_payload = Vec::new();
    let mut descriptors = Vec::new();

    for b in blocks {
        // Block descriptor
        descriptors.push(b.block_type);
        descriptors.extend_from_slice(&b.ref_idx.to_le_bytes());
        descriptors.extend_from_slice(&(b.data.len() as u16).to_le_bytes());
        descriptors.push(b.sub_blocks.len() as u8);

        if b.block_type == BT_RAW {
            raw_payload.extend_from_slice(&b.data);
        } else {
            delta_payload.extend_from_slice(&b.data);
        }

        for sb in &b.sub_blocks {
            descriptors.push(sb.block_type);
            descriptors.extend_from_slice(&sb.offset.to_le_bytes());
            descriptors.extend_from_slice(&sb.len.to_le_bytes());
            descriptors.extend_from_slice(&sb.ref_idx.to_le_bytes());
            descriptors.extend_from_slice(&(sb.data.len() as u16).to_le_bytes());
            if sb.block_type == BT_RAW {
                raw_payload.extend_from_slice(&sb.data);
            } else {
                delta_payload.extend_from_slice(&sb.data);
            }
        }
    }

    // rANS encode separately for better modeling
    let enc_raw = crate::rans::encode(&raw_payload);
    let enc_delta = crate::rans::encode(&delta_payload);

    let mut out = Vec::new();
    out.extend_from_slice(&(orig_len as u32).to_le_bytes());
    out.extend_from_slice(&(blocks.len() as u32).to_le_bytes());
    out.extend_from_slice(&(enc_raw.len() as u32).to_le_bytes());
    out.extend_from_slice(&enc_raw);
    out.extend_from_slice(&(enc_delta.len() as u32).to_le_bytes());
    out.extend_from_slice(&enc_delta);
    out.extend_from_slice(&descriptors);
    out
}

// ─── Block matching helpers ──────────────────────────────────────────────────

fn compute_fingerprints(data: &[u8], block_size: usize) -> Vec<u64> {
    let n_blocks = (data.len() + block_size - 1) / block_size;
    let mut fps = Vec::with_capacity(n_blocks);
    for block_start in (0..data.len()).step_by(block_size) {
        let block_end = (block_start + block_size).min(data.len());
        fps.push(crate::pipeline::simhash_block(&data[block_start..block_end]));
    }
    fps
}

fn band_sig(hash: u64, band: usize) -> u16 {
    ((hash >> (band * 16)) & 0xFFFF) as u16
}

fn find_match(
    lsh: &[HashMap<u16, Vec<usize>>],
    fp: u64,
    all_fps: &[u64],
    current_idx: usize,
) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64)> = None;
    let mut seen = Vec::new();

    for band in 0..4 {
        let sig = band_sig(fp, band);
        if let Some(candidates) = lsh[band].get(&sig) {
            for &ri in candidates {
                if ri >= current_idx { continue; }
                if seen.contains(&ri) { continue; }
                seen.push(ri);
                let sim = simhash_similarity(fp, all_fps[ri]);
                if sim >= MATCH_THRESHOLD {
                    if best.map_or(true, |(_, s)| sim > s) {
                        best = Some((ri, sim));
                    }
                }
            }
        }
    }
    best
}

fn insert_lsh(lsh: &mut [HashMap<u16, Vec<usize>>], fp: u64, idx: usize) {
    for band in 0..4 {
        let sig = band_sig(fp, band);
        lsh[band].entry(sig).or_default().push(idx);
    }
}

fn subdivide_block(
    full_data: &[u8],
    _parent_block: &[u8],
    parent_idx: usize,
    sub_size: usize,
    _lsh: &[HashMap<u16, Vec<usize>>],
    _fps: &[u64],
    n: usize,
) -> Vec<EncodedSubBlock> {
    let parent_start = parent_idx * BLOCK_4K;
    let parent_end = (parent_start + BLOCK_4K).min(n);
    let parent = &full_data[parent_start..parent_end];
    let n_subs = (parent.len() + sub_size - 1) / sub_size;
    let mut subs = Vec::new();

    // For each sub-block, check if any preceding parent's sub-block is similar
    for si in 0..n_subs {
        let s_start = si * sub_size;
        let s_end = (s_start + sub_size).min(parent.len());
        let sub = &parent[s_start..s_end];
        let sub_fp = crate::pipeline::simhash_block(sub);

        // Search preceding blocks' sub-regions
        // Note: We use brute force here instead of LSH. A 16-block lookback window 
        // with 4 sub-blocks each means at most 64 SimHash comparisons per sub-block. 
        // This is bounded and acceptable performance-wise. If the lookback window 
        // is ever significantly raised, we should build an LSH index for sub-blocks. 
        let mut best_sub: Option<(u32, f64, Vec<u8>)> = None;
        let search_start = parent_idx.saturating_sub(16); // look back up to 16 blocks
        for prev_bi in search_start..parent_idx {
            let prev_start = prev_bi * BLOCK_4K;
            let prev_end = (prev_start + BLOCK_4K).min(n);
            let prev_block = &full_data[prev_start..prev_end];

            for prev_si in 0..((prev_block.len() + sub_size - 1) / sub_size) {
                let ps_start = prev_si * sub_size;
                let ps_end = (ps_start + sub_size).min(prev_block.len());
                let prev_sub = &prev_block[ps_start..ps_end];
                let prev_fp = crate::pipeline::simhash_block(prev_sub);

                let sim = simhash_similarity(sub_fp, prev_fp);
                if sim >= MATCH_THRESHOLD {
                    let delta = xor_delta(sub, prev_sub);
                    let dh = byte_entropy(&delta);
                    let sh = byte_entropy(sub);
                    if sh - dh >= DELTA_MIN_GAIN {
                        let ref_id = prev_bi as u32 * (BLOCK_4K as u32 / sub_size as u32)
                                   + prev_si as u32;
                        if best_sub.as_ref().map_or(true, |(_, s, _)| sim > *s) {
                            best_sub = Some((ref_id, sim, delta));
                        }
                    }
                }
            }
        }

        if let Some((ref_id, _, delta)) = best_sub {
            let bt = if sub_size == BLOCK_1K { BT_DELTA_1K } else { BT_DELTA_256 };
            subs.push(EncodedSubBlock {
                block_type: bt,
                offset: s_start as u16,
                len: sub.len() as u16,
                data: delta,
                ref_idx: ref_id,
            });
        }
        // If no good sub-match, the parent delta stands as-is for this region
    }

    subs
}

fn xor_delta(a: &[u8], b: &[u8]) -> Vec<u8> {
    let len = a.len().max(b.len());
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let av = a.get(i).copied().unwrap_or(0);
        let bv = b.get(i).copied().unwrap_or(0);
        out.push(av ^ bv);
    }
    out
}

fn simhash_similarity(a: u64, b: u64) -> f64 {
    let matching = 64 - (a ^ b).count_ones() as usize;
    matching as f64 / 64.0
}

fn byte_entropy(data: &[u8]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut freq = [0u32; 256];
    for &b in data { freq[b as usize] += 1; }
    let n = data.len() as f64;
    let mut h = 0.0f64;
    for &f in &freq {
        if f > 0 { let p = f as f64 / n; h -= p * p.log2(); }
    }
    h / 8.0
}

fn read_u32(data: &[u8], p: &mut usize) -> Option<u32> {
    if *p + 4 > data.len() { return None; }
    let v = u32::from_le_bytes(data[*p..*p + 4].try_into().ok()?);
    *p += 4;
    Some(v)
}

fn read_u32_slice(data: &[u8], p: &mut usize) -> Option<u32> {
    if *p + 4 > data.len() { return None; }
    let v = u32::from_le_bytes(data[*p..*p + 4].try_into().ok()?);
    *p += 4;
    Some(v)
}

fn read_u16_slice(data: &[u8], p: &mut usize) -> Option<u16> {
    if *p + 2 > data.len() { return None; }
    let v = u16::from_le_bytes(data[*p..*p + 2].try_into().ok()?);
    *p += 2;
    Some(v)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_simple() {
        let data: Vec<u8> = (0..8192).map(|i| ((i * 7 + 3) % 256) as u8).collect();
        let enc = compress(&data);
        let dec = decompress(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_with_repeats() {
        // Two identical 4KB blocks — should delta-encode
        let block: Vec<u8> = (0..BLOCK_4K).map(|i| (i % 251) as u8).collect();
        let mut data = block.clone();
        data.extend_from_slice(&block);
        data.extend_from_slice(&block);

        let enc = compress(&data);
        let dec = decompress(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_small() {
        let data = b"hello world";
        let enc = compress(data);
        let dec = decompress(&enc).unwrap();
        assert_eq!(dec, data.to_vec());
    }

    #[test]
    fn roundtrip_empty() {
        let enc = compress(&[]);
        let dec = decompress(&enc).unwrap();
        assert!(dec.is_empty());
    }

    #[test]
    fn roundtrip_single_block() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let enc = compress(&data);
        let dec = decompress(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_not_block_aligned() {
        let data: Vec<u8> = (0..5000).map(|i| ((i * 13) % 256) as u8).collect();
        let enc = compress(&data);
        let dec = decompress(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_large_similar_blocks() {
        // Many similar blocks with slight variations
        let base: Vec<u8> = (0..BLOCK_4K).map(|i| (i % 200) as u8).collect();
        let mut data = Vec::with_capacity(BLOCK_4K * 10);
        for j in 0..10 {
            let mut block = base.clone();
            // Modify a few bytes per block
            for k in 0..50 {
                block[k * 80] = (j * 17 + k) as u8;
            }
            data.extend_from_slice(&block);
        }

        let enc = compress(&data);
        let dec = decompress(&enc).unwrap();
        assert_eq!(dec, data);
    }
}
