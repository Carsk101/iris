/// Stage 4 — Context-Aware Frame Packing
///
/// Groups bytes by their preceding byte (context). All bytes that follow
/// the same context byte land in the same row. AV1's horizontal spatial
/// predictor sees a nearly-constant row → near-zero residual.
///
/// KEY INSIGHT for decompression: we store ZERO position data.
/// Instead we follow the context chain:
///   out[0] = byte0
///   prev   = byte0
///   for i in 1..N:
///     out[i]       = next byte from ctx_buf[prev]
///     prev         = out[i]
///
/// Each decoded byte tells you which context buffer the NEXT byte comes from.
/// This reconstructs the original sequence in O(N) with no scatter map.
/// The scatter map overhead (10+ MB on large files) drops to zero.

pub const FRAME_W:     usize = 1920;
pub const FRAME_H_MAX: usize = 1080;

#[derive(Debug, Clone)]
pub struct ContextPackHeader {
    pub frame_width:  u32,
    pub frame_height: u32,
    pub frame_count:  u32,
    pub data_len:     u32,
    pub context_map:  Vec<ContextMapEntry>,
}

#[derive(Debug, Clone)]
pub struct ContextMapEntry {
    pub context_byte: u8,
    pub count:        u32,
}

/// Pack bytes into context-grouped YUV420p frames.
/// Returns (header, raw_yuv_bytes).
/// Does NOT produce a scatter map — decompression uses chain reconstruction.
pub fn pack(data: &[u8], ctx_table: &crate::profile::ContextTable) -> (ContextPackHeader, Vec<u8>) {
    let n = data.len();
    let active = &ctx_table.active_contexts;

    // Build flat buffer: lay bytes in context groups, in appearance order
    let mut flat = Vec::with_capacity(n);
    let mut context_map = Vec::new();

    for &ctx_byte in active {
        let bucket = &ctx_table.buckets[ctx_byte as usize];
        if bucket.is_empty() { continue; }
        context_map.push(ContextMapEntry {
            context_byte: ctx_byte,
            count: bucket.len() as u32,
        });
        for &pos in bucket {
            if (pos as usize) < n {
                flat.push(data[pos as usize]);
            }
        }
    }
    // Position 0 excluded (no preceding byte) — restored via byte0 in container.

    // Carve flat into YUV420p frames
    let w = FRAME_W;
    let pf = w * FRAME_H_MAX;
    let n_frames = (flat.len() + pf - 1).max(1) / pf;
    let n_frames = n_frames.max(1);
    let h = if n_frames == 1 {
        (flat.len() + w - 1) / w
    } else {
        FRAME_H_MAX
    };
    let h = h.max(1);

    let yuv_fsize = w * h * 3 / 2;
    let pixels_per_frame = w * h;
    let mut yuv = vec![0u8; yuv_fsize * n_frames];

    for frame_idx in 0..n_frames {
        let y0  = frame_idx * yuv_fsize;
        let cb0 = y0 + pixels_per_frame;
        let cr0 = cb0 + pixels_per_frame / 4;

        let d0  = frame_idx * pixels_per_frame;
        for px in 0..pixels_per_frame {
            let fi = d0 + px;
            yuv[y0 + px] = if fi < flat.len() { flat[fi] } else { 0 };
        }
        for i in 0..pixels_per_frame / 4 {
            yuv[cb0 + i] = 0x80;
            yuv[cr0 + i] = 0x80;
        }
    }

    let header = ContextPackHeader {
        frame_width:  w as u32,
        frame_height: h as u32,
        frame_count:  n_frames as u32,
        data_len:     n as u32,
        context_map,
    };

    (header, yuv)
}

/// Reconstruct original byte sequence from decoded YUV using context chain.
/// No scatter map needed — follows each byte's context to find the next.
///
/// Algorithm:
///   1. Extract flat buffer from Y planes
///   2. Split flat into per-context slices (using context_map counts)
///   3. out[0] = byte0; prev = byte0
///   4. for i in 1..N: out[i] = ctx_slice[prev].next(); prev = out[i]
pub fn unpack_implicit(header: &ContextPackHeader, yuv: &[u8], byte0: u8) -> Vec<u8> {
    let w    = header.frame_width  as usize;
    let h    = header.frame_height as usize;
    let nf   = header.frame_count  as usize;
    let n    = header.data_len     as usize;
    let pf   = w * h;
    let fsize = pf * 3 / 2;

    // Extract flat from Y planes
    let mut flat = Vec::with_capacity(nf * pf);
    for fi in 0..nf {
        let y0  = fi * fsize;
        let end = (y0 + pf).min(yuv.len());
        flat.extend_from_slice(&yuv[y0..end]);
    }

    // Build per-context slices
    // ctx_start[c] = start index in flat, ctx_len[c] = count
    let mut ctx_start = [0u32; 256];
    let mut ctx_len   = [0u32; 256];
    let mut flat_pos  = 0u32;

    for entry in &header.context_map {
        let c   = entry.context_byte as usize;
        let cnt = entry.count;
        ctx_start[c] = flat_pos;
        ctx_len[c]   = cnt;
        flat_pos += cnt;
    }

    // Context chain reconstruction
    let mut out  = Vec::with_capacity(n);
    let mut ptrs = [0u32; 256];  // read pointer per context

    if n == 0 { return out; }
    out.push(byte0);
    let mut prev = byte0 as usize;

    for _ in 1..n {
        let ptr = ptrs[prev] as usize;
        let start = ctx_start[prev] as usize;
        let len   = ctx_len[prev]   as usize;
        let b = if ptr < len && start + ptr < flat.len() {
            flat[start + ptr]
        } else {
            0
        };
        out.push(b);
        ptrs[prev] += 1;
        prev = b as usize;
    }

    out
}

pub fn serialize_header(h: &ContextPackHeader) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&h.frame_width.to_le_bytes());
    out.extend_from_slice(&h.frame_height.to_le_bytes());
    out.extend_from_slice(&h.frame_count.to_le_bytes());
    out.extend_from_slice(&h.data_len.to_le_bytes());
    out.extend_from_slice(&(h.context_map.len() as u32).to_le_bytes());
    for e in &h.context_map {
        out.push(e.context_byte);
        out.extend_from_slice(&e.count.to_le_bytes());
    }
    out
}

pub fn deserialize_header(data: &[u8]) -> Option<(ContextPackHeader, usize)> {
    if data.len() < 20 { return None; }
    let mut p = 0usize;
    macro_rules! u32 { () => {{ let v = u32::from_le_bytes(data[p..p+4].try_into().ok()?); p+=4; v }} }
    let fw=u32!(); let fh=u32!(); let fc=u32!(); let dl=u32!();
    let nm=u32!() as usize;
    let mut cm = Vec::with_capacity(nm);
    for _ in 0..nm {
        if p+5 > data.len() { return None; }
        let cb = data[p]; p+=1;
        let cnt = u32::from_le_bytes(data[p..p+4].try_into().ok()?); p+=4;
        cm.push(ContextMapEntry { context_byte: cb, count: cnt });
    }
    Some((ContextPackHeader { frame_width:fw, frame_height:fh, frame_count:fc,
                               data_len:dl, context_map:cm }, p))
}

/// Pack with sort-within-context: bytes within each context bucket are sorted
/// by value before being written to the flat buffer. This makes each context
/// row a monotone sequence → zstd/AV1 compresses to near-zero residual.
///
/// Returns (flat_buffer, position_lists, header).
/// flat_buffer: sorted bytes per context, concatenated.
/// position_lists: for each context, the SORT PERMUTATION mapping flat_idx → original position.
pub fn pack_sorted(
    data: &[u8],
    ctx_table: &crate::profile::ContextTable,
) -> (Vec<u8>, Vec<Vec<u32>>, ContextPackHeader) {
    let n = data.len();
    let active = &ctx_table.active_contexts;

    let mut flat: Vec<u8>       = Vec::with_capacity(n);
    let mut position_lists: Vec<Vec<u32>> = Vec::new();
    let mut context_map: Vec<ContextMapEntry> = Vec::new();

    for &ctx_byte in active {
        let bucket = &ctx_table.buckets[ctx_byte as usize];
        if bucket.is_empty() { continue; }

        // Sort bucket positions by BYTE VALUE at that position,
        // then by original position for stability.
        let mut sorted_bucket: Vec<u32> = bucket.iter()
            .filter(|&&p| (p as usize) < n)
            .cloned()
            .collect();
        sorted_bucket.sort_by_key(|&p| (data[p as usize], p));

        let count = sorted_bucket.len() as u32;
        context_map.push(ContextMapEntry { context_byte: ctx_byte, count });
        position_lists.push(sorted_bucket.clone());

        for &pos in &sorted_bucket {
            flat.push(data[pos as usize]);
        }
    }

    // Frame dimensions (same formula as pack())
    let w = FRAME_W;
    let h = ((flat.len() + w - 1) / w).clamp(1, FRAME_H_MAX);
    let pf = w * h;
    let fc = (flat.len() + pf - 1).max(1) / pf;

    let hdr = ContextPackHeader {
        frame_width:  w as u32,
        frame_height: h as u32,
        frame_count:  fc as u32,
        data_len:     n as u32,
        context_map,
    };

    (flat, position_lists, hdr)
}

/// Unpack from a flat buffer (zstd-route: no YUV, just flat bytes).
/// Identical scatter logic to unpack_full but takes flat bytes directly.
pub fn unpack_full(
    header: &ContextPackHeader,
    flat: &[u8],
    byte0: u8,
    position_lists: &[Vec<u32>],
) -> Vec<u8> {
    let n = header.data_len as usize;
    let mut out = vec![0u8; n];
    if n > 0 { out[0] = byte0; }

    let mut flat_idx = 0usize;
    for (ei, _entry) in header.context_map.iter().enumerate() {
        let positions = if ei < position_lists.len() { &position_lists[ei] } else { break };
        for &pos in positions {
            if flat_idx < flat.len() && (pos as usize) < n {
                out[pos as usize] = flat[flat_idx];
                flat_idx += 1;
            }
        }
    }
    out
}

/// Pack in appearance order (positions monotone increasing per bucket).
/// Returns (flat_buffer, position_lists, header).
/// Use this for zstd-route: varint-delta scatter works because positions are monotone.
pub fn pack_with_positions(
    data: &[u8],
    ctx_table: &crate::profile::ContextTable,
) -> (Vec<u8>, Vec<Vec<u32>>, ContextPackHeader) {
    let n = data.len();
    let active = &ctx_table.active_contexts;

    let mut flat: Vec<u8>             = Vec::with_capacity(n);
    let mut position_lists: Vec<Vec<u32>> = Vec::new();
    let mut context_map: Vec<ContextMapEntry> = Vec::new();

    for &ctx_byte in active {
        let bucket = &ctx_table.buckets[ctx_byte as usize];
        if bucket.is_empty() { continue; }
        let valid: Vec<u32> = bucket.iter().filter(|&&p| (p as usize) < n).cloned().collect();
        if valid.is_empty() { continue; }

        context_map.push(ContextMapEntry { context_byte: ctx_byte, count: valid.len() as u32 });
        for &pos in &valid { flat.push(data[pos as usize]); }
        position_lists.push(valid);
    }

    let w  = FRAME_W;
    let h  = ((flat.len() + w - 1) / w).clamp(1, FRAME_H_MAX);
    let pf = w * h;
    let fc = (flat.len() + pf - 1).max(1) / pf;

    let hdr = ContextPackHeader {
        frame_width:  w as u32,
        frame_height: h as u32,
        frame_count:  fc as u32,
        data_len:     n as u32,
        context_map,
    };

    (flat, position_lists, hdr)
}
