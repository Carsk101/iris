/// iris container — varint-delta scatter encoding + CRC32 integrity
use std::io::{Read, Write};
use anyhow::{Result, bail};

/// Native CRC32 (IEEE 802.3 polynomial, same as crc32fast/zlib)
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

pub const MAGIC:   &[u8; 4] = b"IRS2";
pub const VERSION: u8       = 3;  // bumped: CRC32 trailer added
pub const FLAG_RESONANCE:    u8 = 0b0001;
pub const FLAG_GRAMMAR:      u8 = 0b0010;
pub const FLAG_PRED_GRAPH:   u8 = 0b0100;
pub const FLAG_CONTEXT_PACK: u8 = 0b1000;

pub struct IrisContainer {
    pub encoder:        u8,
    pub flags:          u8,
    pub original_len:   u64,
    pub byte0:          u8,
    pub original_byte0: u8,
    pub resonance_hdr:  Vec<u8>,
    pub grammar_data:   Vec<u8>,
    pub pred_meta:      Vec<u8>,
    pub scatter_map:    Vec<u8>,
    pub ctx_hdr:        Vec<u8>,
    pub video:          Vec<u8>,
}

impl IrisContainer {
    pub fn has_resonance(&self)    -> bool { self.flags & FLAG_RESONANCE    != 0 }
    pub fn has_grammar(&self)      -> bool { self.flags & FLAG_GRAMMAR      != 0 }
    pub fn has_pred_graph(&self)   -> bool { self.flags & FLAG_PRED_GRAPH   != 0 }
    pub fn has_context_pack(&self) -> bool { self.flags & FLAG_CONTEXT_PACK != 0 }

    pub fn write<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(MAGIC)?;
        w.write_all(&[VERSION, self.encoder, self.flags, 0u8])?;
        w.write_all(&self.original_len.to_le_bytes())?;
        w.write_all(&[self.byte0, self.original_byte0])?;

        // Collect all chunk data to compute CRC32 over payloads
        let mut payload_buf = Vec::new();
        for chunk in &[&self.resonance_hdr, &self.grammar_data, &self.pred_meta,
                       &self.scatter_map, &self.ctx_hdr, &self.video] {
            let len_bytes = (chunk.len() as u32).to_le_bytes();
            payload_buf.extend_from_slice(&len_bytes);
            payload_buf.extend_from_slice(chunk);
        }

        w.write_all(&payload_buf)?;

        // CRC32 trailer over all chunk data
        let crc = crc32(&payload_buf);
        w.write_all(&crc.to_le_bytes())?;
        Ok(())
    }

    pub fn read<R: Read>(r: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4]; r.read_exact(&mut magic)?;
        if &magic != MAGIC { bail!("not an iris file"); }
        let mut hdr = [0u8; 4]; r.read_exact(&mut hdr)?;
        let version = hdr[0];
        if version != VERSION && version != 2 { bail!("unsupported version {}", version); }
        let mut lb = [0u8; 8]; r.read_exact(&mut lb)?;
        let original_len = u64::from_le_bytes(lb);
        let mut b = [0u8; 2]; r.read_exact(&mut b)?;

        let container = Self {
            encoder: hdr[1], flags: hdr[2], original_len,
            byte0: b[0], original_byte0: b[1],
            resonance_hdr: read_chunk(r)?,
            grammar_data:  read_chunk(r)?,
            pred_meta:     read_chunk(r)?,
            scatter_map:   read_chunk(r)?,
            ctx_hdr:       read_chunk(r)?,
            video:         read_chunk(r)?,
        };

        // Verify CRC32 if version >= 3
        if version >= 3 {
            let mut crc_buf = [0u8; 4];
            r.read_exact(&mut crc_buf)?;
            let stored_crc = u32::from_le_bytes(crc_buf);

            // Recompute CRC over chunk data
            let mut payload_buf = Vec::new();
            for chunk in &[&container.resonance_hdr, &container.grammar_data,
                           &container.pred_meta, &container.scatter_map,
                           &container.ctx_hdr, &container.video] {
                payload_buf.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
                payload_buf.extend_from_slice(chunk);
            }
            let computed_crc = crc32(&payload_buf);
            if stored_crc != computed_crc {
                bail!("CRC32 mismatch: container is corrupt (stored={:#010x}, computed={:#010x})",
                    stored_crc, computed_crc);
            }
        }

        Ok(container)
    }
}

fn read_chunk<R: Read>(r: &mut R) -> Result<Vec<u8>> {
    let mut lb = [0u8; 4]; r.read_exact(&mut lb)?;
    let len = u32::from_le_bytes(lb) as usize;
    if len == 0 { return Ok(vec![]); }
    let mut buf = vec![0u8; len]; r.read_exact(&mut buf)?;
    Ok(buf)
}

// ── Varint helpers ────────────────────────────────────────────────────────────

fn write_varint(buf: &mut Vec<u8>, mut n: u64) {
    loop {
        if n < 0x80 { buf.push(n as u8); break; }
        buf.push((n as u8 & 0x7f) | 0x80);
        n >>= 7;
    }
}

fn read_varint(data: &[u8], pos: &mut usize) -> Option<u64> {
    let mut result = 0u64;
    let mut shift  = 0u32;
    loop {
        if *pos >= data.len() { return None; }
        let byte = data[*pos]; *pos += 1;
        result |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 { return Some(result); }
        shift += 7;
        if shift >= 64 { return None; }
    }
}

/// Serialize scatter map as varint-delta compressed with zstd.
/// Positions within each bucket are in appearance order (monotone increasing).
/// Delta-varint: for nearly uniform positions (stride ≈ n_contexts),
/// all deltas ≈ same small integer → zstd compresses to ~bytes.
pub fn serialize_scatter(lists: &[Vec<u32>]) -> Result<Vec<u8>> {
    let mut varint_buf = Vec::with_capacity(lists.len() * 4 + 16);
    write_varint(&mut varint_buf, lists.len() as u64);
    for list in lists {
        write_varint(&mut varint_buf, list.len() as u64);
        let mut prev = 0u32;
        for &pos in list {
            write_varint(&mut varint_buf, (pos.saturating_sub(prev)) as u64);
            prev = pos;
        }
    }
    eprintln!("[iris] scatter varint: {}B", varint_buf.len());
    let compressed = crate::range_coder::encode(&varint_buf);
    eprintln!("[iris] scatter range:  {}B", compressed.len());
    Ok(compressed)
}

pub fn deserialize_scatter(compressed: &[u8]) -> Result<Vec<Vec<u32>>> {
    if compressed.is_empty() { return Ok(vec![]); }
    let raw = crate::range_coder::decode(compressed)
        .ok_or_else(|| anyhow::anyhow!("corrupt scatter map"))?;
    let mut pos = 0;
    let n_lists = read_varint(&raw, &mut pos).unwrap_or(0) as usize;
    let mut lists = Vec::with_capacity(n_lists);
    for _ in 0..n_lists {
        let len = read_varint(&raw, &mut pos).unwrap_or(0) as usize;
        let mut list = Vec::with_capacity(len);
        let mut prev = 0u32;
        for _ in 0..len {
            let delta = read_varint(&raw, &mut pos).unwrap_or(0) as u32;
            prev += delta;
            list.push(prev);
        }
        lists.push(list);
    }
    Ok(lists)
}

pub fn serialize_pred_meta(metas: &[crate::prediction::BlockMeta]) -> Vec<u8> {
    let mut out = Vec::with_capacity(metas.len() * 17 + 4);
    out.extend_from_slice(&(metas.len() as u32).to_le_bytes());
    for m in metas { out.extend_from_slice(&m.serialize()); }
    out
}

pub fn deserialize_pred_meta(data: &[u8]) -> Option<Vec<crate::prediction::BlockMeta>> {
    if data.len() < 4 { return Some(vec![]); }
    let n = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
    let mut metas = Vec::with_capacity(n);
    let mut pos = 4;
    for _ in 0..n {
        metas.push(crate::prediction::BlockMeta::deserialize(&data[pos..])?);
        pos += 17;
    }
    Some(metas)
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn container_roundtrip() {
        let c = IrisContainer {
            encoder: 0xFF, flags: FLAG_RESONANCE | FLAG_CONTEXT_PACK,
            original_len: 12345, byte0: 0xAB, original_byte0: 0xCD,
            resonance_hdr: vec![1, 2, 3],
            grammar_data: vec![],
            pred_meta: vec![4, 5, 6, 7],
            scatter_map: vec![8, 9],
            ctx_hdr: vec![10, 11, 12, 13, 14],
            video: vec![0xFF; 100],
        };
        let mut buf = Vec::new();
        c.write(&mut buf).unwrap();

        let c2 = IrisContainer::read(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(c2.encoder, 0xFF);
        assert_eq!(c2.flags, FLAG_RESONANCE | FLAG_CONTEXT_PACK);
        assert_eq!(c2.original_len, 12345);
        assert_eq!(c2.byte0, 0xAB);
        assert_eq!(c2.original_byte0, 0xCD);
        assert_eq!(c2.resonance_hdr, vec![1, 2, 3]);
        assert_eq!(c2.grammar_data, vec![]);
        assert_eq!(c2.pred_meta, vec![4, 5, 6, 7]);
        assert_eq!(c2.scatter_map, vec![8, 9]);
        assert_eq!(c2.ctx_hdr, vec![10, 11, 12, 13, 14]);
        assert_eq!(c2.video, vec![0xFF; 100]);
    }

    #[test]
    fn container_crc_detects_corruption() {
        let c = IrisContainer {
            encoder: 0xFF, flags: 0,
            original_len: 100, byte0: 0, original_byte0: 0,
            resonance_hdr: vec![], grammar_data: vec![],
            pred_meta: vec![], scatter_map: vec![],
            ctx_hdr: vec![], video: vec![42; 50],
        };
        let mut buf = Vec::new();
        c.write(&mut buf).unwrap();

        // Corrupt a byte in the video payload
        let len = buf.len();
        buf[len - 10] ^= 0xFF;

        let result = IrisContainer::read(&mut Cursor::new(&buf));
        assert!(result.is_err(), "should detect CRC corruption");
    }

    #[test]
    fn scatter_roundtrip() {
        let lists = vec![
            vec![1, 5, 10, 100, 1000],
            vec![2, 3, 4],
            vec![0, 50000],
        ];
        let serialized = serialize_scatter(&lists).unwrap();
        let deserialized = deserialize_scatter(&serialized).unwrap();
        assert_eq!(deserialized, lists);
    }

    #[test]
    fn scatter_empty() {
        let lists: Vec<Vec<u32>> = vec![];
        let serialized = serialize_scatter(&lists).unwrap();
        let deserialized = deserialize_scatter(&serialized).unwrap();
        assert_eq!(deserialized, lists);
    }

    #[test]
    fn pred_meta_roundtrip() {
        let metas = vec![
            crate::prediction::BlockMeta {
                block_idx: 0, is_delta: false, ref_block: 0, offset: 0, len: 4096,
            },
            crate::prediction::BlockMeta {
                block_idx: 1, is_delta: true, ref_block: 0, offset: 4096, len: 4096,
            },
        ];
        let bytes = serialize_pred_meta(&metas);
        let metas2 = deserialize_pred_meta(&bytes).unwrap();
        assert_eq!(metas2.len(), 2);
        assert_eq!(metas2[0].block_idx, 0);
        assert!(!metas2[0].is_delta);
        assert_eq!(metas2[1].block_idx, 1);
        assert!(metas2[1].is_delta);
        assert_eq!(metas2[1].ref_block, 0);
    }
}
