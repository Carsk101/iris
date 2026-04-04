/// .iris container format
///
/// Layout:
///   [0..4]   magic: b"IRIS"
///   [4]      version: u8 = 1
///   [5..13]  original_len: u64 le
///   [13..17] width: u32 le
///   [17..21] height: u32 le
///   [21..25] frame_count: u32 le
///   [25..29] route: u8 (0=NearlyFree 1=Verify 2=PlacementSort 3=FullSort) + 3 reserved
///   [29..37] perm_len: u64 le  (length of zstd-compressed permutation)
///   [37..37+perm_len]  permutation (zstd compressed u32 le array)
///   [37+perm_len..]    AV1/HEVC video bitstream (mkv container)

use std::io::{Read, Write, Cursor};
use anyhow::{Result, bail};
use byteorder::{LE, ReadBytesExt, WriteBytesExt};
use crate::sort::Route;
use crate::frame::FrameLayout;

pub const MAGIC: &[u8; 4] = b"IRIS";
pub const VERSION: u8 = 1;

pub struct IrisHeader {
    pub original_len: u64,
    pub width: u32,
    pub height: u32,
    pub frame_count: u32,
    pub route: Route,
}

pub fn write_container<W: Write>(
    writer: &mut W,
    header: &IrisHeader,
    permutation: &[u32],
    video_data: &[u8],
) -> Result<()> {
    // Compress permutation index with zstd
    let mut raw_perm = Vec::with_capacity(permutation.len() * 4);
    for &idx in permutation {
        raw_perm.write_u32::<LE>(idx)?;
    }
    let compressed_perm = zstd::encode_all(Cursor::new(&raw_perm), 19)?;

    // Magic + version
    writer.write_all(MAGIC)?;
    writer.write_u8(VERSION)?;

    // Header fields
    writer.write_u64::<LE>(header.original_len)?;
    writer.write_u32::<LE>(header.width)?;
    writer.write_u32::<LE>(header.height)?;
    writer.write_u32::<LE>(header.frame_count)?;

    let route_byte = match header.route {
        Route::NearlyFree    => 0u8,
        Route::Verify        => 1u8,
        Route::PlacementSort => 2u8,
        Route::FullSort      => 3u8,
    };
    writer.write_u8(route_byte)?;
    writer.write_all(&[0u8; 3])?; // reserved

    // Permutation
    writer.write_u64::<LE>(compressed_perm.len() as u64)?;
    writer.write_all(&compressed_perm)?;

    // Video bitstream
    writer.write_all(video_data)?;

    Ok(())
}

pub fn read_container<R: Read>(reader: &mut R) -> Result<(IrisHeader, Vec<u32>, Vec<u8>)> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        bail!("not an iris file");
    }

    let version = reader.read_u8()?;
    if version != VERSION {
        bail!("unsupported iris version: {}", version);
    }

    let original_len  = reader.read_u64::<LE>()?;
    let width         = reader.read_u32::<LE>()?;
    let height        = reader.read_u32::<LE>()?;
    let frame_count   = reader.read_u32::<LE>()?;

    let route_byte    = reader.read_u8()?;
    let mut _reserved = [0u8; 3];
    reader.read_exact(&mut _reserved)?;

    let route = match route_byte {
        0 => Route::NearlyFree,
        1 => Route::Verify,
        2 => Route::PlacementSort,
        _ => Route::FullSort,
    };

    let perm_len = reader.read_u64::<LE>()? as usize;
    let mut compressed_perm = vec![0u8; perm_len];
    reader.read_exact(&mut compressed_perm)?;

    let raw_perm = zstd::decode_all(Cursor::new(&compressed_perm))?;
    let mut perm_cursor = Cursor::new(&raw_perm);
    let perm_count = raw_perm.len() / 4;
    let mut permutation = Vec::with_capacity(perm_count);
    for _ in 0..perm_count {
        permutation.push(perm_cursor.read_u32::<LE>()?);
    }

    let mut video_data = Vec::new();
    reader.read_to_end(&mut video_data)?;

    let header = IrisHeader { original_len, width, height, frame_count, route };
    Ok((header, permutation, video_data))
}

impl IrisHeader {
    pub fn layout(&self) -> FrameLayout {
        FrameLayout {
            width: self.width as usize,
            height: self.height as usize,
            frame_count: self.frame_count as usize,
            padded_len: self.width as usize * self.height as usize * self.frame_count as usize,
            original_len: self.original_len as usize,
        }
    }
}
