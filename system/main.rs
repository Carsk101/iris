mod sort;
mod frame;
mod container;
mod encoder;

use std::fs;
use std::io::{BufWriter, BufReader};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, Context};
use clap::{Parser, Subcommand};

use sort::vision_sort;
use frame::{FrameLayout, pack_frames, unpack_frames};
use container::{IrisHeader, write_container, read_container};
use encoder::{detect_encoder, encode, decode};

#[derive(Parser)]
#[command(
    name = "iris",
    about = "GPU-accelerated compression via NVENC + VisionSort",
    version = "0.1.0",
    author = "Harsh Patel"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compress a file
    Compress {
        /// Input file path
        input: PathBuf,
        /// Output .iris file path
        output: PathBuf,
    },
    /// Decompress an .iris file
    Decompress {
        /// Input .iris file path
        input: PathBuf,
        /// Output file path
        output: PathBuf,
    },
    /// Show info about an .iris file without decompressing
    Info {
        /// Input .iris file path
        input: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compress { input, output } => cmd_compress(&input, &output),
        Commands::Decompress { input, output } => cmd_decompress(&input, &output),
        Commands::Info { input } => cmd_info(&input),
    }
}

fn cmd_compress(input: &PathBuf, output: &PathBuf) -> Result<()> {
    let t0 = Instant::now();

    eprintln!("[iris] reading {:?}", input);
    let data = fs::read(input).with_context(|| format!("cannot read {:?}", input))?;
    let original_len = data.len();
    eprintln!("[iris] input: {} bytes", original_len);

    // Step 1: VisionSort
    eprintln!("[iris] sorting with VisionSort...");
    let t1 = Instant::now();
    let sort_result = vision_sort(&data);
    eprintln!(
        "[iris] sorted in {:.2}ms | route: {:?} | entropy: {:.3} bits",
        t1.elapsed().as_secs_f64() * 1000.0,
        sort_result.route,
        sort_result.entropy
    );

    // Step 2: Frame layout
    let layout = FrameLayout::from_data_len(original_len);
    eprintln!(
        "[iris] frame layout: {}x{} x {} frames",
        layout.width, layout.height, layout.frame_count
    );

    // Step 3: Pack into YUV420 frames
    let yuv_frames = pack_frames(&sort_result.sorted, &layout);

    // Step 4: Detect encoder + encode
    let enc = detect_encoder();
    let t2 = Instant::now();
    let video_data = encode(&yuv_frames, layout.width, layout.height, layout.frame_count, enc)?;
    eprintln!("[iris] encode took {:.2}ms", t2.elapsed().as_secs_f64() * 1000.0);

    // Step 5: Write .iris container
    let header = IrisHeader {
        original_len: original_len as u64,
        width: layout.width as u32,
        height: layout.height as u32,
        frame_count: layout.frame_count as u32,
        route: sort_result.route,
    };

    let out_file = fs::File::create(output)
        .with_context(|| format!("cannot create {:?}", output))?;
    let mut writer = BufWriter::new(out_file);
    write_container(&mut writer, &header, &sort_result.permutation, &video_data)?;

    let compressed_size = fs::metadata(output)?.len() as usize;
    let ratio = original_len as f64 / compressed_size as f64;

    eprintln!("[iris] done in {:.2}ms", t0.elapsed().as_secs_f64() * 1000.0);
    eprintln!(
        "[iris] {} bytes -> {} bytes | ratio: {:.3}x | savings: {:.1}%",
        original_len,
        compressed_size,
        ratio,
        (1.0 - 1.0 / ratio) * 100.0
    );

    Ok(())
}

fn cmd_decompress(input: &PathBuf, output: &PathBuf) -> Result<()> {
    let t0 = Instant::now();

    eprintln!("[iris] reading {:?}", input);
    let in_file = fs::File::open(input)
        .with_context(|| format!("cannot open {:?}", input))?;
    let mut reader = BufReader::new(in_file);
    let (header, permutation, video_data) = read_container(&mut reader)?;

    eprintln!(
        "[iris] container: {}x{} x {} frames | original: {} bytes | route: {:?}",
        header.width, header.height, header.frame_count,
        header.original_len, header.route
    );

    let layout = header.layout();
    let enc = detect_encoder();

    // Decode video -> YUV frames
    let yuv_frames = decode(
        &video_data,
        layout.width,
        layout.height,
        layout.frame_count,
        enc,
    )?;

    // Unpack Y planes -> sorted bytes
    let sorted = unpack_frames(&yuv_frames, &layout);

    // Unsort via permutation index
    let original = sort::unsort(&sorted, &permutation);

    fs::write(output, &original[..header.original_len as usize])
        .with_context(|| format!("cannot write {:?}", output))?;

    eprintln!("[iris] decompressed in {:.2}ms", t0.elapsed().as_secs_f64() * 1000.0);
    eprintln!("[iris] wrote {} bytes to {:?}", header.original_len, output);

    Ok(())
}

fn cmd_info(input: &PathBuf) -> Result<()> {
    let in_file = fs::File::open(input)
        .with_context(|| format!("cannot open {:?}", input))?;
    let mut reader = BufReader::new(in_file);
    let (header, permutation, video_data) = read_container(&mut reader)?;

    let compressed_size = fs::metadata(input)?.len();

    println!("iris file: {:?}", input);
    println!("  original size : {} bytes", header.original_len);
    println!("  compressed    : {} bytes", compressed_size);
    println!("  ratio         : {:.3}x", header.original_len as f64 / compressed_size as f64);
    println!("  frame dims    : {}x{}", header.width, header.height);
    println!("  frames        : {}", header.frame_count);
    println!("  sort route    : {:?}", header.route);
    println!("  perm entries  : {}", permutation.len());
    println!("  video payload : {} bytes", video_data.len());

    Ok(())
}
