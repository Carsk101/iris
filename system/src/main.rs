mod profile;
mod resonance;
mod grammar;
mod prediction;
mod context_pack;
mod encoder;
mod container;
mod pipeline;

use std::path::PathBuf;
use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name="iris", version="0.2.0", author="Harsh Patel",
    about="GPU-accelerated compression: VisionSort + Resonance + Grammar + Prediction Graph + Context-Aware AV1")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Compress a file
    Compress { input: PathBuf, output: PathBuf },
    /// Decompress an .iris file
    Decompress { input: PathBuf, output: PathBuf },
    /// Show metadata without decompressing
    Info { input: PathBuf },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Compress   { input, output } => pipeline::compress(&input, &output),
        Cmd::Decompress { input, output } => pipeline::decompress(&input, &output),
        Cmd::Info       { input }         => pipeline::info(&input),
    }
}
