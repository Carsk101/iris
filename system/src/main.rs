mod profile;
mod gate;
mod lag_cache;
mod resonance;
mod grammar;
mod prediction;
mod context_pack;
mod container;
mod pipeline;
mod range_coder;
mod rans;
mod block_match;
mod codec;

use std::path::PathBuf;
use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name="iris", version="1.0.0", author="Harsh Patel",
    about="iris v1.0 — zero runtime dependencies compression")]
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
    /// Show metadata of an .iris file without decompressing
    Info { input: PathBuf },
    /// Run the sampling profiler + gate on a raw input and print the
    /// statistics and stage decision. Does not write anything. Useful
    /// for entropy-based structure detection experiments on large
    /// genomic or scientific datasets. Respects $IRIS_GATE=genomic.
    Profile { input: PathBuf },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Compress   { input, output } => pipeline::compress(&input, &output),
        Cmd::Decompress { input, output } => pipeline::decompress(&input, &output),
        Cmd::Info       { input }         => pipeline::info(&input),
        Cmd::Profile    { input }         => pipeline::profile_only(&input),
    }
}
