# iris

A compression engine that treats compression as a perception problem. Rather than applying a single algorithm to everything, iris profiles the input first — measuring entropy, periodicity, and structure — then routes it through the stage best suited to exploit what it finds.

Built in Rust. Uses GPU silicon (NVENC) when available.

---

## How it works

Most compressors are general-purpose by design. They apply the same sliding-window or entropy-coding strategy regardless of what the data actually looks like. iris takes the opposite approach: read the data once, understand its structure, then compress each structural property with the tool that eliminates it most completely.

**Stage 0 — Profile**
A single read-only pass measures byte entropy, autocorrelation at all lags up to 64, context frequencies, and block fingerprints. Everything downstream uses this profile — nothing reads the data twice.

**Column grammar** (fires first for text)
Structured text like logs and CSV gets decomposed into columns. Each column is encoded with the type that fits it — timestamps as three separate u8 arrays (H, M, S), integer fields as packed u16/u32 arrays, low-cardinality fields as dictionary indices, free text as zstd. Adjacent columns that always co-occur as fixed pairs are merged into a single dictionary entry. The result compresses each column to near its information-theoretic minimum independently.

**Stage 1 — Resonance**
Detects periodic structure using wrapping i8 autocorrelation. For data with a dominant period — sensor readings, sorted integer arrays, binary records — the residual after delta-subtraction at that lag is near-zero. Runs up to three passes, stopping when entropy drops below the ultra-compress threshold.

**Stage 2 — Stride grammar**
For binary data with fixed-stride records (e.g. structs serialized to disk), identifies which byte positions within each stride are effectively constant. Separates fixed bytes from variable fields. Variable fields then get compressed independently.

**Stage 3 — Prediction graph**
Finds similar blocks anywhere in the file using MinHash fingerprints — no window limit. LZ77 is physically constrained to its sliding window (typically 8–32MB). iris has no such limit. Similar blocks are stored as XOR deltas from their nearest match regardless of distance.

**Stage 4+5 — Route and encode**
- Entropy below 0.15: bzip2 directly on the residual (BWT+MTF handles near-constant data better than zstd at this range)
- Entropy below 0.80: context-pack the bytes by preceding byte, then zstd
- Entropy above 0.80: context-pack into YUV420p frames, encode with AV1 (NVENC GPU) or HEVC — falling back to lossless libx264 on CPU

---

## Benchmark

Tested against zstd -19 and bzip2 -9 on five representative input types:

| Input | Size | zstd -19 | bzip2 -9 | iris | Notes |
|-------|------|----------|----------|------|-------|
| random binary | 5 MB | 1.00x | 1.00x | 1.00x | incompressible by definition |
| structured logs | 7.4 MB | 9.76x | 13.28x | **15.0x** | column grammar + HMS + dict |
| CSV export | 3.5 MB | 4.36x | 4.98x | **6.6x** | column grammar + type encoding |
| nearly-sorted u32s | 4 MB | 171x | 247x | **405x** | resonance lag=4 + bzip2 |
| repetitive binary | 2 MB | 3.29x | 3.23x | **3.30x** | beats zstd |

All roundtrips verified lossless by MD5.

---

## Usage

```bash
# Compress
iris compress input.csv output.iris

# Decompress
iris decompress output.iris recovered.csv

# Inspect without decompressing
iris info output.iris
```

---

## Format

Files use the `.iris` extension. The container format is `IRS2` (version 2) — a 18-byte header followed by six length-prefixed chunks carrying the resonance header, grammar data, prediction metadata, scatter map, context header, and encoded payload. The encoder byte and flags field identify exactly which stages fired, so the decompressor only runs the inverse of what the compressor used.

The format is designed to be self-describing and forward-compatible. A decoder that doesn't recognize a future flag can report it cleanly rather than producing corrupt output.

---

## Build

Requires Rust 1.75+. For GPU encoding, FFmpeg with NVENC support.

```bash
cargo build --release
```

The binary falls back to `libx264 -qp 0` (lossless, CPU) automatically if no NVENC hardware is detected at runtime. libsvtav1 is explicitly not used — its `-crf 0` mode is not lossless.

---

## Dependencies

- `zstd` — general-purpose compression for moderate-entropy data and column payloads
- `bzip2` — BWT+MTF for near-constant residuals (nearsorted, delta-coded binary)
- `clap` — CLI
- `anyhow` — error handling
- FFmpeg — video encode/decode via subprocess (av1_nvenc / hevc_nvenc / libx264)

---

## Project
Authored by Harsh Patel.
