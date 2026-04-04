# iris

A compression engine that treats compression as a perception problem. Most compressors apply the same algorithm to everything. iris reads the data once, understands its structure, then compresses each structural property with the technique that eliminates it most completely.

Built in Rust. Zero runtime compression dependencies — every byte of entropy coding is iris-native.

---

## Architecture

iris runs data through a profiling pass, then selectively activates up to five compression stages. Each stage strips one layer of structure, reducing entropy for the next. The pipeline is fully adaptive: stages that don't help are skipped, and the container records exactly which stages fired so the decompressor only runs their inverses.

```
                          ┌─────────────┐
                     ┌───▶│ Column      │──┐
                     │    │ Grammar     │  │
                     │    └─────────────┘  │
┌─────────┐    ┌─────┴──┐                 │    ┌──────────┐    ┌───────────┐
│  Input   │───▶│Profile │                ├───▶│  Route   │───▶│ .iris     │
│  File    │    │  Pass  │                │    │  + Encode│    │ Container │
└─────────┘    └─────┬──┘                │    └──────────┘    └───────────┘
                     │    ┌─────────────┐  │
                     ├───▶│ Resonance   │──┤
                     │    └─────────────┘  │
                     │    ┌─────────────┐  │
                     ├───▶│ Stride      │──┤
                     │    │ Grammar     │  │
                     │    └─────────────┘  │
                     │    ┌─────────────┐  │
                     ├───▶│ Prediction  │──┤
                     │    │ Graph       │  │
                     │    └─────────────┘  │
                     │    ┌─────────────┐  │
                     └───▶│ Context     │──┘
                          │ Pack        │
                          └─────────────┘
```

---

## Stage 0 — Profile

A single read-only pass measures everything downstream needs:

- **Byte entropy** (Shannon, normalized to 0–1) — sampled at up to 65,536 points for speed on large files.
- **Autocorrelation** at all lags 1–64 using wrapping `i8` arithmetic. The signed wrapping difference treats the byte ring symmetrically — `lag=1` on sequential data gives `diff=1` everywhere, not `-255`. Sampled every 4th byte and compared to the theoretical baseline MSD of uniform noise (5461).
- **Context frequency table** — for each of the 256 possible preceding bytes, stores the positions of every byte that follows it. This drives context packing.
- **Prediction accuracy** — an exponential moving average of byte-to-byte surprise.
- **Disorder estimate** — random-pair inversions, approximating sortedness.
- **Block fingerprints** — FNV-1a hashes of 4KB blocks for gate decisions.

The profile feeds a `StageGate` that decides which stages to activate:

| Condition | Effect |
|-----------|--------|
| `H > 0.97` | Passthrough — data is incompressible |
| `H < 0.92` | Enable resonance |
| `prediction_accuracy > 0.45 && H < 0.80` | Enable stride grammar |
| `block_count > 1` | Enable prediction graph |
| `H < 0.97` | Enable context packing |

---

## Column Grammar (text fast-path)

Fires before profiling for structured text (logs, CSV, TSV, PSV). Detects the delimiter by variance analysis across the first 20 lines, then splits into columns and encodes each one with the type that fits it best:

| Type | When | Encoding |
|------|------|----------|
| **Constant** | All values identical | Single value stored once |
| **HMS time** | `HH:MM:SS` pattern | Three separate `u8` arrays (H, M, S), each range-coded |
| **Integer** | All-digit values | Packed `u8`/`u16`/`u32` arrays, range-coded |
| **Dictionary** | ≤256 unique values | Null-separated dictionary + `u8` index array, both range-coded |
| **Raw** | Everything else | Newline-joined values, range-coded |

Before type encoding, each column's common prefix and suffix are stripped and stored separately — restored on decode.

Adjacent dictionary columns that always co-occur as fixed pairs are merged into a single dictionary entry when it saves space. The column grammar is only used if its total payload beats raw range coding of the entire file.

---

## Stage 1 — Resonance

Detects periodic structure using the autocorrelation lags from the profile. For data with a dominant period *L*:

```
residual[i] = data[i] wrapping_sub data[i - L]
```

Periodic data at that lag produces near-zero residuals. The first *L* bytes are stored verbatim as the reconstruction seed.

Runs up to **three passes** — each pass re-profiles the residual and extracts the next strongest lag. Stops early if the residual entropy drops below 0.15 (the range coder will take it from there). Each pass's lag, strength, and prefix are stacked in the container; decompression applies them in reverse order.

---

## Stage 2 — Stride Grammar

For binary data with fixed-stride records (structs serialized to disk, fixed-width database pages). Uses the dominant resonance lag as the stride, then:

1. For each byte position within the stride, collect all values across records.
2. Positions with entropy below 0.3 are classified as **fixed** — store the mode value once.
3. Remaining positions are **variable fields** — their values are separated out and compressed independently.

Only fires when ≥25% of stride positions are fixed (otherwise the template overhead doesn't pay for itself).

---

## Stage 3 — Prediction Graph

Finds similar blocks anywhere in the file using **SimHash** fingerprints — no window limit. LZ77 is physically constrained to its sliding window (typically 8–32 MB). iris has no such limit.

**SimHash computation:** For each 4KB block, slide a 4-byte shingle window. Hash each shingle with FNV-1a to get a 64-bit feature hash. Accumulate a 64-dimensional signed weight vector (`+1` for set bits, `-1` for clear bits). Final fingerprint = sign bits. This correctly approximates cosine similarity via Hamming distance.

**Neighbor lookup:** Uses Locality-Sensitive Hashing (LSH) with 4 bands × 16 bits each for O(n) expected-time nearest-neighbor search. Blocks sharing any band signature are candidates; the best match by Hamming similarity above 0.55 is selected.

**Delta encoding:** Similar blocks are stored as XOR deltas from their nearest match. The delta is only used if it reduces entropy by ≥15% compared to the raw block. XOR deltas of similar blocks are highly compressible — often near-zero.

---

## Stage 4 — Context Packing

Groups bytes by their preceding byte (context). All bytes that follow the same context byte land in the same bucket. This clusters statistically similar bytes together, making the entropy coder's job easier.

Two modes:

- **Implicit chain reconstruction** — stores zero position data. Decompression follows the context chain: each decoded byte tells you which context buffer the next byte comes from. O(N) with no scatter map.
- **Explicit scatter** — positions within each bucket are stored as varint-delta compressed lists (monotone increasing → small deltas → highly compressible). Used on the rANS route.

---

## Stage 5 — Entropy Coding (three native routes)

The working data's entropy after stages 1–4 determines which entropy coder handles it:

### Route A — Range Coder (`H < 0.15`)

For near-constant residuals (post-resonance data that's almost all zeros). A 64-bit state range coder with:

- **Sparse frequency table** — only non-zero symbols are stored in the header (`u16` count + `(u8 sym, u32 freq)` pairs), not a flat 1024-byte table. For post-resonance residuals with 3–4 unique symbols, the header is ~20 bytes.
- **Exact frequencies** — no model adaptation, no smoothing. The table is computed from this specific data and encodes to the theoretical minimum.
- **Corruption guard** — if a corrupt stream points to a zero-frequency symbol, the decoder returns `None` instead of hanging.

Encodes to near-theoretical entropy minimum. A 10,000-byte buffer with 3 unique symbols compresses to <100 bytes.

### Route B — rANS (`0.15 ≤ H < 0.80`)

An order-1 context-conditioned rANS (asymmetric numeral systems) coder:

- **256 context tables** — one per preceding byte, each with up to 256 symbol frequencies normalized to a power-of-two total (4096). Only active contexts are stored.
- **Proportional normalization** — raw counts are scaled to sum to 4096 with rounding correction on the most frequent symbol. Every symbol that appeared gets at least frequency 1.
- **Sparse table storage** — per context, writes only non-zero `(sym, freq)` pairs.
- **Encode backward, decode forward** — natural rANS order.
- **Renormalization threshold** at `1 << 23` — pushes/pulls bytes to keep state in range.

After context grouping, LZ77 finds almost nothing useful. What remains is per-symbol entropy coding, and rANS matches arithmetic coding's compression ratio with faster decode (shift instead of divide on the power-of-two total).

### Route C — Hierarchical Block Matcher (`H ≥ 0.80`)

For high-entropy data where neither range coding nor rANS+context helps much. Replaces the old ffmpeg/AV1 subprocess:

- **4KB blocks** with SimHash fingerprints + LSH for O(n) matching.
- **Subdivides to 1KB** if the XOR delta entropy exceeds 0.70 — searches a 16-block lookback window (bounded brute force: at most 64 SimHash comparisons per sub-block).
- **Split rANS streams** — raw block payloads and delta payloads are collected into separate buffers and rANS-encoded independently for better statistical modeling.
- **XOR delta + rANS residual coding** — delta must save at least 10% entropy to be used; otherwise the block is stored raw.

---

## Container Format

Files use the `.iris` extension. Container format: `IRS2` version 3.

```
[4B magic: "IRS2"]
[1B version][1B encoder][1B flags][1B reserved]
[8B original_len LE]
[1B byte0][1B original_byte0]
[6 × length-prefixed chunks:]
  - resonance header
  - grammar data
  - prediction metadata
  - scatter map
  - context header
  - encoded payload
[4B CRC32 trailer]
```

**Encoder byte registry:**

| Byte | Encoder |
|------|---------|
| `0xFF` | rANS + context pack |
| `0xFE` | Column grammar |
| `0xFD` | Hierarchical block matcher |
| `0xFC` | Range coder |
| `0xFB` | Raw passthrough (rANS expansion guard) |

**Flags** (bitmask):

| Bit | Meaning |
|-----|---------|
| `0001` | Resonance active |
| `0010` | Grammar active |
| `0100` | Prediction graph active |
| `1000` | Context pack / entropy coder active |

The CRC32 trailer (IEEE 802.3 polynomial, native implementation) covers all chunk data. Corruption is detected on read with an exact mismatch diagnostic.

The format is self-describing and forward-compatible. A decoder that doesn't recognize a future encoder byte or flag reports it cleanly rather than producing corrupt output.

---

## Passthrough Guard

When data is near-incompressible (`H > 0.97`), iris still attempts rANS encoding but compares the output size to the input. If rANS expands the data, iris stores it raw with encoder byte `0xFB`. This guarantees iris never makes a file larger than the original (plus the ~30-byte container overhead).

---

## Usage

```
# Compress
iris compress input.csv output.iris

# Decompress
iris decompress output.iris recovered.csv

# Inspect without decompressing
iris info output.iris
```

---

## Build

Requires Rust 1.75+.

```
cargo build --release
```

---

## Dependencies

```toml
[dependencies]
clap   = { version = "=4.4.18", features = ["derive"] }
anyhow = "=1.0.75"
```

That's it. All compression logic — range coding, rANS, block matching, SimHash, LSH, CRC32, varint encoding — is implemented natively in iris. No bzip2, no zstd, no ffmpeg.

---

## Source Map

```
system/src/
├── main.rs          CLI entry point (41 lines)
├── profile.rs       Stage 0: single-pass profiling + stage gate (189 lines)
├── resonance.rs     Stage 1: multi-pass periodic structure extraction (151 lines)
├── grammar.rs       Stage 2: column grammar + stride grammar (679 lines)
├── prediction.rs    Stage 3: SimHash + LSH prediction graph (305 lines)
├── context_pack.rs  Stage 4: context-aware byte grouping (376 lines)
├── pipeline.rs      Orchestrator: compress / decompress / info (426 lines)
├── range_coder.rs   Stage 5a: 64-bit range coder (289 lines)
├── rans.rs          Stage 5b: order-1 rANS coder (411 lines)
├── block_match.rs   Stage 5c: hierarchical block matcher (571 lines)
└── container.rs     IRS2/v3 container + CRC32 + varint scatter (303 lines)
```

Total: ~3,750 lines of Rust. 44 tests.

---

## How iris beats general-purpose compressors

General-purpose compressors (zstd, bzip2, lz4) apply one strategy to all data. iris exploits structure they can't see:

- **Resonance** eliminates periodicity before entropy coding touches it. A sorted `u32` array with `lag=4` produces residuals of ~1 — zstd sees the original values.
- **Column grammar** encodes each column at its type-theoretic minimum. Timestamps as three `u8` arrays, not as ASCII strings. Integer columns as packed binary, not decimal digits.
- **Prediction graph** finds similar blocks at arbitrary distance. LZ77 can't reference a block 500 MB away. iris can.
- **Context packing** clusters bytes by statistical neighborhood before entropy coding. The entropy coder sees runs of similar values instead of interleaved contexts.
- **Three-way routing** picks the right entropy coder for the residual's actual distribution instead of forcing one algorithm on everything.

---

## Project

Authored by Harsh Patel.
Version 1.0.0 — zero-dependency native compression.
