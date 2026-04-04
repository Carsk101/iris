# iris

**GPU-accelerated compression using NVENC + VisionSort.**

iris encodes arbitrary files as lossless AV1 video, using NVIDIA's dedicated
encode silicon (NVENC) as a compression engine. The trick: files are sorted
with VisionSort before encoding. Sorted data has maximum run-length structure —
adjacent similar values cluster together. When laid across video frames
sequentially, AV1's inter-frame motion estimator treats those runs as temporally
stable blocks and compresses them aggressively.

VisionSort's entropy-based routing (NearlyFree / Verify / PlacementSort / FullSort)
also tells iris which regions of the permutation index are nearly deterministic,
allowing cheaper index storage on low-entropy inputs.

---

## pipeline

```
raw bytes
  -> VisionSort  (sort + build entropy model + permutation index)
  -> frame layout (sorted bytes packed into YUV420p frames)
  -> NVENC encode (av1_nvenc lossless, or hevc_nvenc / libsvtav1 fallback)
  -> .iris container (video bitstream + zstd-compressed permutation index)

decompress reverses:
  -> NVDEC decode  (via ffmpeg)
  -> unpack Y planes -> sorted bytes
  -> unsort via permutation index -> original bytes
```

---

## requirements

- Rust 1.75+
- FFmpeg with NVENC support (`ffmpeg -encoders | grep nvenc`)
- NVIDIA GPU (RTX 3000+ for hevc_nvenc, RTX 4000+ for av1_nvenc)
- No GPU: falls back to libsvtav1 (CPU AV1, same pipeline, no NVENC)

---

## install

```bash
git clone https://github.com/harshpatel/iris
cd iris
cargo build --release
# binary at target/release/iris
```

---

## usage

```bash
# compress
iris compress input.bin output.iris

# decompress
iris decompress output.iris recovered.bin

# inspect without decompressing
iris info output.iris
```

---

## output

```
[iris] encoder: av1_nvenc (RTX 4000+ detected)
[iris] sorting with VisionSort...
[iris] sorted in 42.3ms | route: PlacementSort | entropy: 5.821 bits
[iris] frame layout: 1920x1080 x 3 frames
[iris] encoded 3 frames (1920x1080) -> 1247832 bytes
[iris] done in 1203ms
[iris] 6220800 bytes -> 1284291 bytes | ratio: 4.845x | savings: 79.4%
```

---

## .iris format

| offset | field | type |
|--------|-------|------|
| 0 | magic `IRIS` | 4 bytes |
| 4 | version | u8 |
| 5 | original_len | u64 le |
| 13 | width | u32 le |
| 17 | height | u32 le |
| 21 | frame_count | u32 le |
| 25 | route | u8 + 3 reserved |
| 29 | perm_len | u64 le |
| 37 | permutation | zstd-compressed u32[] |
| 37+perm_len | video | AV1/HEVC mkv bitstream |

---

## encoder detection

| GPU | encoder | notes |
|-----|---------|-------|
| RTX 4000+ | av1_nvenc | best ratio |
| RTX 3000 | hevc_nvenc | fallback |
| none | libsvtav1 | CPU, same pipeline |

---

## project lineage

iris is built on [VisionSort](https://github.com/harshpatel/visionsort) —
a sorting algorithm that treats sorting as a perception problem, using
Bayesian within-call learning and entropy-based routing to characterize
data distribution during the sort itself.

---

*Harsh Patel, 2026*
