# iris v0.2 — IDE Codebase Report
> Two bugs break roundtrips. Fix them in order. All source files are in this directory.

---

## Current Benchmark State

| File | iris ratio | Target | Roundtrip |
|------|-----------|--------|-----------|
| random.bin (5MB) | 1.00x | — | ✓ OK |
| logs.txt (7.4MB) | 14.995x | **15x** | ✗ FAIL |
| data.csv (3.5MB) | 6.64x | **6x** ✓ | ✗ FAIL |
| nearsorted.bin (4MB) | 265x | **300x** | ✗ FAIL |
| repetitive.bin (2MB) | 3.30x | — | ✓ OK |

---

## BUG 1 — Binary file triggers column grammar (nearsorted broken)

**File:** `grammar.rs` → `infer_columns()`, top of function

**Problem:** `nearsorted.bin` contains null bytes (0x00). The ASCII guard currently whitelists `\n`, `\r`, `\t` from non_print count — but does NOT reject nulls. So the guard passes even for binary data.

**Fix — add ONE line after the existing non_print check:**
```rust
// Add this immediately after the non_print * 20 > probe.len() check:
let has_nulls = probe.iter().any(|&b| b == 0);
if has_nulls { return None; }
```

**Effect:** nearsorted.bin → falls through to resonance → lag=4 → entropy=0.106 → bzip2 → ~405x

---

## BUG 2 — Prefix/suffix not reattached on decode (logs, CSV broken)

**File:** `grammar.rs` → `decode_spec()`, end of function

**Problem:** `encode_smart()` strips common prefix/suffix from column values before encoding (e.g. strips `[` and `]` from `[ERROR]`, strips `service=` from `service=7`). These are stored in `spec.pfx` and `spec.sfx`. But `decode_spec()` returns raw decoded values WITHOUT reattaching the prefix/suffix.

**Symptom:**
```
orig: '2026-01-01 19:41:51 [ERROR] service=7 msg="Session expired" latency=2506ms'
recv: '2026-01-01 19:41:51 ERROR 7 Session expired 2506'
```

**Fix — the end of `decode_spec()` should be:**
```rust
    // ... all the match arms ...

    // Reattach prefix/suffix to every decoded value
    if spec.pfx.is_empty() && spec.sfx.is_empty() {
        raw_rows
    } else {
        raw_rows.into_iter().map(|v| {
            let mut out = spec.pfx.clone();
            out.extend_from_slice(&v);
            out.extend_from_slice(&spec.sfx);
            out
        }).collect()
    }
}
```

Check the current `decode_spec()` — if the reattachment block exists but the variable is named `raw_rows`, make sure it's actually being RETURNED, not just computed and dropped. The match block must assign to `raw_rows`, then the reattachment runs after the match.

---

## Pipeline Architecture

```
Input
  → grammar::infer_columns()    ← text fast-path (CSV/logs), returns early if fires
  → profile::profile()          ← single pass: entropy, resonance lags, context table
  → Stage 1: resonance::extract ← wrapping i8 delta, up to 3 passes
  → Stage 2: grammar::infer     ← stride grammar (binary fixed-period data only)
  → Stage 3: prediction::build  ← MinHash XOR-delta dedup (skipped if H < 0.15)
  → Route:
      H < 0.15  → bzip2 direct           (nearsorted → 405x)
      H < 0.80  → zstd + context pack    (moderate entropy)
      H ≥ 0.80  → AV1/HEVC via NVENC     (GPU, high entropy)
```

## Container Format (IRS2)

```
Bytes 0-3:   "IRS2" magic
Byte  4:     version = 2
Byte  5:     encoder (0xFF=zstd, 0xFE=column-grammar, 0/1/2=AV1)
Byte  6:     flags (bit0=RESONANCE, bit1=GRAMMAR, bit2=PRED_GRAPH, bit3=CONTEXT_PACK)
Byte  7:     reserved = 0
Bytes 8-15:  original_len: u64 LE
Bytes 16-17: byte0, original_byte0
Then 6 chunks (each u32 LE length prefix):
  1. resonance_hdr  2. grammar_data  3. pred_meta
  4. scatter_map    5. ctx_hdr       6. video/payload
```

## Column Grammar Encoding Types

```
ENC_RAW    = 0   zstd of newline-joined original values
ENC_U8     = 1   u8 array + zstd  (digit-only cols 0-255)
ENC_U16    = 2   u16-LE array + zstd  (digit-only cols 0-65535)
ENC_U32    = 3   u32-LE array + zstd
ENC_DICT   = 4   null-sep dict (zstd) + u8 indices (zstd)  (≤256 unique values)
ENC_HMS    = 5   HH:MM:SS → 3 separate u8 arrays, each zstd
ENC_MERGED = 6   two adjacent dict-encoded cols merged as one dict entry
ENC_EMPTY  = 0xFF  placeholder for a col absorbed into the previous ENC_MERGED
```

## Build & Test Commands

```bash
cd /path/to/iris2

# After making fixes:
touch src/*.rs
cargo build --release 2>&1 | grep "^error"

# Full benchmark with roundtrip check:
IRIS=./target/release/iris
for entry in "random:inputs/random.bin" "logs:inputs/logs.txt" "csv:inputs/data.csv" "nearsorted:inputs/nearsorted.bin" "repetitive:inputs/repetitive.bin"; do
    name="${entry%%:*}"; input="/tmp/iris_bench/${entry##*:}"
    [ -f "$input" ] || continue
    $IRIS compress "$input" "/tmp/t_${name}.iris" 2>/dev/null
    $IRIS decompress "/tmp/t_${name}.iris" "/tmp/t_${name}.out" 2>/dev/null
    orig_sz=$(wc -c < "$input")
    comp_sz=$(wc -c < "/tmp/t_${name}.iris")
    ratio=$(python3 -c "print(f'{$orig_sz/$comp_sz:.2f}x')")
    result=$([ "$(md5sum "$input"|cut -d' ' -f1)" = "$(md5sum "/tmp/t_${name}.out" 2>/dev/null|cut -d' ' -f1)" ] && echo "OK" || echo "FAIL")
    echo "$name: $ratio  roundtrip=$result"
done
```

## Expected Results After Both Fixes

```
random:      1.00x   roundtrip=OK
logs:        ~15.0x  roundtrip=OK
csv:         ~6.6x   roundtrip=OK
nearsorted:  ~405x   roundtrip=OK
repetitive:  3.30x   roundtrip=OK
```
