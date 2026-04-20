# iris — genomic / scientific-data harness

This folder holds a small, self-contained test harness for the
"Entropy-Based Detection of Structure in Genomic Data"-style
experiment: we want to see whether iris’s adaptive gate correctly
identifies and exploits structure in medium-entropy scientific datasets
(VCF, FASTQ, RNA-seq, mutation matrices) without being fooled by their
size.

The harness is deliberately *outside* the Rust crate — it shells out
to the `iris` binary via `subprocess`. That way you can drop it into a
notebook, CI job, or research environment without touching the Cargo
graph.

## Files

| File                  | Purpose                                                  |
| --------------------- | -------------------------------------------------------- |
| `synth_genomic.py`    | Offline synthetic generators (VCF, FASTQ, RNA-seq, mutmatrix) |
| `genomic_harness.py`  | Profile/compress/decompress/verify loop with summary     |

## Typical usage

```bash
# 1. Build iris in release mode
cd ../..
cargo build --release -p iris

# 2. Run the harness on 10 MB synthetic datasets across both the
#    production gate and the research (genomic) gate
cd system/examples
python3 genomic_harness.py \
    --iris ../target/release/iris \
    --synth all \
    --mb 10

# 3. Run on a real dataset (e.g. a 1000 Genomes VCF chunk, an SRA
#    FASTQ slice, or a TCGA mutation matrix extract)
python3 genomic_harness.py \
    --iris ../target/release/iris \
    --input /data/1000g_chr22_5m.vcf \
            /data/SRR123_R1_10m.fastq \
            /data/tcga_mutations_20m.tsv
```

## What the harness prints

For every input and every gate profile it shows:

- input and output sizes, ratio, compress/decompress wall times,
- the five profiling statistics that the adaptive gate reads
  (`H`, `Hσ`, `skew`, `peak_stability`),
- which iris stages fired (`resonance`, `grammar`, `pred-graph`,
  `block-match`, `rANS-route`, `ultra-compress`, `column`),
- the gate’s own human-readable rationale string,
- a roundtrip-check column that fails loudly if compression is lossy.

A summary row at the bottom gives the aggregate compression ratio per
gate so you can quickly tell whether the `genomic` gate profile
actually helps.

## Gate profiles

- **`default`** — production thresholds, biased toward safety (e.g.
  never makes files bigger).
- **`genomic`** — research profile in `AdaptiveGate::genomic_research`
  that lowers the entropy floors, raises the skew/stability
  rescues, and is specifically tuned for 5–50 MB scientific inputs.
  Enable it by setting `IRIS_GATE=genomic` (the harness does this
  for you when you pass `--gates genomic`).

## Extending the harness

Add a new generator to `synth_genomic.py`:

```python
def gen_bigwig(out_path, mb, seed=0):
    ...

GENS["bigwig"] = gen_bigwig
```

…and it becomes usable from the harness immediately:

```bash
python3 genomic_harness.py --synth bigwig --mb 20
```

For comparisons against zstd/xz/bzip2, extend `genomic_harness.py`
with a `--compare-with` option — the `run_one` helper is already
factored to make that straightforward.
