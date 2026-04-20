#!/usr/bin/env python3
"""Synthetic generators for genomic-style datasets.

Produces files that have the same *statistical structure* as real public
datasets but without requiring any network access. Useful for running the
entropy-based structure-detection harness offline.

Generators:
  * vcf       — tab-separated variant records, mimicking 1000 Genomes VCF
  * fastq     — 4-line FASTQ records with quality strings (ASCII 33-73)
  * rnaseq    — newline-separated integer counts per gene, log-linear
  * mutmatrix — binary mutation matrix (samples × positions) written as
                packed 0/1 bytes

The sizes default to ~10 MB, which is within the 5–50 MB envelope the
study uses. Pass --mb=N to change.
"""
from __future__ import annotations
import argparse
import os
import random
import sys


CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
NUCLEOTIDES = "ACGT"


def gen_vcf(out_path: str, mb: int, seed: int = 0xdead) -> None:
    rng = random.Random(seed)
    size = mb * 1024 * 1024
    header = (
        "##fileformat=VCFv4.2\n"
        "##reference=GRCh38\n"
        "##INFO=<ID=AF,Number=1,Type=Float,Description=\"Allele freq\">\n"
        "##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Depth\">\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    )
    with open(out_path, "wb") as f:
        f.write(header.encode())
        pos = 10_000
        chrom_i = 0
        written = len(header)
        while written < size:
            chrom = CHROMS[chrom_i % len(CHROMS)]
            # Positions are monotone within a chromosome; small jitter.
            pos += rng.randint(20, 8_000)
            ref = rng.choice(NUCLEOTIDES)
            alt = rng.choice([n for n in NUCLEOTIDES if n != ref])
            af = rng.random() * 0.5
            dp = rng.randint(5, 120)
            rid = f"rs{rng.randint(100000, 999999999)}"
            line = f"{chrom}\t{pos}\t{rid}\t{ref}\t{alt}\t{rng.randint(20,80)}\tPASS\tAF={af:.4f};DP={dp}\n"
            f.write(line.encode())
            written += len(line)
            # Advance chromosome every ~5% of data.
            if rng.random() < 1e-4:
                chrom_i += 1
                pos = 10_000


def gen_fastq(out_path: str, mb: int, read_len: int = 150, seed: int = 0xbeef) -> None:
    rng = random.Random(seed)
    size = mb * 1024 * 1024
    with open(out_path, "wb") as f:
        written = 0
        i = 0
        while written < size:
            header = f"@SRR1234567.{i} HWI-ST1276:71:C1162ACXX:1:1101:{rng.randint(1000,99999)}:{rng.randint(1000,99999)} length={read_len}\n"
            seq = "".join(rng.choice(NUCLEOTIDES) for _ in range(read_len)) + "\n"
            plus = "+\n"
            # Quality string: Phred+33, slight declining bias toward 3' end.
            qs = bytearray()
            for p in range(read_len):
                base = 73 - int(p / read_len * 10)  # 73 → 63
                qs.append(max(33, min(73, base + rng.randint(-5, 3))))
            q = qs.decode("latin-1") + "\n"
            rec = header + seq + plus + q
            f.write(rec.encode())
            written += len(rec)
            i += 1


def gen_rnaseq(out_path: str, mb: int, n_genes: int = 60_000, seed: int = 0xc0de) -> None:
    rng = random.Random(seed)
    size = mb * 1024 * 1024
    # Gene counts follow a roughly log-normal distribution. The file is
    # `gene_id \t count \n`.
    with open(out_path, "wb") as f:
        written = 0
        gi = 0
        while written < size:
            gene = f"ENSG{gi:011d}"
            # log-normal count
            c = int(rng.lognormvariate(3.0, 1.8))
            line = f"{gene}\t{c}\n"
            f.write(line.encode())
            written += len(line)
            gi = (gi + 1) % n_genes


def gen_mutmatrix(out_path: str, mb: int, n_samples: int = 500, seed: int = 0xfeed) -> None:
    """Binary mutation matrix: samples × positions, 1 = variant, 0 = ref.
    Strong sparsity (~1% density), rows stored contiguously. Very
    regular structure; strongly periodic at stride=n_positions.
    """
    rng = random.Random(seed)
    size = mb * 1024 * 1024
    n_positions = max(1024, size // n_samples)
    with open(out_path, "wb") as f:
        for _ in range(n_samples):
            row = bytearray(b"\x00" * n_positions)
            # sprinkle variants
            nvar = n_positions // 100
            for _ in range(nvar):
                row[rng.randrange(n_positions)] = 1
            f.write(row)


GENS = {
    "vcf":       gen_vcf,
    "fastq":     gen_fastq,
    "rnaseq":    gen_rnaseq,
    "mutmatrix": gen_mutmatrix,
}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="synthetic genomic-ish data")
    ap.add_argument("kind", choices=list(GENS))
    ap.add_argument("output")
    ap.add_argument("--mb", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0xdead)
    args = ap.parse_args(argv[1:])
    GENS[args.kind](args.output, mb=args.mb, seed=args.seed)
    print(f"wrote {args.output} ({os.path.getsize(args.output)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
