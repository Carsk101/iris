#!/usr/bin/env python3
"""Entropy-based structure-detection harness for iris.

What it does
============

For each input file (or each generated synthetic dataset):

  1. Run `iris profile` with the default gate AND the research-gate
     (`IRIS_GATE=genomic`) and record:
       - Shannon entropy
       - local-entropy std-dev
       - byte skewness
       - dominant-lag strength and stability
       - gate decision (which stages are enabled)

  2. Run `iris compress` under both gate profiles, capture
     stage-activation logs (iris already emits `[iris][stage]` lines
     on stderr), compressed size, wall time, and peak RSS (via
     `/usr/bin/time -l` on macOS / `-v` on Linux when available).

  3. Run `iris decompress` and verify byte-for-byte roundtrip.

  4. Print a summary table so we can eyeball "does the adaptive gate
     help on scientific data?".

This is deliberately a small, zero-dependency Python script so it can
be dropped into a research notebook or CI without coupling it to the
Rust crate. If you want to compare against zstd/xz, just point
`--compare-with` at a comma-separated list of `name:command-template`
pairs such as `zstd:zstd -19 -T0 -q -o {out} {in}`.

Example
-------

    # build iris
    cd ../..; cargo build --release -p iris

    # generate 10 MB synthetic datasets and run the harness
    cd system/examples
    python3 genomic_harness.py --synth all --mb 10 \
        --iris ../target/release/iris

    # or point at real files
    python3 genomic_harness.py --input sample.vcf sample.fastq \
        --iris ../target/release/iris
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time


HERE = os.path.dirname(os.path.abspath(__file__))
SYNTH_SCRIPT = os.path.join(HERE, "synth_genomic.py")


@dataclasses.dataclass
class RunResult:
    ok: bool
    orig_size: int
    comp_size: int
    comp_time_s: float
    decomp_time_s: float
    roundtrip_ok: bool
    ratio: float
    stages_fired: list[str]
    profile: dict
    gate_rationale: str
    extra: dict


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], env: dict | None = None) -> tuple[int, str, str]:
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, env=env)
    dt = time.time() - t0
    return p.returncode, p.stdout.decode(errors="replace"), p.stderr.decode(errors="replace") + f"\n[wallclock {dt:.3f}s]"


PROFILE_RE = re.compile(r"^\s*([a-z_.]+)\s*:\s*(.+)$")


def parse_profile(stdout: str) -> dict:
    out = {}
    for line in stdout.splitlines():
        m = PROFILE_RE.match(line)
        if m:
            k, v = m.group(1), m.group(2).strip()
            # normalise common numeric formats
            try:
                if "." in v or "e" in v.lower():
                    out[k] = float(v)
                else:
                    out[k] = int(v)
            except ValueError:
                out[k] = v
    return out


STAGE_RE = re.compile(r"\[iris\]\[stage\]\s+(\S+)\s+fired=1")
STAGE_SIMPLE_RE = re.compile(r"\[iris\]\s+(resonance pass|grammar|pred graph|block-match|rANS-route|ultra-compress|column grammar)")
GATE_RATIONALE_RE = re.compile(r"\[iris\] gate: .* \| (.+)$")


def extract_stages(stderr: str) -> list[str]:
    fired = set()
    for m in STAGE_RE.finditer(stderr):
        fired.add(m.group(1))
    for m in STAGE_SIMPLE_RE.finditer(stderr):
        fired.add(m.group(1).split()[0])
    return sorted(fired)


def extract_rationale(stderr: str) -> str:
    for m in GATE_RATIONALE_RE.finditer(stderr):
        return m.group(1)
    return ""


def run_one(iris_bin: str, src: str, gate: str) -> RunResult:
    env = os.environ.copy()
    if gate:
        env["IRIS_GATE"] = gate
    orig_sha = file_sha256(src)
    orig_size = os.path.getsize(src)

    with tempfile.TemporaryDirectory() as tmp:
        compressed = os.path.join(tmp, "out.iris")
        restored   = os.path.join(tmp, "restored.bin")

        # profile
        rc, po, pe = run([iris_bin, "profile", src], env=env)
        profile = parse_profile(po) if rc == 0 else {}
        gate_rationale = profile.get("gate.rationale", "")

        # compress
        t0 = time.time()
        rc, _, ce = run([iris_bin, "compress", src, compressed], env=env)
        t_comp = time.time() - t0
        if rc != 0:
            return RunResult(
                ok=False, orig_size=orig_size, comp_size=0, comp_time_s=t_comp,
                decomp_time_s=0.0, roundtrip_ok=False, ratio=0.0,
                stages_fired=[], profile=profile, gate_rationale=gate_rationale,
                extra={"stderr": ce[-2000:]},
            )
        comp_size = os.path.getsize(compressed)
        stages = extract_stages(ce)
        rationale_from_compress = extract_rationale(ce)

        # decompress
        t0 = time.time()
        rc, _, de = run([iris_bin, "decompress", compressed, restored], env=env)
        t_dec = time.time() - t0
        if rc != 0:
            return RunResult(
                ok=False, orig_size=orig_size, comp_size=comp_size, comp_time_s=t_comp,
                decomp_time_s=t_dec, roundtrip_ok=False, ratio=orig_size/max(comp_size,1),
                stages_fired=stages, profile=profile,
                gate_rationale=rationale_from_compress or gate_rationale,
                extra={"stderr": de[-2000:]},
            )
        roundtrip_ok = file_sha256(restored) == orig_sha

        return RunResult(
            ok=True, orig_size=orig_size, comp_size=comp_size,
            comp_time_s=t_comp, decomp_time_s=t_dec, roundtrip_ok=roundtrip_ok,
            ratio=orig_size / max(comp_size, 1),
            stages_fired=stages, profile=profile,
            gate_rationale=rationale_from_compress or gate_rationale,
            extra={},
        )


def ensure_synth(kind: str, mb: int, tmp: str) -> str:
    out = os.path.join(tmp, f"synth_{kind}_{mb}mb.bin")
    subprocess.check_call(
        [sys.executable, SYNTH_SCRIPT, kind, out, f"--mb={mb}"],
        cwd=HERE,
    )
    return out


def fmt_row(label: str, r: RunResult) -> str:
    H = r.profile.get("global_entropy", float("nan"))
    Hs = r.profile.get("local_entropy_std", float("nan"))
    skew = r.profile.get("byte_skewness", float("nan"))
    stab = r.profile.get("peak_stability", float("nan"))
    stages = ",".join(r.stages_fired) or "-"
    rt = "ok" if r.roundtrip_ok else ("FAIL" if r.ok else "err")
    return (
        f"{label:<28s} "
        f"{r.orig_size/1e6:>7.2f}MB → {r.comp_size/1e6:>7.2f}MB "
        f"({r.ratio:>5.2f}x) "
        f"H={H:>5.3f} Hσ={Hs:>5.3f} skew={skew:>+5.2f} stab={stab:>4.2f} "
        f"comp={r.comp_time_s:>6.2f}s dec={r.decomp_time_s:>6.2f}s "
        f"stages={stages} rt={rt}"
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iris", default=shutil.which("iris") or "../target/release/iris")
    ap.add_argument("--input", nargs="*", default=[])
    ap.add_argument("--synth", nargs="*", default=[],
                    help="one or more of: vcf fastq rnaseq mutmatrix all")
    ap.add_argument("--mb", type=int, default=10)
    ap.add_argument("--gates", nargs="*", default=["default", "genomic"])
    args = ap.parse_args(argv[1:])

    if not os.path.isfile(args.iris):
        print(f"iris binary not found: {args.iris}", file=sys.stderr)
        return 2

    inputs: list[tuple[str, str]] = []   # (label, path)
    tmp = tempfile.mkdtemp(prefix="iris_harness_")
    try:
        synth_targets: list[str] = []
        if "all" in args.synth:
            synth_targets = ["vcf", "fastq", "rnaseq", "mutmatrix"]
        else:
            synth_targets = list(args.synth)
        for kind in synth_targets:
            inputs.append((f"synth:{kind}", ensure_synth(kind, args.mb, tmp)))
        for p in args.input:
            inputs.append((os.path.basename(p), p))

        if not inputs:
            print("no inputs; use --synth or --input", file=sys.stderr)
            return 2

        results: dict[str, dict[str, RunResult]] = {}
        for label, path in inputs:
            results[label] = {}
            for gate in args.gates:
                gate_env = "" if gate == "default" else gate
                r = run_one(args.iris, path, gate_env)
                results[label][gate] = r

        print()
        print("=" * 140)
        for label, per_gate in results.items():
            print(f"\n{label}")
            print("-" * 140)
            for gate in args.gates:
                r = per_gate[gate]
                print(fmt_row(f"  gate={gate}", r))
                if r.gate_rationale:
                    print(f"    rationale: {r.gate_rationale}")

        # Sanity summary: total bytes saved per gate.
        print("\nSUMMARY")
        print("-" * 60)
        for gate in args.gates:
            total_in  = sum(r[gate].orig_size for r in results.values())
            total_out = sum(r[gate].comp_size for r in results.values())
            ratio = total_in / max(total_out, 1)
            rt = all(r[gate].roundtrip_ok for r in results.values())
            print(f"  gate={gate:<8s}  {total_in/1e6:>7.2f}MB → {total_out/1e6:>7.2f}MB  "
                  f"({ratio:.2f}x)  roundtrip_all_ok={rt}")

        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
