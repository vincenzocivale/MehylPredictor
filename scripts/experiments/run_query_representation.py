#!/usr/bin/env python3
"""Run the four locus-attention query representations sequentially.

Architecture selection only: every arm uses mode=development with the SAME
inner split (split_seed=17 by default) and the SAME final locus-attention recipe.
The only changed factor is --query-source.

Default order:
  Q0 ntv3 -> Q1 mean_only -> Q2 hybrid_detached -> Q3 hybrid_joint

A failed arm stops the sequence; completed arms are skipped and incomplete arms
with checkpoints/last.pt are resumed automatically.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE = "configs/models/rna_methylation_locus_attention.yaml"
STUDY = "query_representation_2026_09"
ARMS = (
    ("q0_ntv3", "ntv3"),
    ("q1_mean_only", "mean_only"),
    ("q2_hybrid_detached", "hybrid_detached"),
    ("q3_hybrid_joint", "hybrid_joint"),
)


def paths(data_root: str | None) -> dict[str, str]:
    root = Path(data_root or os.environ.get("METHYL_DATA_ROOT", "/dune/DATASETS/MethylPredictionData"))
    prepared = root / "derived" / "methylprophet_table5_tcga_chr1"
    canonical = root / "datasets" / "methylprophet_repro_v1"
    return {
        "canonical": str(canonical),
        "prepared": str(prepared),
        "features": str(prepared / "features"),
        "rna": str(prepared / "rna"),
        "registry": str(canonical / "cpg" / "registries" / "array_cpg_map.parquet"),
        "targets": str(root / "derived" / "cpg_statistics" / "chr1"),
        "output": str(root / "experiments"),
    }


def run_id(arm: str, seed: int) -> str:
    return f"query-representation-{arm}-seed{seed}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--data-root")
    ap.add_argument("--seed", type=int, default=17, help="training/init seed")
    ap.add_argument("--split-seed", type=int, default=17, help="fixed inner split seed")
    ap.add_argument("--early-stop-patience", type=int, default=6)
    ap.add_argument("--arms", help="optional comma-separated subset of q0_ntv3,q1_mean_only,q2_hybrid_detached,q3_hybrid_joint")
    ap.add_argument("--print-commands", action="store_true")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    p = paths(args.data_root)
    wanted = None if not args.arms else {x.strip() for x in args.arms.split(",") if x.strip()}
    selected = [x for x in ARMS if wanted is None or x[0] in wanted]
    if wanted is not None and wanted != {x[0] for x in selected}:
        unknown = sorted(wanted - {x[0] for x in selected})
        raise SystemExit(f"unknown arm(s): {', '.join(unknown)}")

    log_dir = REPO_ROOT / "logs" / STUDY
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONPATH": str(REPO_ROOT / "src")}

    for arm, query_source in selected:
        rid = run_id(arm, args.seed)
        run_dir = Path(p["output"]) / "runs" / "locus_cls_joint" / "chr1" / rid
        summary = run_dir / "training" / "summary.json"
        last = run_dir / "checkpoints" / "last.pt"
        cmd = [
            sys.executable, "scripts/train.py",
            "--model", "rna_methylation", "--scope", "chr1",
            "--engine", "matched_chr1_shared_backbone", "--mode", "development",
            "--recipe", RECIPE, "--seed", str(args.seed),
            "--query-source", query_source,
            "--development-split-seed", str(args.split_seed),
            "--early-stop-patience", str(args.early_stop_patience),
            "--canonical-root", p["canonical"], "--prepared-root", p["prepared"],
            "--feature-cache", p["features"], "--rna-cache", p["rna"],
            "--registry", p["registry"], "--cpg-targets-dir", p["targets"],
            "--output-root", p["output"], "--run-id", rid,
        ]
        if summary.is_file():
            print(f"[query-suite] {arm}: already complete; skipping", flush=True)
            continue
        if run_dir.is_dir():
            if not last.is_file():
                raise SystemExit(f"{arm}: incomplete run dir exists without checkpoints/last.pt: {run_dir}")
            cmd.append("--resume")
        if args.print_commands:
            print(shlex.join(cmd))
            continue
        log = log_dir / f"{rid}.log"
        print(f"[query-suite] {arm} ({query_source}) -> {log}", flush=True)
        with log.open("a") as handle:
            handle.write("\n=== " + shlex.join(cmd) + "\n")
            handle.flush()
            result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            print(f"[query-suite] {arm}: FAILED rc={result.returncode}; sequence stopped", flush=True)
            return result.returncode
        print(f"[query-suite] {arm}: complete", flush=True)

    if not args.print_commands:
        collect_cmd = [
            sys.executable, "scripts/experiments/collect_query_representation.py",
            "--output-root", p["output"], "--seed", str(args.seed), "--split-seed", str(args.split_seed),
        ]
        print("[query-suite] collecting -> results/reference/ablations/query_representation_2026_09", flush=True)
        return subprocess.run(collect_cmd, cwd=REPO_ROOT, env=env).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
