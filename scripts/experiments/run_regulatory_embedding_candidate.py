#!/usr/bin/env python3
"""Run one development candidate, optionally after a specific live training job."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def process_identity(pid):
    try:
        # Process names may contain spaces; field 22 follows the closing ')'.
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["f7", "f8"], required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--data-root", type=Path, default=Path("/dune/DATASETS/MethylPredictionData"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/regulatory_embedding")
    parser.add_argument("--after-pid", type=int)
    parser.add_argument("--after-run", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.after_pid is None) != (args.after_run is None):
        parser.error("--after-pid and --after-run must be provided together")
    if args.after_pid is not None:
        identity = process_identity(args.after_pid)
        if identity is not None:
            command = (Path("/proc") / str(args.after_pid) / "cmdline").read_bytes().split(b"\0")
            if args.after_run.name.encode() not in command:
                raise ValueError("the specified PID does not belong to the expected run")
            print(f"Waiting for PID {args.after_pid}, starttime={identity}", flush=True)
            while process_identity(args.after_pid) == identity:
                time.sleep(15)
        summary_path = args.after_run / "training/summary.json"
        if not summary_path.is_file():
            raise RuntimeError("preceding process ended without a completion summary; not launching another run")
        summary = json.loads(summary_path.read_text())
        print(f"Preceding run completed: {summary}", flush=True)
    compute_pids = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True,
    ).strip()
    if compute_pids:
        raise RuntimeError(f"GPU has other compute processes ({compute_pids}); candidate not started")
    data = args.data_root
    prepared = data / "derived/methylprophet_table5_tcga_chr1"
    canonical = data / "datasets/methylprophet_repro_v1"
    run_id = f"regulatory-{args.variant}-dev-seed{args.seed}-e30"
    run_dir = args.output_root / "runs/locus_cls_joint/chr1" / run_id
    if run_dir.exists() and not args.resume:
        raise FileExistsError(run_dir)
    command = [
        sys.executable, str(ROOT / "scripts/train.py"), "--model", "rna_methylation",
        "--scope", "chr1", "--engine", "matched_chr1_shared_backbone", "--mode", "development",
        "--recipe", str(ROOT / f"configs/models/functional_fusion/{args.variant}.yaml"),
        "--output-root", str(args.output_root), "--run-id", run_id, "--epochs", "30",
        "--seed", str(args.seed), "--development-split-seed", "17", "--early-stop-patience", "8",
        "--canonical-root", str(canonical), "--prepared-root", str(prepared),
        "--feature-cache", str(prepared / "features"), "--rna-cache", str(prepared / "rna"),
        "--registry", str(canonical / "cpg/registries/array_cpg_map.parquet"),
        "--cpg-targets-dir", str(data / "derived/cpg_statistics/chr1"),
        "--functional-atlas", str(data / "derived/ntv3_functional_peak_atlas_chr1_all_sources"),
        "--annotation-cache", str(data / "derived/ntv3_probe_targets/chr1_annotation_features_all_sources"),
        "--functional-only",
    ]
    if args.resume:
        command.append("--resume")
    logs = ROOT / "logs/regulatory_embedding"
    logs.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"command": command, "run_dir": str(run_dir)}, indent=2), flush=True)
    with (logs / f"{run_id}.log").open("a") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print((run_dir / "training/summary.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
