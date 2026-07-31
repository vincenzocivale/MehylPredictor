#!/usr/bin/env python3
"""Run the R0, R1, R2 and R4 representation study sequentially on CUDA."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_F2 = {
    17: "artifacts/rna_branch/stage_f_fusion/first_tranche/f2_concat_product_seed17/best.pt",
    23: "artifacts/rna_branch/stage_f_fusion/confirm/f2_concat_product_seed23/best.pt",
    41: "artifacts/rna_branch/stage_f_fusion/confirm/f2_concat_product_seed41/best.pt",
}
GENE_EMBEDDINGS = "artifacts/rna_branch/stage_t_gene_tokens/inputs/ntv3_gene_embeddings.npz"


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})


def _require_cuda() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is unavailable; full representation runs were not started")
    print(f"CUDA ready: {torch.cuda.get_device_name(0)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hallmark-gmt", required=True, help="official MSigDB Hallmark human GMT")
    parser.add_argument("--base", default="configs/rna_branch/stage_f_base.yaml")
    parser.add_argument("--output-root", default="artifacts/rna_branch/representation_search")
    parser.add_argument("--seeds", default="17,23,41")
    parser.add_argument("--random-seed", type=int, default=20260730)
    parser.add_argument(
        "--rerun-complete",
        action="store_true",
        help="rerun a config even when its output directory already has metrics.json",
    )
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    missing = [seed for seed in seeds if seed not in DEFAULT_F2 or not (ROOT / DEFAULT_F2[seed]).is_file()]
    if missing:
        raise ValueError(f"missing production F2 checkpoint(s) for seed(s): {missing}")
    if not (ROOT / GENE_EMBEDDINGS).is_file():
        raise FileNotFoundError(ROOT / GENE_EMBEDDINGS)
    if not Path(args.hallmark_gmt).is_file():
        raise FileNotFoundError(args.hallmark_gmt)

    # This deliberately happens before creating any run outputs: CPU fallback is excluded.
    _require_cuda()
    root = Path(args.output_root)
    module_dir = root / "modules"
    _run([
        sys.executable, "scripts/rna_branch/prepare_hallmark_modules.py", "--config", args.base,
        "--hallmark-gmt", args.hallmark_gmt, "--output-dir", str(module_dir),
        "--random-seed", str(args.random_seed),
    ])
    for name in ("hallmark", "hallmark_random_matched"):
        _run([
            sys.executable, "scripts/rna_branch/build_gene_module_matrix.py", "--config", args.base,
            "--membership", str(module_dir / f"{name}_aligned.tsv") if name == "hallmark"
            else str(module_dir / "hallmark_random_matched.tsv"),
            "--output", str(module_dir / f"{name}.npz"),
        ])

    command = [
        sys.executable, "scripts/rna_branch/make_representation_configs.py", "--base", args.base,
        "--output-root", str(root), "--gene-embeddings", GENE_EMBEDDINGS, "--seeds", args.seeds,
        "--module-weights", f"hallmark={module_dir / 'hallmark.npz'}",
        "--module-weights", f"hallmark_random_matched={module_dir / 'hallmark_random_matched.npz'}",
    ]
    for seed in seeds:
        command.extend(["--f2-checkpoint", f"{seed}={DEFAULT_F2[seed]}"])
    _run(command)

    configs = root / "configs"
    family_prefixes = ("r0_", "r1_", "r2_", "r4_")
    for prefix in family_prefixes:
        for config in sorted(configs.glob(f"{prefix}*.yaml")):
            output_dir = Path(yaml.safe_load(config.read_text())["output_dir"])
            if not args.rerun_complete and (output_dir / "metrics.json").is_file():
                print(f"= skipping completed run {config.stem}", flush=True)
                continue
            _run([sys.executable, "-m", "methylation_predictor.rna_branch.cli", "validate", "--config", str(config)])
            _run([sys.executable, "-m", "methylation_predictor.rna_branch.cli", "train", "--config", str(config)])

    _run([
        sys.executable, "-m", "methylation_predictor.rna_branch.aggregate_report",
        "--screening-dir", str(root),
        "--baseline-dir", "artifacts/rna_branch/stage_f_fusion/first_tranche",
        "--baseline-dir", "artifacts/rna_branch/stage_f_fusion/confirm",
        "--baseline-family", "f2_concat_product",
        "--output", str(root / "representation_summary.csv"),
    ])


if __name__ == "__main__":
    main()
