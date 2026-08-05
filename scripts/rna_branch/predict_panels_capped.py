#!/usr/bin/env python3
"""One-off: evaluate the 4 rna_branch panels from a checkpoint, capping the
CpG count per panel so each predict_panel() call finishes within this
harness's observed ~5-6 minute continuous-process limit. Genome-wide
in_distribution/sample_ood panels touch the full ~327k-CpG train_cpg split,
which is too large to evaluate in one process within that window (confirmed
empirically: uncapped attempts were killed via SIGTERM without completing).
locus_ood/double_ood use the much smaller ~41k-CpG test split and don't need
capping, but the same cap is applied uniformly for consistent methodology
across panels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from methylation_predictor.rna_branch.config import load_config
from methylation_predictor.rna_branch.trainer import ExperimentRunner


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-cpgs", type=int, default=30_000)
    args = p.parse_args()

    config = load_config(args.config)
    runner = ExperimentRunner(config)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=runner.device, weights_only=False)
        runner.model.load_state_dict(checkpoint["model_state"])
        runner._refresh_train_centroids()

        panels = {
            "in_distribution": ("train", "train"),
            "sample_ood": ("test", "train"),
            "locus_ood": ("train", "test"),
            "double_ood": ("test", "test"),
        }
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {}
        for name, (sample_split, cpg_split) in panels.items():
            print(f"[panel] {name}: {sample_split} x {cpg_split}, max_cpgs={args.max_cpgs}", flush=True)
            panel = runner.predict_panel(
                sample_split, cpg_split, max_cpgs=args.max_cpgs, keep_predictions=True, seed_offset=9001,
            )
            np.savez_compressed(
                output_dir / f"panel_{name}.npz",
                target=panel.target, prediction=panel.prediction,
                prior=runner.bundle.loci.prior[panel.cpg_indices],
                sample_idx=runner.bundle.samples.ids[panel.sample_indices].astype(str),
                cpg_idx=runner.bundle.loci.ids[panel.cpg_indices].astype(str),
                cancer_type=runner.bundle.samples.cancer_types[panel.sample_indices].astype(str),
            )
            summary[name] = panel.metrics
            print(f"[panel] {name}: mse={panel.metrics.get('mse')} skill_vs_prior={panel.metrics.get('skill_vs_prior')}", flush=True)
        (output_dir / "panels_capped_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
        print("ALL PANELS COMPLETE", flush=True)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
