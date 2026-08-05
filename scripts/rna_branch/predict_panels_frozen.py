#!/usr/bin/env python3
"""Evaluate the 4 rna_branch panels from a checkpoint on an EXACT, previously
frozen CpG subset (read from another model's panel_*.npz `cpg_idx` field),
instead of drawing a fresh random subsample. Use this for any model that must
be compared apples-to-apples against an already-frozen reference run (e.g.
concat vs. the frozen bilinear panels) -- see FREEZE_MANIFEST.json next to the
reference panels for provenance of the frozen subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from methylation_predictor.rna_branch.config import load_config
from methylation_predictor.rna_branch.trainer import ExperimentRunner

PANELS = {
    "in_distribution": ("train", "train"),
    "sample_ood": ("test", "train"),
    "locus_ood": ("train", "test"),
    "double_ood": ("test", "test"),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--reference-panels-dir", required=True, help="dir with panel_<name>.npz to freeze CpG subset from")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    config = load_config(args.config)
    runner = ExperimentRunner(config)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=runner.device, weights_only=False)
        runner.model.load_state_dict(checkpoint["model_state"])
        runner._refresh_train_centroids()

        id_to_pos = {str(v): i for i, v in enumerate(runner.bundle.loci.ids)}

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {}
        for name, (sample_split, cpg_split) in PANELS.items():
            ref = np.load(Path(args.reference_panels_dir) / f"panel_{name}.npz", allow_pickle=True)
            frozen_ids = ref["cpg_idx"].astype(str)
            missing = [i for i in frozen_ids if i not in id_to_pos]
            if missing:
                raise ValueError(f"{name}: {len(missing)} frozen cpg ids not found in this bundle, e.g. {missing[:5]}")
            override = np.array([id_to_pos[i] for i in frozen_ids], dtype=np.int64)
            print(f"[panel] {name}: {sample_split} x {cpg_split}, frozen cpgs={len(override)}", flush=True)
            panel = runner.predict_panel(
                sample_split, cpg_split, cpg_indices_override=override, keep_predictions=True,
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
        (output_dir / "panels_frozen_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
        (output_dir / "FROZEN_SUBSET_SOURCE.json").write_text(json.dumps({
            "reference_panels_dir": str(Path(args.reference_panels_dir).resolve()),
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch"),
        }, indent=2) + "\n")
        print("ALL PANELS COMPLETE", flush=True)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
