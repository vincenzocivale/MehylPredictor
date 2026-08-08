#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the official held-out test panel
(sample_split="test" x cpg_split="test") -- the double-OOD panel, held out
on both axes from every train/dev stage.

Deliberately outside `scripts/train.sh`/`configs/train.yaml` (whose
`evaluation.panels` is intentionally empty -- see the comment near the top of
`configs/train.yaml`): this is the separate, explicit evaluation step run
once training is finished and a final checkpoint is selected.

Reuses `ExperimentRunner.predict_panel` (the same forward-pass/metrics path
`train()` would use for any configured panel) rather than re-implementing
inference; the only new logic here is loading a config + a standalone
checkpoint into a fresh runner and evaluating them, run outside of `train()`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from methylation_predictor.config import load_config
from methylation_predictor.trainer import ExperimentRunner
from methylation_predictor.utils import json_safe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="training config whose data.* paths are used")
    parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint .pt to evaluate (must match config's architecture)")
    parser.add_argument("--sample-split", default="test")
    parser.add_argument("--cpg-split", default="test")
    parser.add_argument("--output", type=Path, required=True, help="where to write the metrics JSON")
    parser.add_argument("--save-predictions", type=Path, default=None, help="optional .npz path for per-cell target/prediction")
    args = parser.parse_args()

    config = load_config(args.config)
    runner = ExperimentRunner(config)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=runner.device, weights_only=False)
        missing, unexpected = runner.model.load_state_dict(checkpoint["model_state"], strict=True)
        assert not missing and not unexpected  # same architecture as training -> exact match expected

        result = runner.predict_panel(
            args.sample_split,
            args.cpg_split,
            keep_predictions=args.save_predictions is not None,
        )
        metrics = {
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "sample_split": args.sample_split,
            "cpg_split": args.cpg_split,
            "n_samples": int(len(result.sample_indices)),
            "n_cpgs": int(len(result.cpg_indices)),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "metrics": result.metrics,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(json_safe(metrics), indent=2, sort_keys=True) + "\n")
        print(json.dumps(json_safe(metrics), indent=2, sort_keys=True))

        if args.save_predictions is not None:
            import numpy as np
            args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.save_predictions,
                target=result.target,
                prediction=result.prediction,
                prior=runner.bundle.loci.prior[result.cpg_indices],
                sample_idx=runner.bundle.samples.ids[result.sample_indices].astype(str),
                cpg_idx=runner.bundle.loci.ids[result.cpg_indices].astype(str),
                cancer_type=runner.bundle.samples.cancer_types[result.sample_indices].astype(str),
            )
    finally:
        runner.close()


if __name__ == "__main__":
    main()
