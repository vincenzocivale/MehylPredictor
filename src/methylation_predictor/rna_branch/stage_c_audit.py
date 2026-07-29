"""Validation-only checkpoint selection and Stage-C audit summaries.

This utility deliberately loads only checkpoints emitted by the trainer and
does not inspect test metrics until a validation criterion has selected one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .trainer import ExperimentRunner
from .utils import json_safe, write_json


CRITERIA = {"min_mse": ("mse", False), "max_dynamic_skill": ("dynamic_skill", True),
            "max_within_cancer_skill": ("within_cancer_skill", True)}


def audit_run(run_dir: Path) -> dict[str, object]:
    config = load_config(run_dir / "config.yaml")
    history = pd.read_csv(run_dir / "training_history.csv")
    rows = []
    for criterion, (metric, maximize) in CRITERIA.items():
        column = f"validation_{metric}"
        eligible = history.dropna(subset=[column])
        idx = eligible[column].idxmax() if maximize else eligible[column].idxmin()
        epoch = int(history.loc[idx, "epoch"])
        print(f"[audit] run={run_dir.name} criterion={criterion} epoch={epoch}", flush=True)
        checkpoint = run_dir / f"epoch_{epoch:03d}.pt"
        runner = ExperimentRunner(config)
        try:
            state = torch.load(checkpoint, map_location=runner.device, weights_only=False)
            runner.model.load_state_dict(state["model_state"])
            panels = {}
            for name, panel in config.evaluation.panels.items():
                result = runner.predict_panel(panel["sample_split"], panel["cpg_split"],
                                              config.evaluation.max_cpgs_per_panel,
                                              keep_predictions=(name == "double_ood"))
                panels[name] = result.metrics
                if name == "double_ood":
                    np.savez_compressed(run_dir / f"predictions_double_ood_{criterion}.npz",
                        target=result.target, prediction=result.prediction,
                        prior=runner.bundle.loci.prior[result.cpg_indices],
                        sample_idx=runner.bundle.samples.ids[result.sample_indices].astype(str),
                        cpg_idx=runner.bundle.loci.ids[result.cpg_indices].astype(str),
                        cancer_type=runner.bundle.samples.cancer_types[result.sample_indices].astype(str))
        finally:
            runner.close()
        rows.append({"criterion": criterion, "epoch": epoch,
                     "validation": {key.replace("validation_", ""): value for key, value in history.loc[idx].items() if key.startswith("validation_")},
                     "test_panels": panels})
        print(f"[audit] completed run={run_dir.name} criterion={criterion}", flush=True)
    result = {"run": run_dir.name, "selections": rows}
    write_json(run_dir / "phase0_checkpoint_audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-dir", required=True)
    parser.add_argument("--run-dir", default=None, help="audit one run only (for schedulers)")
    args = parser.parse_args()
    root = Path(args.screening_dir)
    paths = [Path(args.run_dir)] if args.run_dir else [
        path for path in sorted(root.iterdir()) if (path / "training_history.csv").is_file()
        and len(list(path.glob("epoch_*.pt"))) >= 20
    ]
    results = [audit_run(path) for path in paths]
    if not args.run_dir:
        write_json(root.parent / "phase0_checkpoint_audit.json", {"runs": results})
    print(json.dumps(json_safe({"completed_runs": len(results)})))


if __name__ == "__main__":
    main()
