#!/usr/bin/env python3
"""Collect final NTv3 comparisons and run leakage-controlled downstream tests."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="artifacts/genomic_encoder/ntv3_prior")
    p.add_argument("--mean-rna-prior", required=True)
    p.add_argument("--reference-stats", required=True); p.add_argument("--evaluation-stats", required=True)
    p.add_argument("--fold-manifest", required=True)
    a = p.parse_args(); root = Path(a.root)
    entries = []
    for directory in sorted(root.glob("final_NTv3_*/*")):
        metrics_path = directory / "metrics.json"
        if not metrics_path.exists(): continue
        metrics = json.loads(metrics_path.read_text())
        if "test" not in metrics: raise ValueError(f"final output lacks test metrics: {directory}")
        val = metrics["validation"]["mlp_ensemble"]
        test = metrics["test"]["mlp_ensemble"]
        entries.append({"directory": str(directory), "checkpoint": directory.parent.name.removeprefix("final_"),
                        "configuration": directory.name, "validation": val, "test": test})
        subprocess.run([sys.executable, "-m", "methylation_predictor.genomic_encoder.downstream_evaluation", "--locus-predictions", str(directory / "locus_predictions.parquet"),
                        "--reference-stats", a.reference_stats, "--evaluation-stats", a.evaluation_stats,
                        "--mean-rna-prior", a.mean_rna_prior, "--output", str(directory / "downstream_metrics.json")], check=True)
        if directory.parent.name.endswith("_post"):
            subprocess.run([sys.executable, "-m", "methylation_predictor.genomic_encoder.evaluate_ntv3_posttraining_exposure", "--locus-predictions", str(directory / "locus_predictions.parquet"),
                            "--fold-manifest", a.fold_manifest, "--output", str(directory / "posttraining_exposure_metrics.json")], check=True)
    if len(entries) != 8: raise ValueError(f"expected 8 final checkpoint/configuration comparisons, got {len(entries)}")
    entries.sort(key=lambda x: x["validation"]["mse"])
    (root / "final_comparison_leaderboard.json").write_text(json.dumps({"selection": "minimum validation beta-MSE of MLP ensemble", "entries": entries}, indent=2) + "\n")


if __name__ == "__main__": main()
