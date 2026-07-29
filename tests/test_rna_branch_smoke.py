from __future__ import annotations

from pathlib import Path

from methylation_predictor.rna_branch.config import load_config
from methylation_predictor.rna_branch.low_rank import run_low_rank
from methylation_predictor.rna_branch.trainer import ExperimentRunner


def test_training_smoke(synthetic_project):
    config = load_config(synthetic_project)
    runner = ExperimentRunner(config)
    try:
        result = runner.train()
    finally:
        runner.close()
    assert result["best_epoch"] >= 1
    assert Path(config.output_dir, "metrics.json").is_file()
    assert "double_ood" in result["panels"]


def test_low_rank_smoke(synthetic_project, tmp_path):
    config = load_config(synthetic_project)
    result = run_low_rank(config, tmp_path / "svd.json", [1, 2, 3])
    assert result["samples"] == 15
    assert result["cpgs"] == 7
    assert "3" in result["explained_variance"]
