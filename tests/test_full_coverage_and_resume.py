from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from methylation_predictor.rna_branch.config import load_config
from methylation_predictor.rna_branch.trainer import ExperimentRunner


def _set_full_coverage(config_path: Path, epochs: int, cpg_batch_size: int = 3, sample_batch_size: int = 4) -> Path:
    raw = yaml.safe_load(Path(config_path).read_text())
    raw["training"]["cpg_sampling"] = "full_coverage"
    raw["training"]["cpg_batch_size"] = cpg_batch_size
    raw["training"]["sample_batch_size"] = sample_batch_size
    raw["training"]["epochs"] = epochs
    raw["training"]["min_epochs"] = epochs
    raw["training"]["patience"] = epochs
    raw["training"]["validation_every"] = 1
    new_path = Path(config_path).with_name("config_full_coverage.yaml")
    new_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return new_path


def test_full_coverage_epoch_reaches_100_percent(synthetic_project):
    config_path = _set_full_coverage(synthetic_project, epochs=2)
    config = load_config(config_path)
    runner = ExperimentRunner(config)
    try:
        runner.train()
    finally:
        runner.close()

    history = pd.read_csv(Path(config.output_dir) / "training_history.csv")
    assert (history["coverage_cpg_coverage_fraction"] == 1.0).all()
    assert (history["coverage_sample_coverage_fraction"] == 1.0).all()
    assert len(history) == 2


def test_resume_continues_from_last_completed_epoch_without_duplicating_history(synthetic_project):
    config_path = _set_full_coverage(synthetic_project, epochs=1)
    config = load_config(config_path)
    runner = ExperimentRunner(config)
    try:
        runner.train()
    finally:
        runner.close()

    output_dir = Path(config.output_dir)
    assert (output_dir / "checkpoint_latest.pt").is_file()
    history_after_first = pd.read_csv(output_dir / "training_history.csv")
    assert len(history_after_first) == 1

    # Bump epochs to 3 in the SAME output_dir and re-run -- must resume from
    # epoch 2, not restart from epoch 1 or duplicate epoch 1's history row.
    raw = yaml.safe_load(Path(config_path).read_text())
    raw["training"]["epochs"] = 3
    raw["training"]["min_epochs"] = 3
    raw["training"]["patience"] = 3
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    config2 = load_config(config_path)
    runner2 = ExperimentRunner(config2)
    try:
        result = runner2.train()
    finally:
        runner2.close()

    history_after_resume = pd.read_csv(output_dir / "training_history.csv")
    assert list(history_after_resume["epoch"]) == [1, 2, 3]
    assert result["best_epoch"] >= 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_resume_on_cuda_device_restores_rng_state(synthetic_project):
    """Regression test: torch.load(..., map_location=self.device) moves EVERY
    tensor in the checkpoint onto that device, including the CPU-only RNG
    state ByteTensor -- torch.set_rng_state then fails with 'RNG state must
    be a torch.ByteTensor' unless it's forced back to CPU before restoring.
    Reproduced for real during this task's own preflight smoke run on the
    actual GPU box."""
    config_path = _set_full_coverage(synthetic_project, epochs=1)
    raw = yaml.safe_load(Path(config_path).read_text())
    raw["training"]["device"] = "cuda"
    raw["training"]["amp"] = False
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    config = load_config(config_path)
    runner = ExperimentRunner(config)
    try:
        runner.train()
    finally:
        runner.close()

    raw["training"]["epochs"] = 2
    raw["training"]["min_epochs"] = 2
    raw["training"]["patience"] = 2
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    config2 = load_config(config_path)
    runner2 = ExperimentRunner(config2)
    try:
        result = runner2.train()  # must not raise TypeError: RNG state must be a torch.ByteTensor
    finally:
        runner2.close()
    assert result["best_epoch"] >= 1


def test_checkpoint_selection_final_always_overwrites_best(synthetic_project):
    raw = yaml.safe_load(Path(synthetic_project).read_text())
    raw["training"]["checkpoint_selection"] = "final"
    raw["training"]["epochs"] = 2
    raw["training"]["min_epochs"] = 2
    raw["training"]["patience"] = 2
    raw["training"]["validation_every"] = 1000000  # only fires on the final epoch (epoch == epochs)
    config_path = Path(synthetic_project).with_name("config_final.yaml")
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    config = load_config(config_path)
    runner = ExperimentRunner(config)
    try:
        result = runner.train()
    finally:
        runner.close()
    assert result["best_epoch"] == 2
