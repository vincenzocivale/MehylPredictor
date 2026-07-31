from __future__ import annotations

from pathlib import Path

import torch
import yaml

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


def test_pretrained_f2_warm_start_frozen_and_residual_learns(synthetic_pretrained_project):
    """End-to-end (via ExperimentRunner, real checkpoint file) check that the nested
    F2+adapter architecture is genuinely frozen: warm-started predictions match F2
    exactly, the residual branch receives a real gradient, and after one optimizer
    step F2's parameters are bit-identical while the residual branch's have moved.
    The prediction-exact check alone is not sufficient -- it would also pass if the
    residual branch received no usable gradient at all.
    """
    f2_config_path, nested_config_path = synthetic_pretrained_project

    f2_runner = ExperimentRunner(load_config(f2_config_path))
    try:
        f2_runner.train()
    finally:
        f2_runner.close()

    nested_runner = ExperimentRunner(load_config(nested_config_path))
    try:
        warm_start_info = nested_runner._apply_warm_start(nested_runner.config.training.warm_start_checkpoint)
        from methylation_predictor.rna_branch.trainer import freeze_parameters

        frozen_names = freeze_parameters(nested_runner.model, warm_start_info["loaded"])
        assert frozen_names, "expected at least one parameter to be frozen from the F2 checkpoint"

        f2_only = ExperimentRunner(load_config(f2_config_path))
        try:
            f2_state = torch.load(
                nested_runner.config.training.warm_start_checkpoint, map_location="cpu", weights_only=False
            )["model_state"]
            f2_only.model.load_state_dict(f2_state)
            sample_indices = nested_runner.bundle.sample_indices("validation")[:4]
            cpg_indices = nested_runner.bundle.cpg_indices("validation")[:4]
            rna_nested, loci, prior, variability, _ = nested_runner._tensor_inputs(sample_indices, cpg_indices)
            rna_f2 = torch.from_numpy(f2_only.bundle.rna(sample_indices))
            nested_runner.model.eval()
            f2_only.model.eval()
            with torch.no_grad():
                pred_nested = nested_runner.model(rna_nested, loci, prior, variability, nested_runner.reference_rna)["beta"]
                pred_f2 = f2_only.model(rna_f2, loci, prior, variability, f2_only.reference_rna)["beta"]
            assert torch.max(torch.abs(pred_nested - pred_f2)).item() < 1e-6
        finally:
            f2_only.close()

        baseline_params_before = {
            name: param.detach().clone()
            for name, param in nested_runner.model.named_parameters()
            if name in set(frozen_names)
        }
        residual_params_before = {
            name: param.detach().clone()
            for name, param in nested_runner.model.named_parameters()
            if name not in set(frozen_names)
        }

        nested_runner.model.train()
        train_samples = nested_runner.bundle.training_sample_pool("train")
        train_cpgs = nested_runner.bundle.training_cpg_pool("train")
        rna, loci, prior, variability, beta = nested_runner._tensor_inputs(
            train_samples[:8], train_cpgs[:6]
        )
        optimizer = torch.optim.AdamW(
            [p for p in nested_runner.model.parameters() if p.requires_grad], lr=0.05
        )
        optimizer.zero_grad(set_to_none=True)
        outputs = nested_runner.model(rna, loci, prior, variability, nested_runner.reference_rna)
        loss = torch.nan_to_num((outputs["beta"] - beta) ** 2).mean()
        loss.backward()

        residual_grad_norm = sum(
            param.grad.detach().norm().item() ** 2
            for name, param in nested_runner.model.named_parameters()
            if name not in set(frozen_names) and param.grad is not None
        ) ** 0.5
        assert residual_grad_norm > 0.0
        for name, param in nested_runner.model.named_parameters():
            if name in set(frozen_names):
                assert param.grad is None, f"frozen parameter {name} unexpectedly received a gradient"

        optimizer.step()

        for name, param in nested_runner.model.named_parameters():
            if name in baseline_params_before:
                torch.testing.assert_close(param.detach(), baseline_params_before[name])
        changed = any(
            not torch.allclose(param.detach(), residual_params_before[name])
            for name, param in nested_runner.model.named_parameters()
            if name in residual_params_before
        )
        assert changed, "expected at least one residual parameter to change after the optimizer step"
    finally:
        nested_runner.close()


def test_seed_initial_checkpoint_can_win(synthetic_pretrained_project):
    """If the residual adapter never improves validation (forced here with a huge
    residual learning rate that makes training diverge), the run must fall back to
    epoch 0 -- i.e. exactly F2's own prediction -- instead of returning whichever
    diverged epoch happened to be least bad."""
    f2_config_path, nested_config_path = synthetic_pretrained_project

    f2_runner = ExperimentRunner(load_config(f2_config_path))
    try:
        f2_runner.train()
    finally:
        f2_runner.close()

    raw_config = yaml.safe_load(Path(nested_config_path).read_text())
    raw_config["training"]["residual_learning_rate"] = 50.0
    raw_config["training"]["epochs"] = 2
    raw_config["training"]["min_epochs"] = 0
    raw_config["training"]["patience"] = 100
    diverging_path = Path(nested_config_path).with_name("config_nested_diverging.yaml")
    diverging_path.write_text(yaml.safe_dump(raw_config, sort_keys=False))

    runner = ExperimentRunner(load_config(diverging_path))
    try:
        result = runner.train()
    finally:
        runner.close()
    assert result["best_epoch"] == 0
