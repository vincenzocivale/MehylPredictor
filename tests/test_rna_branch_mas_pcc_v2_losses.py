"""Stage MAS-PCC-v2: centered-MSE / amplitude-loss (anti-collapse terms) and
gradual ramp scheduling, added after the P1-P5 screening showed
locus_pearson_weight alone is gameable by amplitude collapse (see
docs/rna_branch_mas_pcc_experiments.md and the Stage MAS-PCC memory)."""
from __future__ import annotations

import torch

from methylation_predictor.rna_branch.config import LossConfig
from methylation_predictor.rna_branch.losses import (
    locus_amplitude_loss,
    locus_centered_mse,
    masked_locus_pearson,
    ramp_loss_config,
    residual_loss,
)


def test_locus_centered_mse_zero_for_exact_match() -> None:
    target = torch.tensor([[0.1, 0.8], [0.3, 0.6], [0.5, 0.4], [0.7, 0.2]])
    mask = torch.ones_like(target, dtype=torch.bool)
    loss, valid_loci = locus_centered_mse(target, target, mask, min_observed_samples=4)
    assert valid_loci == 2
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_locus_centered_mse_catches_amplitude_collapse_that_pearson_misses() -> None:
    """The P1-P4 failure mode: a prediction shrunk toward the locus mean stays
    perfectly correlated with the truth (pearson loss ~0) but is a bad point
    estimate. centered_mse must penalize this even though pearson does not."""
    target = torch.tensor([[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3], [0.9, 0.1]])
    mean = target.mean(dim=0, keepdim=True)
    collapsed = mean + 0.02 * (target - mean)  # same direction, ~2% of the true amplitude
    mask = torch.ones_like(target, dtype=torch.bool)

    corr, valid = masked_locus_pearson(collapsed, target, mask, min_observed_samples=5, epsilon=1e-8)
    assert valid.all()
    torch.testing.assert_close(corr, torch.tensor([1.0, 1.0]), atol=1e-4, rtol=0.0)

    mse_loss, valid_loci = locus_centered_mse(collapsed, target, mask, min_observed_samples=5)
    assert valid_loci == 2
    # True centred variance is ~0.062 per column; a 98%-collapsed prediction should
    # recover nearly all of it as squared error.
    assert mse_loss.item() > 0.05


def test_locus_amplitude_loss_zero_for_matched_amplitude_positive_for_collapse() -> None:
    target = torch.tensor([[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3], [0.9, 0.1]])
    mask = torch.ones_like(target, dtype=torch.bool)

    matched_loss, matched_valid = locus_amplitude_loss(
        target, target, mask, min_observed_samples=5, epsilon=1e-6
    )
    assert matched_valid == 2
    torch.testing.assert_close(matched_loss, torch.tensor(0.0), atol=1e-4, rtol=0.0)

    mean = target.mean(dim=0, keepdim=True)
    collapsed = mean + 0.02 * (target - mean)
    collapsed_loss, collapsed_valid = locus_amplitude_loss(
        collapsed, target, mask, min_observed_samples=5, epsilon=1e-6
    )
    # Unlike masked_locus_pearson's valid mask, amplitude loss must NOT exclude the
    # near-zero-variance (collapsed) prediction -- that is exactly the case it exists
    # to penalize, not skip.
    assert collapsed_valid == 2
    assert collapsed_loss.item() > 0.9  # ratio ~0.02 -> (0.02-1)^2 ~ 0.96


def test_ramp_loss_config_interpolates_linearly_between_warmup_and_target() -> None:
    config = LossConfig(
        beta_mse_weight=1.0,
        locus_pearson_weight=0.8,
        warmup_locus_pearson_weight=0.0,
        ramp_start_epoch=3,
        ramp_epochs=4,
    )
    before = ramp_loss_config(config, 1)
    assert before.locus_pearson_weight == 0.0
    at_start = ramp_loss_config(config, 3)
    assert at_start.locus_pearson_weight == 0.0
    midpoint = ramp_loss_config(config, 5)
    torch.testing.assert_close(torch.tensor(midpoint.locus_pearson_weight), torch.tensor(0.4), atol=1e-6, rtol=0.0)
    at_end = ramp_loss_config(config, 7)
    assert at_end.locus_pearson_weight == 0.8
    past_end = ramp_loss_config(config, 20)
    assert past_end.locus_pearson_weight == 0.8
    # beta_mse_weight has identical warmup/target defaults (1.0), so it is a no-op
    # throughout the ramp even though other weights are moving.
    assert midpoint.beta_mse_weight == 1.0


def test_ramp_loss_config_disabled_by_default_is_noop() -> None:
    config = LossConfig(locus_pearson_weight=1.0)
    assert config.ramp_epochs == 0
    result = ramp_loss_config(config, 5)
    assert result is config


def test_centered_mse_and_amplitude_terms_are_opt_in_and_backpropagate() -> None:
    prediction = torch.tensor(
        [[0.2, 0.7], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2]], requires_grad=True
    )
    target = torch.tensor([[0.1, 0.8], [0.3, 0.7], [0.7, 0.3], [0.9, 0.1]])
    prior = torch.tensor([0.5, 0.5])
    delta = torch.logit(prediction.clamp(1e-4, 1 - 1e-4)) - torch.logit(prior).unsqueeze(0)
    config = LossConfig(
        beta_mse_weight=0.0,
        residual_huber_weight=0.0,
        shrinkage_weight=0.0,
        centered_mse_weight=0.5,
        amplitude_weight=0.1,
        locus_min_observed_samples=4,
    )
    loss, pieces = residual_loss({"beta": prediction, "delta_logit": delta}, target, prior, config)
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert pieces["valid_centered_loci"] == 2
    assert pieces["valid_amplitude_loci"] == 2


def test_default_config_leaves_new_terms_at_zero() -> None:
    """With all new weights at their defaults, existing configs are numerically unchanged."""
    prediction = torch.tensor([[0.2, 0.7], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2]])
    target = torch.tensor([[0.1, 0.8], [0.3, 0.7], [0.7, 0.3], [0.9, 0.1]])
    prior = torch.tensor([0.5, 0.5])
    delta = torch.logit(prediction.clamp(1e-4, 1 - 1e-4)) - torch.logit(prior).unsqueeze(0)
    config = LossConfig()
    _, pieces = residual_loss({"beta": prediction, "delta_logit": delta}, target, prior, config)
    assert pieces["centered_mse_loss"] == 0.0
    assert pieces["amplitude_loss"] == 0.0
    assert pieces["valid_centered_loci"] == 0
    assert pieces["valid_amplitude_loci"] == 0
