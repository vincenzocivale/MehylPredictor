from __future__ import annotations

import torch

from methylation_predictor.rna_branch.config import LossConfig
from methylation_predictor.rna_branch.losses import (
    locus_correlation_losses,
    masked_locus_pearson,
    pairwise_difference_loss,
    residual_loss,
    scheduled_loss_config,
)


def test_masked_locus_pearson_recovers_perfect_positive_and_negative() -> None:
    target = torch.tensor([
        [0.1, 0.8], [0.2, 0.7], [0.3, 0.6], [0.4, 0.5], [0.5, 0.4],
    ])
    prediction = torch.stack((2.0 * target[:, 0] + 3.0, -target[:, 1]), dim=1)
    mask = torch.ones_like(target, dtype=torch.bool)
    corr, valid = masked_locus_pearson(
        prediction, target, mask, min_observed_samples=3, epsilon=1e-8
    )
    assert valid.tolist() == [True, True]
    torch.testing.assert_close(corr, torch.tensor([1.0, -1.0]), atol=1e-6, rtol=0.0)


def test_locus_loss_excludes_missing_and_constant_loci() -> None:
    target = torch.tensor([
        [0.1, 0.5, float("nan")],
        [0.2, 0.5, 0.3],
        [0.3, 0.5, 0.4],
        [0.4, 0.5, 0.5],
    ])
    prediction = target.nan_to_num(0.0).clone()
    mask = torch.isfinite(target)
    config = LossConfig(locus_min_observed_samples=4, locus_lower_tail_fraction=0.6)
    mean_loss, tail_loss, valid_loci = locus_correlation_losses(
        prediction, target.nan_to_num(0.0), mask, config
    )
    assert valid_loci == 1
    torch.testing.assert_close(mean_loss, torch.tensor(0.0), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(tail_loss, torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_pairwise_difference_loss_prefers_correct_patient_order_and_amplitude() -> None:
    target = torch.tensor([[0.1, 0.8], [0.3, 0.6], [0.7, 0.2], [0.9, 0.1]])
    mask = torch.ones_like(target, dtype=torch.bool)
    correct, _ = pairwise_difference_loss(
        target, target, mask, max_pairs=16, huber_delta=0.05
    )
    reversed_loss, _ = pairwise_difference_loss(
        target.flip(0), target, mask, max_pairs=16, huber_delta=0.05
    )
    assert correct.item() == 0.0
    assert reversed_loss.item() > 0.04


def test_new_terms_are_opt_in_and_backpropagate() -> None:
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
        locus_pearson_weight=1.0,
        locus_lower_tail_weight=0.5,
        pairwise_difference_weight=0.5,
        locus_min_observed_samples=4,
    )
    loss, pieces = residual_loss({"beta": prediction, "delta_logit": delta}, target, prior, config)
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert pieces["valid_correlation_loci"] == 2


def test_mas_pcc_metric_alias_matches_locus_dynamic_median() -> None:
    import numpy as np
    from methylation_predictor.rna_branch.metrics import evaluate_predictions

    target = np.array([
        [0.1, 0.8], [0.2, 0.6], [0.4, 0.4], [0.8, 0.2],
    ])
    prediction = np.array([
        [0.2, 0.7], [0.3, 0.5], [0.5, 0.3], [0.9, 0.1],
    ])
    metrics = evaluate_predictions(
        target, prediction, prior=np.array([0.5, 0.5]),
        cancer_types=np.array(["A", "A", "B", "B"]),
    )
    assert metrics["mas_pcc"] == metrics["locus_dynamic_pearson_median"]
    assert metrics["mas_pcc"] > 0.99


def test_objective_schedule_uses_mse_warmup_then_configured_weights() -> None:
    config = LossConfig(
        beta_mse_weight=0.0,
        beta_huber_weight=0.1,
        residual_huber_weight=0.0,
        locus_pearson_weight=1.0,
        pairwise_difference_weight=0.5,
        objective_warmup_epochs=2,
    )
    warm = scheduled_loss_config(config, 2)
    active = scheduled_loss_config(config, 3)
    assert warm.beta_mse_weight == 1.0
    assert warm.locus_pearson_weight == 0.0
    assert warm.pairwise_difference_weight == 0.0
    assert active is config
