"""Losses for bounded residual methylation prediction.

The locus-Pearson (MAS-PCC) term operates on Cartesian ``samples x CpGs``
minibatches. It is strictly opt-in: with locus_pearson_weight=0 the objective
reduces to the flat pointwise residual loss.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .config import LossConfig


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask]
    if selected.numel() == 0:
        return values.sum() * 0.0
    return selected.mean()


def _validate_locus_batch_shape(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
    if prediction.ndim != 2 or target.shape != prediction.shape or mask.shape != prediction.shape:
        raise ValueError("prediction, target, and mask must share [samples, cpgs] shape")


def _locus_centred_stats(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    min_observed_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-locus (per-CpG) centring across the sample axis.

    Returns ``(pred_centred, truth_centred, counts, denominator, pred_ss,
    truth_ss, has_min_samples)``, all computed in float32 with missing
    observations excluded before centring so NaNs cannot enter the backward
    graph.
    """
    pred = prediction.float()
    truth = target.float()
    weights = mask.to(dtype=pred.dtype)
    counts = weights.sum(dim=0)
    denominator = counts.clamp_min(1.0)

    pred_safe = torch.where(mask, pred, torch.zeros_like(pred))
    truth_safe = torch.where(mask, truth, torch.zeros_like(truth))
    pred_mean = pred_safe.sum(dim=0) / denominator
    truth_mean = truth_safe.sum(dim=0) / denominator
    pred_centred = torch.where(mask, pred - pred_mean.unsqueeze(0), torch.zeros_like(pred))
    truth_centred = torch.where(mask, truth - truth_mean.unsqueeze(0), torch.zeros_like(truth))

    pred_ss = pred_centred.square().sum(dim=0)
    truth_ss = truth_centred.square().sum(dim=0)
    has_min_samples = counts >= min_observed_samples
    return pred_centred, truth_centred, counts, denominator, pred_ss, truth_ss, has_min_samples


def masked_locus_pearson(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    min_observed_samples: int,
    epsilon: float,
    min_target_std: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pearson correlation across samples, independently for every CpG.

    Returns ``(correlations, valid_locus_mask)``.  The calculation is promoted to
    float32 under AMP and excludes missing observations before centring, so NaNs
    cannot enter the backward graph.
    """
    _validate_locus_batch_shape(prediction, target, mask)
    if min_observed_samples < 2:
        raise ValueError("min_observed_samples must be >= 2")

    pred_centred, truth_centred, counts, denominator, pred_ss, truth_ss, has_min_samples = _locus_centred_stats(
        prediction, target, mask, min_observed_samples
    )
    covariance = (pred_centred * truth_centred).sum(dim=0)
    target_std = torch.sqrt(truth_ss / denominator)
    valid = (has_min_samples & (pred_ss > epsilon) & (truth_ss > epsilon)
             & (target_std >= min_target_std))
    scale = torch.sqrt((pred_ss * truth_ss).clamp_min(epsilon))
    correlations = covariance / scale
    correlations = correlations.clamp(-1.0, 1.0)
    return correlations, valid


def locus_correlation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    config: LossConfig,
) -> tuple[torch.Tensor, int]:
    correlations, valid = masked_locus_pearson(
        prediction,
        target,
        mask,
        min_observed_samples=config.locus_min_observed_samples,
        epsilon=config.locus_pearson_epsilon,
        min_target_std=config.locus_pearson_min_target_std,
    )
    valid_values = correlations[valid]
    if valid_values.numel() == 0:
        zero = prediction.sum() * 0.0
        return zero, 0
    return 1.0 - valid_values.mean(), int(valid_values.numel())


def residual_loss(
    outputs: dict[str, torch.Tensor],
    target_beta: torch.Tensor,
    prior: torch.Tensor,
    config: LossConfig,
    epsilon: float = 1e-4,
    sigma: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    mask = torch.isfinite(target_beta)
    prediction = outputs["beta"]
    # Replace missing targets before any differentiable operation. Masking a tensor
    # only after arithmetic with NaNs can still create NaN gradients (0 * NaN).
    safe_target = torch.where(mask, target_beta, prior.unsqueeze(0))
    beta_mse = masked_mean((prediction - safe_target) ** 2, mask)

    clipped_target = safe_target.clamp(epsilon, 1.0 - epsilon)
    clipped_prior = prior.clamp(epsilon, 1.0 - epsilon)
    target_delta = torch.logit(clipped_target) - torch.logit(clipped_prior).unsqueeze(0)
    huber = masked_mean(
        F.huber_loss(outputs["delta_logit"], target_delta, reduction="none", delta=config.residual_huber_delta),
        mask,
    )
    shrinkage = masked_mean(outputs["delta_logit"] ** 2, mask)

    # V1: standardized-residual counterparts, operating on the model's raw
    # (pre-sigma-scale) output against the sigma-normalized target. Only
    # active when a variance-normalized model supplied both sigma and
    # outputs["raw_delta"], and at least one weight is nonzero.
    if (
        sigma is not None
        and "raw_delta" in outputs
        and (config.standardized_residual_huber_weight != 0.0 or config.standardized_shrinkage_weight != 0.0)
    ):
        sigma_floor = sigma.clamp_min(config.sigma_min)
        standardized_target = target_delta / sigma_floor.unsqueeze(0)
        standardized_huber = masked_mean(
            F.huber_loss(
                outputs["raw_delta"], standardized_target,
                reduction="none", delta=config.standardized_residual_huber_delta,
            ),
            mask,
        )
        standardized_shrinkage = masked_mean(outputs["raw_delta"] ** 2, mask)
    else:
        standardized_huber = prediction.sum() * 0.0
        standardized_shrinkage = prediction.sum() * 0.0

    if config.locus_pearson_weight != 0.0:
        pearson_loss, valid_loci = locus_correlation_loss(prediction, safe_target, mask, config)
    else:
        pearson_loss = prediction.sum() * 0.0
        valid_loci = 0

    total = (
        config.beta_mse_weight * beta_mse
        + config.residual_huber_weight * huber
        + config.shrinkage_weight * shrinkage
        + config.standardized_residual_huber_weight * standardized_huber
        + config.standardized_shrinkage_weight * standardized_shrinkage
        + config.locus_pearson_weight * pearson_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "beta_mse": float(beta_mse.detach().cpu()),
        "residual_huber": float(huber.detach().cpu()),
        "shrinkage": float(shrinkage.detach().cpu()),
        "standardized_residual_huber": float(standardized_huber.detach().cpu()),
        "standardized_shrinkage": float(standardized_shrinkage.detach().cpu()),
        "locus_pearson_loss": float(pearson_loss.detach().cpu()),
        "valid_correlation_loci": valid_loci,
        "observed": int(mask.sum().detach().cpu()),
    }
