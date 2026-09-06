"""Shared loss building blocks for the RNA-methylation pipeline.

``masked_mean``, ``masked_locus_pearson``/``locus_correlation_loss`` (the
MAS-PCC term, opt-in via ``locus_pearson_weight``), and ``beta_nll_term`` are
reused by ``rna_training.locus_cls_trainer._direct_beta_loss``, the sole
RNA-methylation loss function. The earlier prior-anchored two-stage loss
(``residual_loss``) that used to live here has been retired alongside that
architecture generation.
"""
from __future__ import annotations

import torch

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


def beta_nll_term(
    outputs: dict[str, torch.Tensor],
    safe_target: torch.Tensor,
    mask: torch.Tensor,
    config: LossConfig,
) -> torch.Tensor:
    """Beta log-likelihood term, shared by every model family's loss function.

    Methylation beta values live in [0, 1] with variance that collapses towards
    both boundaries, which a plain MSE treats as homoscedastic. Parameterized by
    the model's own prediction (``outputs["beta"]``) as the mean, and a
    predicted concentration (``outputs["concentration"]``, absent unless the
    model built a concentration head): mu*phi and (1-mu)*phi are the two Beta
    shape parameters. A no-op (returns 0) whenever the weight is zero or no
    concentration was produced, so it is always safe to add unconditionally.
    """
    if config.beta_nll_weight == 0.0 or outputs.get("concentration") is None:
        return outputs["beta"].sum() * 0.0
    concentration = outputs["concentration"].float().clamp_min(config.concentration_min)
    mu = outputs["beta"].float().clamp(config.beta_nll_epsilon, 1.0 - config.beta_nll_epsilon)
    # WGBS beta values hit exactly 0 and 1, where the density diverges.
    observation = safe_target.float().clamp(config.beta_nll_epsilon, 1.0 - config.beta_nll_epsilon)
    alpha = mu * concentration
    beta_shape = (1.0 - mu) * concentration
    log_likelihood = (
        torch.lgamma(concentration)
        - torch.lgamma(alpha)
        - torch.lgamma(beta_shape)
        + (alpha - 1.0) * torch.log(observation)
        + (beta_shape - 1.0) * torch.log1p(-observation)
    )
    return masked_mean(-log_likelihood, mask)
