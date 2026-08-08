"""Losses for bounded residual methylation prediction.

The MAS-PCC terms operate on Cartesian ``samples x CpGs`` minibatches.  They are
strictly opt-in: with all new weights at zero, the historical objective is
unchanged.
"""
from __future__ import annotations

import math
from dataclasses import fields, replace

import torch
from torch.nn import functional as F

from .config import LossConfig


_STRUCTURED_WEIGHT_FIELDS = (
    "locus_pearson_weight",
    "locus_lower_tail_weight",
    "pairwise_difference_weight",
    "global_prior_ratio_weight",
    "locus_skill_weight",
    "locus_ccc_weight",
    "within_cancer_dynamic_weight",
)


def scheduled_loss_config(config: LossConfig, epoch: int) -> LossConfig:
    """Return the objective active at a one-indexed training epoch.

    Warm-up uses the historical pointwise objective. After warm-up,
    ``objective_ramp_epochs > 0`` linearly ramps every structured term in
    ``_STRUCTURED_WEIGHT_FIELDS`` (plus beta_mse/beta_huber/residual_huber/
    shrinkage/beta_macro) from its warmup value to its configured value,
    instead of the historical instant switch. This is independent of the
    separate ``ramp_loss_config``, which callers select via
    ``config.loss.ramp_epochs`` instead.
    """
    if epoch < 1:
        raise ValueError("epoch must be one-indexed and >= 1")
    if config.objective_warmup_epochs < 0 or config.objective_ramp_epochs < 0:
        raise ValueError("objective warm-up/ramp epochs must be >= 0")
    if epoch <= config.objective_warmup_epochs:
        return replace(
            config,
            beta_mse_weight=config.warmup_beta_mse_weight,
            beta_huber_weight=config.warmup_beta_huber_weight,
            residual_huber_weight=config.warmup_residual_huber_weight,
            shrinkage_weight=config.warmup_shrinkage_weight,
            **{name: 0.0 for name in _STRUCTURED_WEIGHT_FIELDS},
        )
    if config.objective_ramp_epochs == 0:
        return config
    ramp_step = epoch - config.objective_warmup_epochs
    fraction = min(1.0, ramp_step / config.objective_ramp_epochs)
    values = {field.name: getattr(config, field.name) for field in fields(config)}
    for name in _STRUCTURED_WEIGHT_FIELDS:
        values[name] = float(values[name]) * fraction
    warmup_targets = {
        "beta_mse_weight": config.warmup_beta_mse_weight,
        "beta_huber_weight": config.warmup_beta_huber_weight,
        "residual_huber_weight": config.warmup_residual_huber_weight,
        "shrinkage_weight": config.warmup_shrinkage_weight,
    }
    for name, warmup_value in warmup_targets.items():
        target_value = float(getattr(config, name))
        values[name] = float(warmup_value) + fraction * (target_value - float(warmup_value))
    values["beta_macro_weight"] = float(config.beta_macro_weight) * fraction
    return LossConfig(**values)


def ramp_loss_config(config: LossConfig, epoch: int) -> LossConfig:
    """Linearly interpolate the ``warmup_*`` weights toward the configured
    (target) weights, instead of ``scheduled_loss_config``'s instant switch.

    ``lambda(e) = warmup + (target - warmup) * clamp((e - ramp_start_epoch) /
    ramp_epochs, 0, 1)``, applied independently to every ramp-aware weight.  At
    ``t=0`` every weight equals its ``warmup_*`` value; at ``t=1`` it equals the
    configured value exactly. ``ramp_epochs <= 0`` disables the ramp (no-op,
    returns ``config`` unchanged) so callers should fall back to
    ``scheduled_loss_config`` in that case.
    """
    if epoch < 1:
        raise ValueError("epoch must be one-indexed and >= 1")
    if config.ramp_epochs <= 0:
        return config
    progress = (epoch - config.ramp_start_epoch) / config.ramp_epochs
    t = min(1.0, max(0.0, progress))

    def lerp(warmup_value: float, target_value: float) -> float:
        return warmup_value + (target_value - warmup_value) * t

    return replace(
        config,
        beta_mse_weight=lerp(config.warmup_beta_mse_weight, config.beta_mse_weight),
        beta_huber_weight=lerp(config.warmup_beta_huber_weight, config.beta_huber_weight),
        residual_huber_weight=lerp(config.warmup_residual_huber_weight, config.residual_huber_weight),
        shrinkage_weight=lerp(config.warmup_shrinkage_weight, config.shrinkage_weight),
        locus_pearson_weight=lerp(config.warmup_locus_pearson_weight, config.locus_pearson_weight),
        locus_lower_tail_weight=lerp(config.warmup_locus_lower_tail_weight, config.locus_lower_tail_weight),
        pairwise_difference_weight=lerp(
            config.warmup_pairwise_difference_weight, config.pairwise_difference_weight
        ),
        centered_mse_weight=lerp(config.warmup_centered_mse_weight, config.centered_mse_weight),
        amplitude_weight=lerp(config.warmup_amplitude_weight, config.amplitude_weight),
    )


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

    Shared by every MAS-PCC-oriented loss below: correlation is scale-invariant
    to this centring, but centered_mse_weight/amplitude_weight are not -- they
    penalize the model shrinking the centred signal toward zero, which is
    exactly how a pure correlation objective can be gamed by amplitude
    collapse without these terms.

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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pearson correlation across samples, independently for every CpG.

    Returns ``(correlations, valid_locus_mask)``.  The calculation is promoted to
    float32 under AMP and excludes missing observations before centring, so NaNs
    cannot enter the backward graph.
    """
    _validate_locus_batch_shape(prediction, target, mask)
    if min_observed_samples < 2:
        raise ValueError("min_observed_samples must be >= 2")

    pred_centred, truth_centred, _, _, pred_ss, truth_ss, has_min_samples = _locus_centred_stats(
        prediction, target, mask, min_observed_samples
    )
    covariance = (pred_centred * truth_centred).sum(dim=0)
    valid = has_min_samples & (pred_ss > epsilon) & (truth_ss > epsilon)
    scale = torch.sqrt((pred_ss * truth_ss).clamp_min(epsilon))
    correlations = covariance / scale
    correlations = correlations.clamp(-1.0, 1.0)
    return correlations, valid


def locus_centered_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    min_observed_samples: int,
) -> tuple[torch.Tensor, int]:
    """Mean squared error between per-locus centred prediction and target.

    Unlike ``masked_locus_pearson``, this is *not* scale-invariant: shrinking
    the predicted amplitude toward zero strictly increases this loss once the
    true centred signal has real variance, so it cannot be gamed by amplitude
    collapse the way a pure correlation objective can.
    """
    _validate_locus_batch_shape(prediction, target, mask)
    if min_observed_samples < 2:
        raise ValueError("min_observed_samples must be >= 2")

    pred_centred, truth_centred, _, _, _, _, has_min_samples = _locus_centred_stats(
        prediction, target, mask, min_observed_samples
    )
    combined_mask = mask & has_min_samples.unsqueeze(0)
    squared_error = (pred_centred - truth_centred) ** 2
    loss = masked_mean(squared_error, combined_mask)
    return loss, int(has_min_samples.sum().detach().cpu())


def locus_amplitude_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    min_observed_samples: int,
    epsilon: float,
) -> tuple[torch.Tensor, int]:
    """Penalize the per-locus predicted/true amplitude ratio deviating from 1.

    ``amplitude = std(pred_centred) / std(true_centred)`` -- the same quantity
    reported as ``dynamic_amplitude_ratio`` in metrics.py, but computed per CpG
    and pooled across the batch instead of once globally. Deliberately does
    *not* reuse masked_locus_pearson's validity mask: that mask excludes
    near-zero-variance predictions, which is exactly the collapsed case this
    loss must penalize, not skip.
    """
    _validate_locus_batch_shape(prediction, target, mask)
    if min_observed_samples < 2:
        raise ValueError("min_observed_samples must be >= 2")

    _, _, _, denominator, pred_ss, truth_ss, has_min_samples = _locus_centred_stats(
        prediction, target, mask, min_observed_samples
    )
    valid = has_min_samples & (truth_ss > epsilon)
    pred_std = torch.sqrt(pred_ss / denominator + epsilon)
    truth_std = torch.sqrt(truth_ss / denominator + epsilon)
    ratio = pred_std / truth_std
    per_locus_loss = (ratio - 1.0) ** 2
    valid_values = per_locus_loss[valid]
    if valid_values.numel() == 0:
        zero = prediction.sum() * 0.0
        return zero, 0
    return valid_values.mean(), int(valid_values.numel())


def locus_correlation_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    config: LossConfig,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    correlations, valid = masked_locus_pearson(
        prediction,
        target,
        mask,
        min_observed_samples=config.locus_min_observed_samples,
        epsilon=config.locus_pearson_epsilon,
    )
    valid_values = correlations[valid]
    if valid_values.numel() == 0:
        zero = prediction.sum() * 0.0
        return zero, zero, 0

    mean_loss = 1.0 - valid_values.mean()
    fraction = config.locus_lower_tail_fraction
    if not 0.0 < fraction <= 1.0:
        raise ValueError("locus_lower_tail_fraction must be in (0, 1]")
    k = max(1, math.ceil(fraction * valid_values.numel()))
    lower_values = torch.topk(valid_values, k=k, largest=False, sorted=False).values
    lower_tail_loss = 1.0 - lower_values.mean()
    return mean_loss, lower_tail_loss, int(valid_values.numel())


def _masked_locus_moments(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Per-CpG moments shared by the biological-fidelity structured losses below.

    Returns ``(counts, pred_mean, truth_mean, pred_centred, truth_centred,
    pred_ss, truth_ss, covariance)``, all float32 with missing observations
    excluded before centring.
    """
    _validate_locus_batch_shape(prediction, target, mask)
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
    covariance = (pred_centred * truth_centred).sum(dim=0)
    return counts, pred_mean, truth_mean, pred_centred, truth_centred, pred_ss, truth_ss, covariance


def global_prior_ratio_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    prior: torch.Tensor,
    mask: torch.Tensor,
    config: LossConfig,
) -> torch.Tensor:
    """Observed-cell MSE ratio against the same fixed prior used in evaluation."""
    pred = prediction.float()
    truth = target.float()
    prior_matrix = prior.float().unsqueeze(0).expand_as(pred)
    model_sse = torch.where(mask, (pred - truth).square(), torch.zeros_like(pred)).sum()
    prior_sse = torch.where(
        mask, (prior_matrix - truth).square(), torch.zeros_like(pred)
    ).sum()
    ratio = model_sse / prior_sse.clamp_min(config.global_prior_ratio_epsilon)
    if config.global_prior_ratio_clip > 0:
        ratio = ratio.clamp_max(config.global_prior_ratio_clip)
    return ratio


def locus_skill_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    prior: torch.Tensor,
    mask: torch.Tensor,
    config: LossConfig,
) -> tuple[torch.Tensor, int]:
    """Mean clipped per-CpG SSE ratio against the fixed non-leaky prior.

    Minimizing this is equivalent to maximizing mean per-CpG skill. Eligibility
    excludes low-variance loci and unstable near-zero prior denominators; the
    ordinary beta MSE still trains on every observed entry.
    """
    counts, _, _, _, _, _, truth_ss, _ = _masked_locus_moments(prediction, target, mask)
    pred = prediction.float()
    truth = target.float()
    prior_matrix = prior.float().unsqueeze(0).expand_as(pred)
    model_sse = torch.where(mask, (pred - truth).square(), torch.zeros_like(pred)).sum(dim=0)
    prior_sse = torch.where(mask, (prior_matrix - truth).square(), torch.zeros_like(pred)).sum(dim=0)
    target_std = torch.sqrt(truth_ss / counts.clamp_min(1.0))
    minimum_prior_sse = config.locus_skill_denominator_epsilon * counts.clamp_min(1.0)
    valid = (
        (counts >= config.locus_min_observed_samples)
        & (target_std >= config.locus_min_target_std)
        & (prior_sse > minimum_prior_sse)
    )
    if not valid.any():
        return prediction.sum() * 0.0, 0
    ratio = model_sse[valid] / prior_sse[valid].clamp_min(minimum_prior_sse[valid])
    if config.locus_skill_loss_clip > 0:
        ratio = ratio.clamp_max(config.locus_skill_loss_clip)
    return ratio.mean(), int(valid.sum().detach().cpu())


def locus_ccc_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    config: LossConfig,
) -> tuple[torch.Tensor, int]:
    """Mean ``1 - Lin CCC`` across quantitatively evaluable CpGs."""
    counts, pred_mean, truth_mean, _, _, pred_ss, truth_ss, covariance = _masked_locus_moments(
        prediction, target, mask
    )
    target_std = torch.sqrt(truth_ss / counts.clamp_min(1.0))
    mean_gap_ss = counts * (pred_mean - truth_mean).square()
    denominator = pred_ss + truth_ss + mean_gap_ss
    ccc = 2.0 * covariance / denominator.clamp_min(config.locus_pearson_epsilon)
    valid = (
        (counts >= config.locus_min_observed_samples)
        & (target_std >= config.locus_min_target_std)
        & (denominator > config.locus_pearson_epsilon)
    )
    if not valid.any():
        return prediction.sum() * 0.0, 0
    return (1.0 - ccc[valid].clamp(-1.0, 1.0)).mean(), int(valid.sum().detach().cpu())


def within_cancer_dynamic_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    cancer_type_codes: torch.Tensor,
    config: LossConfig,
) -> tuple[torch.Tensor, int, int]:
    """Per-CpG within-cancer centred SSE ratio against zero dynamics."""
    if cancer_type_codes.ndim != 1 or len(cancer_type_codes) != prediction.shape[0]:
        raise ValueError("cancer_type_codes must have one value per sample")
    pred = prediction.float()
    truth = target.float()
    pred_centred = torch.zeros_like(pred)
    truth_centred = torch.zeros_like(truth)
    centred_mask = torch.zeros_like(mask)
    valid_groups = 0
    for code in torch.unique(cancer_type_codes):
        rows = cancer_type_codes == code
        if int(rows.sum().detach().cpu()) < config.within_cancer_min_samples:
            continue
        group_mask = mask[rows]
        counts = group_mask.sum(dim=0)
        eligible_columns = counts >= config.within_cancer_min_samples
        if not eligible_columns.any():
            continue
        group_pred = pred[rows]
        group_truth = truth[rows]
        denominator = counts.clamp_min(1).float()
        pred_mean = torch.where(group_mask, group_pred, torch.zeros_like(group_pred)).sum(0) / denominator
        truth_mean = torch.where(group_mask, group_truth, torch.zeros_like(group_truth)).sum(0) / denominator
        local_mask = group_mask & eligible_columns.unsqueeze(0)
        pred_centred[rows] = torch.where(local_mask, group_pred - pred_mean.unsqueeze(0), torch.zeros_like(group_pred))
        truth_centred[rows] = torch.where(local_mask, group_truth - truth_mean.unsqueeze(0), torch.zeros_like(group_truth))
        centred_mask[rows] = local_mask
        valid_groups += 1
    counts = centred_mask.sum(dim=0)
    baseline_sse = truth_centred.square().sum(dim=0)
    model_sse = ((pred_centred - truth_centred).square() * centred_mask.float()).sum(dim=0)
    target_std = torch.sqrt(baseline_sse / counts.clamp_min(1.0))
    minimum_sse = config.locus_skill_denominator_epsilon * counts.clamp_min(1.0)
    valid_loci = (
        (counts >= config.locus_min_observed_samples)
        & (target_std >= config.locus_min_target_std)
        & (baseline_sse > minimum_sse)
    )
    if not valid_loci.any():
        return prediction.sum() * 0.0, 0, valid_groups
    ratio = model_sse[valid_loci] / baseline_sse[valid_loci].clamp_min(minimum_sse[valid_loci])
    if config.locus_skill_loss_clip > 0:
        ratio = ratio.clamp_max(config.locus_skill_loss_clip)
    return ratio.mean(), int(valid_loci.sum().detach().cpu()), valid_groups


def pairwise_difference_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_pairs: int,
    huber_delta: float,
) -> tuple[torch.Tensor, int]:
    """Huber loss on inter-sample differences for the same CpG.

    Pair selection is deterministic and shared across loci.  It uses an evenly
    spaced subset of all upper-triangular sample pairs when the batch contains
    more pairs than requested.
    """
    n_samples = prediction.shape[0]
    if n_samples < 2 or max_pairs <= 0:
        return prediction.sum() * 0.0, 0
    pairs = torch.triu_indices(n_samples, n_samples, offset=1, device=prediction.device)
    if pairs.shape[1] > max_pairs:
        select = torch.linspace(
            0, pairs.shape[1] - 1, steps=max_pairs, device=prediction.device
        ).round().long()
        pairs = pairs[:, select]
    left, right = pairs[0], pairs[1]
    pair_mask = mask[left] & mask[right]
    pred_diff = prediction[left].float() - prediction[right].float()
    true_diff = target[left].float() - target[right].float()
    losses = F.huber_loss(pred_diff, true_diff, reduction="none", delta=huber_delta)
    return masked_mean(losses, pair_mask), int(pair_mask.sum().detach().cpu())


def residual_loss(
    outputs: dict[str, torch.Tensor],
    target_beta: torch.Tensor,
    prior: torch.Tensor,
    config: LossConfig,
    epsilon: float = 1e-4,
    tertile_labels: torch.Tensor | None = None,
    cancer_type_codes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    mask = torch.isfinite(target_beta)
    prediction = outputs["beta"]
    # Replace missing targets before any differentiable operation. Masking a tensor
    # only after arithmetic with NaNs can still create NaN gradients (0 * NaN).
    safe_target = torch.where(mask, target_beta, prior.unsqueeze(0))
    beta_mse = masked_mean((prediction - safe_target) ** 2, mask)
    beta_huber = masked_mean(
        F.huber_loss(prediction, safe_target, reduction="none", delta=config.beta_huber_delta),
        mask,
    )
    beta_macro_mse = beta_mse
    if tertile_labels is not None:
        pieces = []
        for label in range(3):
            column_mask = mask & (tertile_labels.unsqueeze(0) == label)
            if column_mask.any():
                pieces.append(masked_mean((prediction - safe_target) ** 2, column_mask))
        if pieces:
            beta_macro_mse = torch.stack(pieces).mean()

    clipped_target = safe_target.clamp(epsilon, 1.0 - epsilon)
    clipped_prior = prior.clamp(epsilon, 1.0 - epsilon)
    target_delta = torch.logit(clipped_target) - torch.logit(clipped_prior).unsqueeze(0)
    huber = masked_mean(
        F.huber_loss(outputs["delta_logit"], target_delta, reduction="none", delta=config.residual_huber_delta),
        mask,
    )
    shrinkage = masked_mean(outputs["delta_logit"] ** 2, mask)

    if config.locus_pearson_weight != 0.0 or config.locus_lower_tail_weight != 0.0:
        pearson_loss, lower_tail_loss, valid_loci = locus_correlation_losses(
            prediction, safe_target, mask, config
        )
    else:
        pearson_loss = prediction.sum() * 0.0
        lower_tail_loss = prediction.sum() * 0.0
        valid_loci = 0

    if config.pairwise_difference_weight != 0.0:
        pairwise_loss, valid_pairwise_entries = pairwise_difference_loss(
            prediction,
            safe_target,
            mask,
            max_pairs=config.pairwise_pairs_per_batch,
            huber_delta=config.pairwise_huber_delta,
        )
    else:
        pairwise_loss = prediction.sum() * 0.0
        valid_pairwise_entries = 0

    if config.centered_mse_weight != 0.0:
        centered_mse_loss, valid_centered_loci = locus_centered_mse(
            prediction, safe_target, mask, min_observed_samples=config.locus_min_observed_samples
        )
    else:
        centered_mse_loss = prediction.sum() * 0.0
        valid_centered_loci = 0

    if config.amplitude_weight != 0.0:
        amplitude_loss, valid_amplitude_loci = locus_amplitude_loss(
            prediction,
            safe_target,
            mask,
            min_observed_samples=config.locus_min_observed_samples,
            epsilon=config.amplitude_epsilon,
        )
    else:
        amplitude_loss = prediction.sum() * 0.0
        valid_amplitude_loci = 0

    if config.global_prior_ratio_weight != 0.0:
        global_ratio = global_prior_ratio_loss(prediction, safe_target, prior, mask, config)
    else:
        global_ratio = prediction.sum() * 0.0
    if config.locus_skill_weight != 0.0:
        skill_loss, valid_skill_loci = locus_skill_loss(prediction, safe_target, prior, mask, config)
    else:
        skill_loss = prediction.sum() * 0.0
        valid_skill_loci = 0
    if config.locus_ccc_weight != 0.0:
        ccc_loss, valid_ccc_loci = locus_ccc_loss(prediction, safe_target, mask, config)
    else:
        ccc_loss = prediction.sum() * 0.0
        valid_ccc_loci = 0
    if config.within_cancer_dynamic_weight != 0.0:
        if cancer_type_codes is None:
            raise ValueError("within_cancer_dynamic_weight requires cancer_type_codes")
        within_cancer_loss, valid_within_cancer_loci, valid_within_cancer_groups = within_cancer_dynamic_loss(
            prediction, safe_target, mask, cancer_type_codes, config
        )
    else:
        within_cancer_loss = prediction.sum() * 0.0
        valid_within_cancer_loci = valid_within_cancer_groups = 0

    total = (
        config.beta_mse_weight * beta_mse
        + config.beta_huber_weight * beta_huber
        + config.beta_macro_weight * beta_macro_mse
        + config.residual_huber_weight * huber
        + config.shrinkage_weight * shrinkage
        + config.locus_pearson_weight * pearson_loss
        + config.locus_lower_tail_weight * lower_tail_loss
        + config.pairwise_difference_weight * pairwise_loss
        + config.centered_mse_weight * centered_mse_loss
        + config.amplitude_weight * amplitude_loss
        + config.global_prior_ratio_weight * global_ratio
        + config.locus_skill_weight * skill_loss
        + config.locus_ccc_weight * ccc_loss
        + config.within_cancer_dynamic_weight * within_cancer_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "beta_mse": float(beta_mse.detach().cpu()),
        "beta_huber": float(beta_huber.detach().cpu()),
        "beta_macro_mse": float(beta_macro_mse.detach().cpu()),
        "residual_huber": float(huber.detach().cpu()),
        "shrinkage": float(shrinkage.detach().cpu()),
        "locus_pearson_loss": float(pearson_loss.detach().cpu()),
        "locus_lower_tail_loss": float(lower_tail_loss.detach().cpu()),
        "pairwise_difference_loss": float(pairwise_loss.detach().cpu()),
        "centered_mse_loss": float(centered_mse_loss.detach().cpu()),
        "amplitude_loss": float(amplitude_loss.detach().cpu()),
        "global_prior_ratio_loss": float(global_ratio.detach().cpu()),
        "locus_skill_loss": float(skill_loss.detach().cpu()),
        "locus_ccc_loss": float(ccc_loss.detach().cpu()),
        "within_cancer_dynamic_loss": float(within_cancer_loss.detach().cpu()),
        "valid_correlation_loci": valid_loci,
        "valid_pairwise_entries": valid_pairwise_entries,
        "valid_centered_loci": valid_centered_loci,
        "valid_amplitude_loci": valid_amplitude_loci,
        "valid_skill_loci": valid_skill_loci,
        "valid_ccc_loci": valid_ccc_loci,
        "valid_within_cancer_loci": valid_within_cancer_loci,
        "valid_within_cancer_groups": valid_within_cancer_groups,
        "observed": int(mask.sum().detach().cpu()),
    }
