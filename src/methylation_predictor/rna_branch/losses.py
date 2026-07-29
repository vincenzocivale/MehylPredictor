"""Losses for bounded residual methylation prediction."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .config import LossConfig


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask]
    if selected.numel() == 0:
        return values.sum() * 0.0
    return selected.mean()


def residual_loss(
    outputs: dict[str, torch.Tensor],
    target_beta: torch.Tensor,
    prior: torch.Tensor,
    config: LossConfig,
    epsilon: float = 1e-4,
    tertile_labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = torch.isfinite(target_beta)
    prediction = outputs["beta"]
    # Replace missing targets before any differentiable operation. Masking a tensor
    # only after arithmetic with NaNs can still create NaN gradients (0 * NaN).
    safe_target = torch.where(mask, target_beta, prior.unsqueeze(0))
    beta_mse = masked_mean((prediction - safe_target) ** 2, mask)
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
    total = (
        config.beta_mse_weight * beta_mse
        + config.beta_macro_weight * beta_macro_mse
        + config.residual_huber_weight * huber
        + config.shrinkage_weight * shrinkage
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "beta_mse": float(beta_mse.detach().cpu()),
        "beta_macro_mse": float(beta_macro_mse.detach().cpu()),
        "residual_huber": float(huber.detach().cpu()),
        "shrinkage": float(shrinkage.detach().cpu()),
        "observed": int(mask.sum().detach().cpu()),
    }
