"""RNA-only training objectives and metrics."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import ObjectiveConfig


@dataclass
class LossOutput:
    loss: torch.Tensor
    total_mse: torch.Tensor
    within_mse: torch.Tensor
    consistency: torch.Tensor


def compute_loss(
    objective: ObjectiveConfig,
    embedding: torch.Tensor,
    prediction_total: torch.Tensor,
    prediction_within: torch.Tensor,
    batch: dict[str, torch.Tensor],
    augmented_embedding: torch.Tensor | None = None,
) -> LossOutput:
    total_mse = F.mse_loss(prediction_total, batch["target_total"])
    within_mse = F.mse_loss(prediction_within, batch["target_within"])
    consistency = torch.zeros((), device=embedding.device, dtype=embedding.dtype)
    if objective.consistency_weight > 0:
        if augmented_embedding is None:
            raise ValueError("consistency objective requires augmented embedding")
        consistency = (1.0 - F.cosine_similarity(embedding, augmented_embedding, dim=-1)).mean()
    loss = (
        objective.total_weight * total_mse
        + objective.within_weight * within_mse
        + objective.consistency_weight * consistency
    )
    return LossOutput(loss=loss, total_mse=total_mse, within_mse=within_mse, consistency=consistency)


def r2_score_torch(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    numerator = (target - prediction).square().sum()
    denominator = (target - target.mean(dim=0, keepdim=True)).square().sum().clamp_min(1e-12)
    return 1.0 - numerator / denominator
