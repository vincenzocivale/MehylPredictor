"""Shared optimizer/scheduler utilities for every trainable model."""
from __future__ import annotations

import math

import torch

SUPPORTED_SCHEDULERS = ("constant", "cosine", "cosine_warmup")


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    name: str,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a deterministic scheduler stepped after optimizer updates.

    ``total_steps`` is a fixed horizon.  A final refit can intentionally execute
    fewer steps while keeping the horizon selected during development, thereby
    reproducing the development LR trajectory up to the selected epoch.
    """
    name = str(name).lower()
    if name not in SUPPORTED_SCHEDULERS:
        raise ValueError(f"scheduler must be one of {SUPPORTED_SCHEDULERS}; got {name!r}")
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    warmup_steps = min(int(warmup_steps), int(total_steps))

    def lr_scale(step: int) -> float:
        current = min(max(int(step) + 1, 1), int(total_steps))
        if name == "constant":
            return 1.0
        if name == "cosine_warmup" and warmup_steps and current <= warmup_steps:
            return current / warmup_steps
        decay_start = warmup_steps if name == "cosine_warmup" else 0
        decay_steps = max(total_steps - decay_start, 1)
        progress = min(max((current - decay_start) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
