"""Independent NTv3 -> CpG mean and residual-scale predictor."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True, frozen=True)
class CpGStatisticsModelConfig:
    input_dim: int = 1536
    hidden_dim: int = 256
    bottleneck_dim: int = 64
    dropout: float = 0.1
    ensemble_seeds: tuple[int, ...] = (17, 29, 43)
    sigma_floor: float = 0.01


class _ScalarHead(nn.Module):
    def __init__(self, cfg: CpGStatisticsModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.input_dim),
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.bottleneck_dim),
            nn.GELU(),
            nn.Linear(cfg.bottleneck_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CpGStatisticsPredictor(nn.Module):
    """Predict ``mu`` and ``sigma`` from a frozen 1536-D NTv3 CpG embedding.

    The mean and scale branches are intentionally independent.  Combining them
    into one software component removes the historical hidden sigma pipeline,
    while independent parameter sets prevent sigma optimization from changing
    the already strong mean-prediction capacity established by the baseline.

    Raw targets are ``logit(mu)`` and ``log(sigma)``.  Public outputs are
    beta-space ``mu`` and positive logit-residual ``sigma``.
    """

    def __init__(self, config: CpGStatisticsModelConfig | None = None):
        super().__init__()
        self.config = config or CpGStatisticsModelConfig()
        mu_heads: list[nn.Module] = []
        sigma_heads: list[nn.Module] = []
        for seed in self.config.ensemble_seeds:
            torch.manual_seed(int(seed))
            mu_heads.append(_ScalarHead(self.config))
            # Stable but distinct initialization from the corresponding mean head.
            torch.manual_seed(int(seed) + 100_003)
            sigma_heads.append(_ScalarHead(self.config))
        self.mu_heads = nn.ModuleList(mu_heads)
        self.sigma_heads = nn.ModuleList(sigma_heads)

    def forward_raw_all(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = torch.stack([head(x) for head in self.mu_heads], dim=0)
        log_sigma = torch.stack([head(x) for head in self.sigma_heads], dim=0)
        return mu, log_sigma

    def forward_raw(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_sigma = self.forward_raw_all(x)
        return mu.mean(dim=0), log_sigma.mean(dim=0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu_logit, log_sigma = self.forward_raw(x)
        return {
            "mu": torch.sigmoid(mu_logit),
            "sigma": torch.exp(log_sigma).clamp_min(self.config.sigma_floor),
            "mu_logit": mu_logit,
            "log_sigma": log_sigma,
        }
