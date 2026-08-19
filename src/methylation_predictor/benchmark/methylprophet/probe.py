from __future__ import annotations

import torch
from torch import nn


class ScalarProbeNet(nn.Module):
    """Historical probe shape: LayerNorm -> 256 -> 64 -> 1."""

    def __init__(self, dim: int = 1536, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
