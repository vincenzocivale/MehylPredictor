"""Trainable RNA-only readout model with matched linear decoders."""
from __future__ import annotations

import torch
from torch import nn

from .config import ReadoutConfig
from .poolers import BasePooler, ReadoutOutput


class RNAReadoutModel(nn.Module):
    def __init__(self, config: ReadoutConfig, pooler: BasePooler, target_dim: int) -> None:
        super().__init__()
        self.config = config
        self.pooler = pooler
        self.total_decoder = nn.Linear(config.model.output_dim, target_dim)
        self.within_decoder = nn.Linear(config.model.output_dim, target_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[ReadoutOutput, torch.Tensor, torch.Tensor]:
        output = self.pooler(batch)
        return output, self.total_decoder(output.embedding), self.within_decoder(output.embedding)

    def encode(self, batch: dict[str, torch.Tensor]) -> ReadoutOutput:
        return self.pooler(batch)
