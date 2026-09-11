"""Locus-conditioned retrieval primitives used by the paper candidates."""

from __future__ import annotations

import torch
from torch import nn


class FeedForwardResidual(nn.Module):
    """Pre-norm 4x-expansion FFN with an internal residual."""

    def __init__(self, width: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * width, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class LocusToRNAAttention(nn.Module):
    """Bare multi-head cross-attention from locus states to RNA program tokens."""

    def __init__(self, width: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if width % n_heads:
            raise ValueError("cross-attention width must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = width // n_heads
        self.q_norm = nn.LayerNorm(width)
        self.k_norm = nn.LayerNorm(width)
        self.v_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.out = nn.Linear(width, width)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, loci: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        batch, n_programs, width = tokens.shape
        n_loci = loci.shape[0]
        q = self.query(self.q_norm(loci)).view(
            n_loci, self.n_heads, self.head_dim
        )
        k = self.key(self.k_norm(tokens)).view(
            batch, n_programs, self.n_heads, self.head_dim
        )
        v = self.value(self.v_norm(tokens)).view(
            batch, n_programs, self.n_heads, self.head_dim
        )
        scores = torch.einsum("lhd,bkhd->bhlk", q, k) / (self.head_dim**0.5)
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        attended = torch.einsum("bhlk,bkhd->blhd", weights, v)
        return self.out(attended.reshape(batch, n_loci, width))


class BatchedLocusToRNAAttention(nn.Module):
    """Cross-attention for an already patient-conditioned locus query."""

    def __init__(self, width: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if width % n_heads:
            raise ValueError("cross-attention width must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = width // n_heads
        self.q_norm = nn.LayerNorm(width)
        self.k_norm = nn.LayerNorm(width)
        self.v_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.out = nn.Linear(width, width)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        batch, n_loci, width = query.shape
        n_programs = tokens.shape[1]
        q = self.query(self.q_norm(query)).view(
            batch, n_loci, self.n_heads, self.head_dim
        )
        k = self.key(self.k_norm(tokens)).view(
            batch, n_programs, self.n_heads, self.head_dim
        )
        v = self.value(self.v_norm(tokens)).view(
            batch, n_programs, self.n_heads, self.head_dim
        )
        scores = torch.einsum("blhd,bkhd->bhlk", q, k) / (self.head_dim**0.5)
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        attended = torch.einsum("bhlk,bkhd->blhd", weights, v)
        return self.out(attended.reshape(batch, n_loci, width))
