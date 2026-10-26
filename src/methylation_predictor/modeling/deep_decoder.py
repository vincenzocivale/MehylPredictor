"""Deeper post-query decoder (``DeepDecoder``) and its Perceiver predictor (``perceiver_deep_decoder``).

Only the part after the CpG query changes: the RNA tokens are re-read in several
cross-attention rounds (the shallow decoder reads them once) and the head is wider.
The CpG branch (``h_c``, ``functional_ffn``, mean head) is untouched, so a comparison
with ``perceiver_cpg_query`` isolates the decoder.

The decoder is shared across CpGs but sees each one only through ``h_c``, so extra
capacity can memorise training CpGs. Guards, all train-time only except the init:

* ``cpg_dropout``: feature dropout on ``h_c`` entering the decoder, so it cannot
  key on an exact CpG fingerprint;
* ``drop_path``: stochastic depth per (patient, CpG) state, ramped over the rounds;
* ``zero_init``: rounds after the first start as the identity, i.e. training begins
  from the shallow decoder and only deepens where the data support it;
* ``dropout`` on attention/FFN/head (defaults to ``final_regressor_dropout``).
"""
from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .perceiver import PerceiverCpGPredictor
from .retrieval import FeedForwardResidual


def _drop_path(branch: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if not training or p == 0.0:
        return branch
    keep = torch.rand(*branch.shape[:-1], 1, device=branch.device) >= p
    return branch * keep / (1.0 - p)


class _GatedFFN(FeedForwardResidual):
    def __init__(self, width: int, dropout: float, drop_path: float, zero_init: bool):
        super().__init__(width, dropout)
        self.drop_path = drop_path
        if zero_init:
            nn.init.zeros_(self.ffn[3].weight)
            nn.init.zeros_(self.ffn[3].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + _drop_path(self.ffn(self.norm(x)), self.drop_path, self.training)


class _Round(nn.Module):
    """Pre-norm cross-attention to the RNA tokens, then residual FFNs."""

    def __init__(self, width: int, kv_width: int, n_heads: int, n_ffn: int, dropout: float,
                 drop_path: float, zero_init: bool):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, n_heads, dropout=dropout, kdim=kv_width,
                                          vdim=kv_width, batch_first=True)
        self.ffn = nn.ModuleList(_GatedFFN(width, dropout, drop_path, zero_init) for _ in range(n_ffn))
        self.drop_path = drop_path
        if zero_init:
            nn.init.zeros_(self.attn.out_proj.weight)
            nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, state: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm(state)
        state = state + _drop_path(self.attn(q, tokens, tokens, need_weights=False)[0], self.drop_path, self.training)
        for ffn in self.ffn:
            state = ffn(state)
        return state


class DeepDecoder(nn.Module):
    """(CpG query, CpG deep state, RNA tokens) -> logit for every patient x CpG pair."""

    def __init__(self, width: int = 256, *, query_dim: int = 256, token_dim: int = 256, n_heads: int = 4,
                 n_rounds: int = 4, ffn_per_round: int = 2, n_head_ffn: int = 2, dropout: float = 0.15,
                 drop_path: float = 0.1, cpg_dropout: float = 0.1, zero_init: bool = True,
                 zero_output: bool = False):
        super().__init__()
        if n_rounds < 1 or width % n_heads:
            raise ValueError("deep decoder needs n_rounds >= 1 and width divisible by n_heads")
        self.cpg_drop = nn.Dropout(cpg_dropout)
        self.query_proj = nn.Identity() if width == query_dim else nn.Linear(query_dim, width)
        self.rounds = nn.ModuleList(
            _Round(width, token_dim, n_heads, ffn_per_round, dropout, drop_path * (r + 1) / n_rounds,
                   zero_init and r > 0)
            for r in range(n_rounds)
        )
        self.state_norm = nn.LayerNorm(width)
        self.concat_norm = nn.LayerNorm(query_dim)
        self.head_in = nn.Sequential(nn.Linear(query_dim + width, width), nn.GELU(), nn.Dropout(dropout))
        self.head_ffn = nn.ModuleList(FeedForwardResidual(width, dropout) for _ in range(n_head_ffn))
        self.head_out = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))
        if zero_output:  # the decoder starts by predicting exactly 0
            nn.init.zeros_(self.head_out[-1].weight)
            nn.init.zeros_(self.head_out[-1].bias)

    def forward(self, query: torch.Tensor, concat: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """query/concat: (n_cpgs, query_dim); tokens: (n_patients, n_tokens, token_dim) -> (n_patients, n_cpgs)."""
        batch = tokens.shape[0]
        state = self.query_proj(self.cpg_drop(query))[None].expand(batch, -1, -1)
        for rnd in self.rounds:
            state = rnd(state, tokens)
        h = self.concat_norm(self.cpg_drop(concat))
        z = self.head_in(torch.cat([h[None].expand(batch, -1, -1), self.state_norm(state)], dim=-1))
        for ffn in self.head_ffn:
            z = ffn(z)
        return self.head_out(z).squeeze(-1)


class DeepDecoderCpGPredictor(PerceiverCpGPredictor):
    """Perceiver predictor with iterated RNA retrieval and a wider head after the CpG query."""

    def __init__(self, input_dim: int, config: ModelConfig, *, width: int = 256, n_rounds: int = 4,
                 ffn_per_round: int = 2, n_head_ffn: int = 2, dropout: float | None = None,
                 drop_path: float = 0.1, cpg_dropout: float = 0.1, zero_init: bool = True,
                 locus_chunk: int = 256, **kw):
        super().__init__(input_dim, config, **kw)
        # the parent's shallow decoder is replaced, not kept as dead parameters
        del (self.retrieval_attn, self.retrieval_ffn, self.h_c_concat_norm, self.r_pc_concat_norm,
             self.final_regressor, self.head_ffn)
        # smaller chunks than the parent bound activation memory (it checkpoints above LOCUS_CHUNK)
        self.LOCUS_CHUNK = int(locus_chunk)
        self.decoder = DeepDecoder(
            width, query_dim=self.WIDTH, token_dim=self.WIDTH, n_heads=config.encoder.n_heads,
            n_rounds=n_rounds, ffn_per_round=ffn_per_round, n_head_ffn=n_head_ffn,
            dropout=self.final_regressor_dropout if dropout is None else float(dropout),
            drop_path=drop_path, cpg_dropout=cpg_dropout, zero_init=zero_init)

    def _retrieve_and_predict(self, h_c_query_chunk, h_c_concat_chunk, tokens):
        return self.decoder(h_c_query_chunk, h_c_concat_chunk, tokens)
