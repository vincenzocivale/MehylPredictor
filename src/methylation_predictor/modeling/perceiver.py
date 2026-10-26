"""Perceiver RNA encoder + CpG-query cross-attention head.

Pipeline: RNA (G genes) -> patch tokens -> Perceiver latents (cross + self attn)
-> CpG query cross-attends the latents -> beta. The CpG representation is the
inherited functional query (``track_embedding`` + ``dense_encoder`` -> ``h_c``);
pass ``cpg_query`` (see ``modeling.cpg_query``) to swap it -- everything else is fixed.
"""
from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .final import EfficientSingleAttentionPredictor
from .retrieval import FeedForwardResidual


class PerceiverRNAEncoder(nn.Module):
    """Transcriptome -> (B, n_latents, width) via learned latents."""

    def __init__(self, input_dim: int, n_latents: int = 64, width: int = 256, n_heads: int = 4,
                 n_self_blocks: int = 2, patch: int = 256, dropout: float = 0.1):
        super().__init__()
        self.patch = patch
        self.n_patches = -(-input_dim // patch)
        self.pad = self.n_patches * patch - input_dim
        self.in_norm = nn.LayerNorm(input_dim)
        # one projection per gene patch: also acts as a positional embedding
        self.patch_proj = nn.Parameter(torch.randn(self.n_patches, patch, width) * patch ** -0.5)
        self.tok_norm = nn.LayerNorm(width)
        self.latents = nn.Parameter(torch.randn(n_latents, width) * 0.02)
        self.lat_norm = nn.LayerNorm(width)
        self.cross = nn.MultiheadAttention(width, n_heads, dropout=dropout, batch_first=True)
        self.cross_ffn = FeedForwardResidual(width, dropout)
        self.self_blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(width, n_heads, 4 * width, dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(n_self_blocks)
        )
        self.out_norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(self.in_norm(x), (0, self.pad)).view(x.shape[0], self.n_patches, self.patch)
        tok = self.tok_norm(torch.einsum("bpg,pgd->bpd", x, self.patch_proj))
        lat = self.latents.expand(x.shape[0], -1, -1)
        lat = lat + self.cross(self.lat_norm(lat), tok, tok, need_weights=False)[0]
        lat = self.cross_ffn(lat)
        for blk in self.self_blocks:
            lat = blk(lat)
        return self.out_norm(lat)


class PerceiverCpGPredictor(EfficientSingleAttentionPredictor):
    """Same CpG-query head/outputs as the final model, Perceiver RNA encoder."""

    def __init__(self, input_dim: int, config: ModelConfig, *, n_self_blocks: int = 2,
                 cpg_query: nn.Module | None = None, cpg_dim: int | None = None, **kw):
        super().__init__(input_dim, config, **kw)
        self.cpg_query = cpg_query
        if cpg_query is not None:  # drop the unused inherited functional query
            del self.track_embedding, self.dense_encoder, self.locus_norm
            if cpg_dim is not None:  # trainer provenance reads N_TRACKS/DENSE_DIM
                self.N_TRACKS, self.DENSE_DIM = 1, int(cpg_dim)
        enc = config.encoder
        # replaces the ProgramTokenEncoder built by the parent (n_programs -> n_latents)
        self.rna_encoder = PerceiverRNAEncoder(input_dim, enc.n_programs, self.WIDTH, enc.n_heads,
                                               n_self_blocks, dropout=enc.dropout)

    def _cpg_query(self, track_indices, offsets, dense):
        if self.cpg_query is None:
            return super()._cpg_query(track_indices, offsets, dense)
        return self.cpg_query(track_indices, offsets, dense)
