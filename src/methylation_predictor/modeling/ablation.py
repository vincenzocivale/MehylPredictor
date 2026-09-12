"""Depth-vs-residual-vs-FFN-capacity ablation, disentangling J0 and J1.

J0 (``SingleRetrievalPredictor``) and J1 (``IterativeRetrievalPredictor``)
differ on several confounded dimensions at once: retrieval (cross-attention
against RNA) depth (1 vs 4 blocks), whether each block's cross-attention
output is added residually to the running locus state (never vs always), the
number of FFN blocks (J0 has ZERO -- its cross-attention output goes straight
to the final concat; J1 has 4, one per retrieval block), and parameter count.
An early, accidentally-built version of J1 (2 blocks, only the second
residual) tracked J0's training loss almost exactly, which is suggestive but
not conclusive on its own -- this module isolates the axes cleanly:

  ``ablation_depth1_residual``    -- n_blocks=1, attn_residual=True
      Same attention depth as J0 (1 cross-attention call), but with the
      residual J1 has (and, like J1, one FFN block after it -- J0 has none).
      Isolates whether the residual ALONE (no extra RNA-attention depth,
      though WITH the extra FFN capacity J1 also carries) already produces
      J1's gain.

  ``ablation_depth4_noresidual``  -- n_blocks=4, attn_residual=False
      Same depth as J1 (4 cross-attention + 4 FFN blocks), but every block's
      attention output REPLACES the running state instead of being added to
      it (matching the accidental first J1 attempt's mechanism, applied
      uniformly to every block instead of only the first). Isolates whether
      depth ALONE (no residual) produces any gain over J0.

  ``ablation_attn1_ffn4_residual`` -- n_blocks=1, attn_residual=True,
      n_ffn_only_blocks=3
      Only ONE real RNA cross-attention call (same as J0/cell A -- the
      expensive, memory-heavy operation), followed by 3 EXTRA residual
      FFN-only blocks that never touch RNA again, so the TOTAL block count
      (4) and FFN count (4) match J1's exactly. Isolates whether J1's gain
      comes from repeatedly re-querying RNA (attention depth) or simply from
      more generic residual non-linear capacity/depth (which is far cheaper
      to compute than repeated cross-attention against the RNA program
      tokens) -- the question of whether "servono più blocchi ma non pesanti
      come la cross attention" explains the gain instead of attention depth
      itself.

Everything else (functional encoder, mean head, final regressor, loss,
training protocol) is identical to J0/J1 -- see their docstrings in
``reference.py``.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig
from .retrieval import BatchedLocusToRNAAttention, FeedForwardResidual
from .rna import ProgramTokenEncoder, consume_legacy_attention_initialization


class DepthResidualAblationPredictor(nn.Module):
    """Configurable-depth, optionally-residual RNA-retrieval stack."""

    N_TRACKS = 4165
    DENSE_DIM = 23
    WIDTH = 256
    LOCUS_CHUNK = 1024

    def __init__(
        self,
        input_dim: int,
        config: ModelConfig,
        *,
        n_blocks: int,
        attn_residual: bool,
        n_ffn_only_blocks: int = 0,
        final_regressor_dropout: float = 0.15,
        use_mean_proxy: bool = True,
    ):
        super().__init__()
        if n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")
        if n_ffn_only_blocks < 0:
            raise ValueError("n_ffn_only_blocks must be >= 0")
        self.n_blocks = int(n_blocks)
        self.attn_residual = bool(attn_residual)
        self.n_ffn_only_blocks = int(n_ffn_only_blocks)
        enc = config.encoder
        if enc.kind != "locus_attention" or enc.program_dim != self.WIDTH:
            raise ValueError(
                "DepthResidualAblationPredictor requires locus_attention RNA "
                "tokens with program_dim=256"
            )

        self.rna_encoder = ProgramTokenEncoder(
            input_dim=input_dim,
            n_programs=enc.n_programs,
            program_dim=enc.program_dim,
            bottleneck_dim=enc.latent_dim,
            layer_norm=enc.layer_norm,
        )
        consume_legacy_attention_initialization(self.WIDTH)

        self.track_embedding = nn.EmbeddingBag(
            self.N_TRACKS, self.WIDTH, mode="mean", include_last_offset=True,
        )
        self.dense_encoder = nn.Sequential(
            nn.LayerNorm(self.DENSE_DIM), nn.Linear(self.DENSE_DIM, self.WIDTH), nn.GELU(),
        )
        self.locus_norm = nn.LayerNorm(self.WIDTH)
        mean_head = nn.Sequential(
            nn.LayerNorm(self.WIDTH), nn.Linear(self.WIDTH, 128), nn.GELU(), nn.Linear(128, 1),
        )
        self.mean_head = mean_head if use_mean_proxy else None
        self.retrieval_attn = nn.ModuleList([
            BatchedLocusToRNAAttention(self.WIDTH, enc.n_heads, enc.dropout) for _ in range(self.n_blocks)
        ])
        self.retrieval_ffn = nn.ModuleList([
            FeedForwardResidual(self.WIDTH, enc.dropout) for _ in range(self.n_blocks)
        ])
        # Extra FFN-only blocks: no attention, never touch RNA again -- pure
        # non-linear capacity/depth on the already RNA-conditioned state, at
        # a fraction of cross-attention's compute/memory cost. Always
        # residual (FeedForwardResidual bakes its own residual in
        # regardless of `attn_residual`, which only governs the attention
        # step above).
        self.ffn_only_blocks = nn.ModuleList([
            FeedForwardResidual(self.WIDTH, enc.dropout) for _ in range(self.n_ffn_only_blocks)
        ])
        self.h_c_concat_norm = nn.LayerNorm(self.WIDTH)
        self.r_pc_concat_norm = nn.LayerNorm(self.WIDTH)
        self.final_regressor_dropout = float(final_regressor_dropout)
        self.final_regressor = nn.Sequential(
            nn.Linear(2 * self.WIDTH, self.WIDTH), nn.GELU(), nn.Dropout(self.final_regressor_dropout),
            nn.Linear(self.WIDTH, 128), nn.GELU(), nn.Dropout(self.final_regressor_dropout),
            nn.Linear(128, 1),
        )

    @property
    def requires_cpg_positions(self) -> bool:
        return False

    def _retrieve_and_predict(self, h_c_chunk: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        state = h_c_chunk[None, :, :].expand(batch, -1, -1)
        for attn, ffn in zip(self.retrieval_attn, self.retrieval_ffn):
            a = attn(state, tokens)
            state = state + a if self.attn_residual else a
            state = ffn(state)  # FeedForwardResidual: always has its own internal residual
        for ffn_only in self.ffn_only_blocks:
            state = ffn_only(state)  # no RNA re-query, just extra residual FFN capacity

        h_for_concat = self.h_c_concat_norm(h_c_chunk)
        r_for_concat = self.r_pc_concat_norm(state)
        z_pc = torch.cat(
            [h_for_concat[None, :, :].expand(batch, -1, -1), r_for_concat], dim=-1,
        )
        return self.final_regressor(z_pc).squeeze(-1)

    def forward(
        self,
        rna: torch.Tensor,
        cpg_embedding: torch.Tensor | None = None,
        cpg_positions: torch.Tensor | None = None,
        *,
        functional_track_indices: torch.Tensor,
        functional_offsets: torch.Tensor,
        functional_dense: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del cpg_embedding, cpg_positions
        tokens = self.rna_encoder(rna)

        peak = self.track_embedding(functional_track_indices, functional_offsets)
        dense = self.dense_encoder(functional_dense)
        h_c = self.locus_norm(peak + dense)
        mu_hat = (
            None if self.mean_head is None
            else torch.sigmoid(self.mean_head(h_c).squeeze(-1))
        )

        n_loci = h_c.shape[0]
        if n_loci <= self.LOCUS_CHUNK:
            logit = self._retrieve_and_predict(h_c, tokens)
        else:
            chunks = []
            for start in range(0, n_loci, self.LOCUS_CHUNK):
                chunk = h_c[start:start + self.LOCUS_CHUNK]
                if self.training and torch.is_grad_enabled():
                    chunks.append(
                        checkpoint(self._retrieve_and_predict, chunk, tokens, use_reentrant=False)
                    )
                else:
                    chunks.append(self._retrieve_and_predict(chunk, tokens))
            logit = torch.cat(chunks, dim=1)

        beta = torch.sigmoid(logit)
        return {
            "beta": beta,
            "delta_logit": logit,
            "raw_delta": logit,
            "prediction_logit": logit,
            "mu_logit": None,
            "mu_hat": mu_hat,
            "residual_logit": logit,
            "h_cpg": h_c,
        }
