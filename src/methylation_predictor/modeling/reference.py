"""Paper-facing implementations of the current full-data candidates."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig
from .retrieval import (
    BatchedLocusToRNAAttention,
    FeedForwardResidual,
    LocusToRNAAttention,
)
from .rna import (
    ProgramTokenEncoder,
    consume_legacy_attention_initialization,
    migrate_legacy_candidate_state_dict,
)


class _LegacyCandidateCheckpointMixin:
    """Strict loader with one explicit pre-phase-2c migration."""

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        migrated = migrate_legacy_candidate_state_dict(state_dict)
        return super().load_state_dict(migrated, strict=strict, assign=assign)


class SingleRetrievalPredictor(_LegacyCandidateCheckpointMixin, nn.Module):
    """J0: functional locus + pure single-step RNA retrieval."""

    N_TRACKS = 4165
    DENSE_DIM = 23
    WIDTH = 256

    def __init__(
        self,
        input_dim: int,
        config: ModelConfig,
        *,
        final_regressor_dropout: float = 0.15,
        use_mean_proxy: bool = True,
    ):
        super().__init__()
        enc = config.encoder
        if enc.kind != "locus_attention" or enc.program_dim != self.WIDTH:
            raise ValueError(
                "SingleRetrievalPredictor requires locus_attention RNA tokens "
                "with program_dim=256"
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
            self.N_TRACKS,
            self.WIDTH,
            mode="mean",
            include_last_offset=True,
        )
        self.dense_encoder = nn.Sequential(
            nn.LayerNorm(self.DENSE_DIM),
            nn.Linear(self.DENSE_DIM, self.WIDTH),
            nn.GELU(),
        )
        self.locus_norm = nn.LayerNorm(self.WIDTH)
        self.cross_attention = LocusToRNAAttention(
            self.WIDTH, enc.n_heads, enc.dropout
        )
        # Build the proxy head unconditionally so disabling the branch
        # does not shift the RNG stream for any later shared parameter.
        mean_head = nn.Sequential(
            nn.LayerNorm(self.WIDTH),
            nn.Linear(self.WIDTH, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.mean_head = mean_head if use_mean_proxy else None
        self.h_c_concat_norm = nn.LayerNorm(self.WIDTH)
        self.r_pc_concat_norm = nn.LayerNorm(self.WIDTH)
        self.final_regressor_dropout = float(final_regressor_dropout)
        self.final_regressor = nn.Sequential(
            nn.Linear(2 * self.WIDTH, self.WIDTH),
            nn.GELU(),
            nn.Dropout(self.final_regressor_dropout),
            nn.Linear(self.WIDTH, 128),
            nn.GELU(),
            nn.Dropout(self.final_regressor_dropout),
            nn.Linear(128, 1),
        )

    @property
    def requires_cpg_positions(self) -> bool:
        return False

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

        peak = self.track_embedding(
            functional_track_indices, functional_offsets
        )
        dense = self.dense_encoder(functional_dense)
        h_c = self.locus_norm(peak + dense)
        mu_hat = (
            None
            if self.mean_head is None
            else torch.sigmoid(self.mean_head(h_c).squeeze(-1))
        )

        r_pc = self.cross_attention(h_c, tokens)
        h_for_concat = self.h_c_concat_norm(h_c)
        r_for_concat = self.r_pc_concat_norm(r_pc)
        z_pc = torch.cat(
            [
                h_for_concat[None, :, :].expand(r_for_concat.shape[0], -1, -1),
                r_for_concat,
            ],
            dim=-1,
        )
        logit = self.final_regressor(z_pc).squeeze(-1)
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


class IterativeRetrievalPredictor(_LegacyCandidateCheckpointMixin, nn.Module):
    """J1: four residual RNA-retrieval/refinement blocks."""

    N_TRACKS = 4165
    DENSE_DIM = 23
    WIDTH = 256
    N_BLOCKS = 4
    LOCUS_CHUNK = 1024

    def __init__(
        self,
        input_dim: int,
        config: ModelConfig,
        *,
        final_regressor_dropout: float = 0.15,
        use_mean_proxy: bool = True,
    ):
        super().__init__()
        enc = config.encoder
        if enc.kind != "locus_attention" or enc.program_dim != self.WIDTH:
            raise ValueError(
                "IterativeRetrievalPredictor requires locus_attention RNA "
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
            self.N_TRACKS,
            self.WIDTH,
            mode="mean",
            include_last_offset=True,
        )
        self.dense_encoder = nn.Sequential(
            nn.LayerNorm(self.DENSE_DIM),
            nn.Linear(self.DENSE_DIM, self.WIDTH),
            nn.GELU(),
        )
        self.locus_norm = nn.LayerNorm(self.WIDTH)
        mean_head = nn.Sequential(
            nn.LayerNorm(self.WIDTH),
            nn.Linear(self.WIDTH, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.mean_head = mean_head if use_mean_proxy else None
        self.retrieval_attn = nn.ModuleList(
            [
                BatchedLocusToRNAAttention(
                    self.WIDTH, enc.n_heads, enc.dropout
                )
                for _ in range(self.N_BLOCKS)
            ]
        )
        self.retrieval_ffn = nn.ModuleList(
            [
                FeedForwardResidual(self.WIDTH, enc.dropout)
                for _ in range(self.N_BLOCKS)
            ]
        )
        self.h_c_concat_norm = nn.LayerNorm(self.WIDTH)
        self.r_pc_concat_norm = nn.LayerNorm(self.WIDTH)
        self.final_regressor_dropout = float(final_regressor_dropout)
        self.final_regressor = nn.Sequential(
            nn.Linear(2 * self.WIDTH, self.WIDTH),
            nn.GELU(),
            nn.Dropout(self.final_regressor_dropout),
            nn.Linear(self.WIDTH, 128),
            nn.GELU(),
            nn.Dropout(self.final_regressor_dropout),
            nn.Linear(128, 1),
        )

    @property
    def requires_cpg_positions(self) -> bool:
        return False

    def _retrieve_and_predict(
        self,
        h_c_chunk: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch = tokens.shape[0]
        state = h_c_chunk[None, :, :].expand(batch, -1, -1)
        for attn, ffn in zip(self.retrieval_attn, self.retrieval_ffn):
            state = state + attn(state, tokens)
            state = ffn(state)

        h_for_concat = self.h_c_concat_norm(h_c_chunk)
        r_for_concat = self.r_pc_concat_norm(state)
        z_pc = torch.cat(
            [
                h_for_concat[None, :, :].expand(batch, -1, -1),
                r_for_concat,
            ],
            dim=-1,
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

        peak = self.track_embedding(
            functional_track_indices, functional_offsets
        )
        dense = self.dense_encoder(functional_dense)
        h_c = self.locus_norm(peak + dense)
        mu_hat = (
            None
            if self.mean_head is None
            else torch.sigmoid(self.mean_head(h_c).squeeze(-1))
        )

        n_loci = h_c.shape[0]
        if n_loci <= self.LOCUS_CHUNK:
            logit = self._retrieve_and_predict(h_c, tokens)
        else:
            chunks = []
            for start in range(0, n_loci, self.LOCUS_CHUNK):
                chunk = h_c[start : start + self.LOCUS_CHUNK]
                if self.training and torch.is_grad_enabled():
                    chunks.append(
                        checkpoint(
                            self._retrieve_and_predict,
                            chunk,
                            tokens,
                            use_reentrant=False,
                        )
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
