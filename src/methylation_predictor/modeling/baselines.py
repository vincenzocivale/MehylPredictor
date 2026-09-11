"""Functional-matched paper baselines.

These baselines share J0's functional CpG encoder, RNA ProgramTokenEncoder,
training-only mean-proxy task, loss and data protocol. Only the RNA/locus
fusion mechanism changes.
"""
from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .retrieval import LocusToRNAAttention
from .rna import ProgramTokenEncoder, consume_legacy_attention_initialization


BASELINE_VARIANTS = {
    "functional_baseline_global_shift",
    "functional_baseline_mlp",
    "functional_baseline_bilinear",
}


class FunctionalBaselinePredictor(nn.Module):
    N_TRACKS = 4165
    DENSE_DIM = 23
    WIDTH = 256

    def __init__(
        self,
        input_dim: int,
        config: ModelConfig,
        *,
        variant: str,
        final_regressor_dropout: float = 0.15,
        use_mean_proxy: bool = True,
    ):
        super().__init__()
        if variant not in BASELINE_VARIANTS:
            raise ValueError(
                f"unknown functional baseline variant {variant!r}; "
                f"known: {sorted(BASELINE_VARIANTS)}"
            )
        enc = config.encoder
        if enc.kind != "locus_attention" or enc.program_dim != self.WIDTH:
            raise ValueError(
                "functional baselines require J0's locus_attention token encoder "
                "with program_dim=256"
            )

        self.variant = variant
        self.rna_encoder = ProgramTokenEncoder(
            input_dim=input_dim,
            n_programs=enc.n_programs,
            program_dim=enc.program_dim,
            bottleneck_dim=enc.latent_dim,
            layer_norm=enc.layer_norm,
        )
        consume_legacy_attention_initialization(self.WIDTH)

        self.track_embedding = nn.EmbeddingBag(
            self.N_TRACKS, self.WIDTH, mode="mean", include_last_offset=True
        )
        self.dense_encoder = nn.Sequential(
            nn.LayerNorm(self.DENSE_DIM),
            nn.Linear(self.DENSE_DIM, self.WIDTH),
            nn.GELU(),
        )
        self.locus_norm = nn.LayerNorm(self.WIDTH)

        # J0 builds its real retrieval attention before mean_head. Burn the
        # same initialization draws so all shared parameters through mean_head
        # are bit-identical under the same seed.
        LocusToRNAAttention(self.WIDTH, enc.n_heads, enc.dropout)

        mean_head = nn.Sequential(
            nn.LayerNorm(self.WIDTH),
            nn.Linear(self.WIDTH, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.mean_head = mean_head if use_mean_proxy else None

        self.rna_pool_norm = nn.LayerNorm(self.WIDTH)
        self.locus_head_norm = nn.LayerNorm(self.WIDTH)
        self.final_regressor_dropout = float(final_regressor_dropout)

        if variant == "functional_baseline_global_shift":
            self.locus_head = nn.Sequential(
                nn.Linear(self.WIDTH, 128),
                nn.GELU(),
                nn.Dropout(self.final_regressor_dropout),
                nn.Linear(128, 1),
            )
            self.rna_shift_head = nn.Sequential(
                nn.Linear(self.WIDTH, 128),
                nn.GELU(),
                nn.Dropout(self.final_regressor_dropout),
                nn.Linear(128, 1),
            )
            self.rna_product = None
            self.locus_product = None
            self.final_regressor = None
        else:
            self.locus_head = None
            self.rna_shift_head = None
            if variant == "functional_baseline_bilinear":
                self.rna_product = nn.Linear(self.WIDTH, self.WIDTH)
                self.locus_product = nn.Linear(self.WIDTH, self.WIDTH)
            else:
                self.rna_product = None
                self.locus_product = None
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

    def _functional_locus(
        self,
        functional_track_indices,
        functional_offsets,
        functional_dense,
    ):
        peak = self.track_embedding(functional_track_indices, functional_offsets)
        dense = self.dense_encoder(functional_dense)
        return self.locus_norm(peak + dense)

    def forward(
        self,
        rna,
        cpg_embedding=None,
        cpg_positions=None,
        *,
        functional_track_indices,
        functional_offsets,
        functional_dense,
    ):
        del cpg_embedding, cpg_positions

        h_c = self._functional_locus(
            functional_track_indices,
            functional_offsets,
            functional_dense,
        )
        mu_hat = (
            None
            if self.mean_head is None
            else torch.sigmoid(self.mean_head(h_c).squeeze(-1))
        )

        tokens = self.rna_encoder(rna)
        g_p = self.rna_pool_norm(tokens.mean(dim=1))
        h = self.locus_head_norm(h_c)
        batch, n_loci = rna.shape[0], h_c.shape[0]

        if self.variant == "functional_baseline_global_shift":
            locus_logit = self.locus_head(h).squeeze(-1)
            patient_shift = self.rna_shift_head(g_p).squeeze(-1)
            logit = locus_logit[None, :] + patient_shift[:, None]
        elif self.variant == "functional_baseline_mlp":
            z = torch.cat(
                [
                    h[None].expand(batch, n_loci, -1),
                    g_p[:, None].expand(batch, n_loci, -1),
                ],
                dim=-1,
            )
            logit = self.final_regressor(z).squeeze(-1)
        else:
            interaction = (
                self.locus_product(h)[None]
                * self.rna_product(g_p)[:, None]
            )
            z = torch.cat(
                [h[None].expand(batch, n_loci, -1), interaction],
                dim=-1,
            )
            logit = self.final_regressor(z).squeeze(-1)

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
