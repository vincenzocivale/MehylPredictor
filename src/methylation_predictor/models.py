"""Canonical RNA-to-DNAm model selected by the architecture study.

The production architecture is deliberately singular:

    RNA (25,017 genes)
      -> LayerNorm -> Linear(..., 256)
      -> RNA latent z

    [z, frozen NTv3 CpG embedding, proj(z) * proj(CpG)]
      -> MLP -> delta_logit

    beta_hat = sigmoid(logit(frozen_prior) + delta_logit)

The variability gate, mean-RNA anchor, direct-prediction branch and no-product
variants were research ablations.  They are intentionally not executable in
the production model after the confirming Array-chr1 experiment selected the
simpler RNA256 residual architecture.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import ModelConfig


@dataclass
class RNARepresentation:
    global_vector: torch.Tensor


class LinearRNAEncoder(nn.Module):
    """LayerNorm followed by the canonical 256-D linear RNA projection."""

    def __init__(self, input_dim: int, latent_dim: int, layer_norm: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim) if layer_norm else nn.Identity()
        self.projection = nn.Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> RNARepresentation:
        return RNARepresentation(self.projection(self.norm(x)))


class ProductInteraction(nn.Module):
    """Canonical RNA-CpG interaction with an explicit multiplicative term."""

    def __init__(
        self,
        rna_dim: int,
        locus_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        product_dim = min(rna_dim, locus_dim)
        self.rna_product = nn.Linear(rna_dim, product_dim)
        self.locus_product = nn.Linear(locus_dim, product_dim)
        joint_dim = rna_dim + locus_dim + product_dim
        self.network = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Linear(joint_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, rna: RNARepresentation, loci: torch.Tensor) -> torch.Tensor:
        batch = rna.global_vector.shape[0]
        n_loci = loci.shape[0]
        rna_expanded = rna.global_vector[:, None, :].expand(batch, n_loci, -1)
        loci_expanded = loci[None, :, :].expand(batch, n_loci, -1)
        product = (
            self.rna_product(rna.global_vector)[:, None, :]
            * self.locus_product(loci)[None, :, :]
        )
        joint = torch.cat([rna_expanded, loci_expanded, product], dim=-1)
        return self.network(joint).squeeze(-1)

    def zero_output(self) -> None:
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)


class RNA2DNAmModel(nn.Module):
    """Final frozen-prior RNA-to-DNAm architecture.

    ``variability``/``reference_rna``/cancer arguments remain optional only to
    keep the existing evaluation stack source-compatible.  They are ignored:
    the selected production architecture has neither a variability gate nor a
    mean-RNA anchor.
    """

    def __init__(
        self,
        input_dim: int,
        locus_dim: int,
        config: ModelConfig,
        epsilon: float = 1e-4,
        gene_embeddings: torch.Tensor | None = None,
        raw_rna_dim: int | None = None,
    ):
        super().__init__()
        if gene_embeddings is not None or raw_rna_dim is not None:
            raise ValueError(
                "pretrained/gene-token RNA inputs are not part of the canonical model"
            )
        if config.encoder.kind != "linear":
            raise ValueError(
                f"canonical model requires model.encoder.kind='linear'; got {config.encoder.kind!r}"
            )
        if config.encoder.latent_dim != 256:
            raise ValueError(
                "canonical model requires a 256-D RNA latent; "
                f"got {config.encoder.latent_dim}"
            )
        if config.interaction.kind != "concat":
            raise ValueError(
                f"canonical model requires model.interaction.kind='concat'; got {config.interaction.kind!r}"
            )
        if not config.zero_init_residual:
            raise ValueError("canonical model requires zero_init_residual=true")

        self.config = config
        self.epsilon = float(epsilon)
        self.rna_encoder = LinearRNAEncoder(
            input_dim=input_dim,
            latent_dim=256,
            layer_norm=config.encoder.layer_norm,
        )
        self.interaction = ProductInteraction(
            rna_dim=256,
            locus_dim=locus_dim,
            hidden_dim=config.interaction.hidden_dim,
            dropout=config.interaction.dropout,
        )
        self.interaction.zero_output()

    @property
    def supports_factorized_inference(self) -> bool:
        return False

    def forward(
        self,
        rna: torch.Tensor,
        loci: torch.Tensor,
        prior: torch.Tensor,
        variability: torch.Tensor | None = None,
        reference_rna: torch.Tensor | None = None,
        cancer_codes: torch.Tensor | None = None,
        cancer_centroids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del variability, reference_rna, cancer_codes, cancer_centroids

        representation = self.rna_encoder(rna)
        delta_logit = self.interaction(representation, loci)
        prior = prior.clamp(self.epsilon, 1.0 - self.epsilon)
        prior_logit = torch.logit(prior)
        prediction_logit = prior_logit.unsqueeze(0) + delta_logit
        beta = torch.sigmoid(prediction_logit)

        # ``gate`` is a diagnostic compatibility field, not a model component.
        # Returning ones lets historical metric/evaluation code consume the
        # canonical output dictionary without carrying a gate implementation.
        gate = torch.ones(loci.shape[0], dtype=loci.dtype, device=loci.device)
        return {
            "beta": beta,
            "delta_logit": delta_logit,
            "raw_delta_logit": delta_logit,
            "prediction_logit": prediction_logit,
            "gate": gate,
            "prior_logit": prior_logit,
        }
