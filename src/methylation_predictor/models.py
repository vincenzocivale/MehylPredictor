"""Canonical RNA-to-DNAm model.

This module intentionally exposes one architecture only:
Linear RNA encoder -> concat/product RNA-CpG interaction -> variability gate ->
residual correction of the frozen CpG prior in logit space.
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
    """LayerNorm followed by the canonical 64-D linear RNA projection."""

    def __init__(self, input_dim: int, latent_dim: int, layer_norm: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim) if layer_norm else nn.Identity()
        self.projection = nn.Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> RNARepresentation:
        return RNARepresentation(self.projection(self.norm(x)))


class ConcatInteraction(nn.Module):
    """RNA-CpG interaction: concatenation of RNA, CpG embedding and projected
    element-wise product, fed through an MLP residual head."""

    def __init__(
        self,
        rna_dim: int,
        locus_dim: int,
        hidden_dim: int,
        dropout: float,
        include_product: bool = True,
    ):
        super().__init__()
        self.include_product = bool(include_product)
        product_dim = min(rna_dim, locus_dim)
        joint_dim = rna_dim + locus_dim + (product_dim if self.include_product else 0)

        # Keep these names stable for checkpoint compatibility.
        self.rna_product = nn.Linear(rna_dim, product_dim) if self.include_product else None
        self.locus_product = nn.Linear(locus_dim, product_dim) if self.include_product else None
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
        pieces = [rna_expanded, loci_expanded]
        if self.include_product:
            assert self.rna_product is not None and self.locus_product is not None
            product = (
                self.rna_product(rna.global_vector)[:, None, :]
                * self.locus_product(loci)[None, :, :]
            )
            pieces.append(product)
        joint = torch.cat(pieces, dim=-1)
        return self.network(joint).squeeze(-1)

    def zero_output(self) -> None:
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)


class NoResidualGate(nn.Module):
    """Identity residual gate used only by explicit architecture ablations.

    The module intentionally has no parameters. ``forward`` keeps the same
    two-argument interface as ``ResidualGate`` so the rest of the training and
    evaluation stack does not need an ablation-specific code path.
    """

    kind = "none"

    def forward(self, loci: torch.Tensor, variability: torch.Tensor) -> torch.Tensor:
        del variability
        return torch.ones(loci.shape[0], dtype=loci.dtype, device=loci.device)


class ResidualGate(nn.Module):
    """Canonical locus-specific gate using embedding + two variability features."""

    def __init__(self, locus_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.kind = "variability"
        self.network = nn.Sequential(
            nn.LayerNorm(locus_dim + 2),
            nn.Linear(locus_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, loci: torch.Tensor, variability: torch.Tensor) -> torch.Tensor:
        if variability.ndim != 2 or variability.shape[-1] != 2:
            raise ValueError(
                "canonical variability gate expects exactly two variability features "
                f"per locus; received shape={tuple(variability.shape)}"
            )
        features = torch.cat([loci, variability], dim=-1)
        return torch.sigmoid(self.network(features)).squeeze(-1)


class RNA2DNAmModel(nn.Module):
    """Canonical frozen-prior RNA-to-DNAm model.

    logit(beta_hat[s,i]) =
        logit(prior[i]) +
        gate[i] * (
            interaction(RNA[s], CpG[i])
            - interaction(mean_RNA, CpG[i])
        )
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
                "pretrained/gene-token RNA inputs were removed by the training-only refactor"
            )
        if config.encoder.kind != "linear":
            raise ValueError(
                f"only model.encoder.kind='linear' is supported; got {config.encoder.kind!r}"
            )
        if config.interaction.kind != "concat":
            raise ValueError(
                f"only model.interaction.kind='concat' is supported; got {config.interaction.kind!r}"
            )
        if config.gate.kind not in {"variability", "none"}:
            raise ValueError(
                "model.gate.kind must be 'variability' or the explicit ablation "
                f"'none'; got {config.gate.kind!r}"
            )
        if config.prediction_mode not in {"residual_prior", "direct"}:
            raise ValueError(
                "model.prediction_mode must be 'residual_prior' or 'direct'; "
                f"got {config.prediction_mode!r}"
            )
        if config.prediction_mode == "direct" and config.anchor_to_mean_rna:
            raise ValueError(
                "direct prediction cannot use mean-RNA anchoring: without the prior, "
                "anchoring would force the reference RNA profile to beta=0.5. Set "
                "model.anchor_to_mean_rna=false for the direct-prediction ablation."
            )

        self.config = config
        self.epsilon = float(epsilon)
        self.rna_encoder = LinearRNAEncoder(
            input_dim=input_dim,
            latent_dim=config.encoder.latent_dim,
            layer_norm=config.encoder.layer_norm,
        )
        self.interaction = ConcatInteraction(
            rna_dim=config.encoder.latent_dim,
            locus_dim=locus_dim,
            hidden_dim=config.interaction.hidden_dim,
            dropout=config.interaction.dropout,
            include_product=config.interaction.include_product,
        )
        if config.gate.kind == "variability":
            self.gate = ResidualGate(
                locus_dim=locus_dim,
                hidden_dim=config.gate.hidden_dim,
                dropout=config.gate.dropout,
            )
        else:
            self.gate = NoResidualGate()

        if config.zero_init_residual:
            self.interaction.zero_output()

    @property
    def supports_factorized_inference(self) -> bool:
        # Factorized inference belonged to the removed bilinear branch.
        return False

    def forward(
        self,
        rna: torch.Tensor,
        loci: torch.Tensor,
        prior: torch.Tensor,
        variability: torch.Tensor,
        reference_rna: torch.Tensor | None = None,
        cancer_codes: torch.Tensor | None = None,
        cancer_centroids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del cancer_codes, cancer_centroids

        representation = self.rna_encoder(rna)
        raw = self.interaction(representation, loci)

        if self.config.anchor_to_mean_rna:
            if reference_rna is None:
                reference_rna = torch.zeros(
                    (1, rna.shape[1]), dtype=rna.dtype, device=rna.device
                )
            if reference_rna.shape[0] != 1:
                raise ValueError("reference_rna must contain exactly one reference profile")
            reference = self.rna_encoder(reference_rna)
            reference_raw = self.interaction(reference, loci)
            raw = raw - reference_raw.expand_as(raw)

        gate = self.gate(loci, variability)
        # In the explicit no-gate ablation the residual is used directly.
        # Avoid multiplying by an all-ones tensor so AMP dtype/rounding is not
        # changed by a type-promotion side effect.
        model_logit = raw if self.config.gate.kind == "none" else raw * gate.unsqueeze(0)
        prior = prior.clamp(self.epsilon, 1.0 - self.epsilon)
        prior_logit = torch.logit(prior)

        if self.config.prediction_mode == "residual_prior":
            delta_logit = model_logit
            prediction_logit = prior_logit.unsqueeze(0) + delta_logit
        else:
            # Direct ablation: the interaction branch learns the complete
            # methylation logit. delta_logit is still exposed relative to the
            # prior for diagnostics and residual-Huber bookkeeping; the prior
            # cancels algebraically from that Huber target.
            prediction_logit = model_logit
            delta_logit = prediction_logit - prior_logit.unsqueeze(0)

        beta = torch.sigmoid(prediction_logit)

        return {
            "beta": beta,
            "delta_logit": delta_logit,
            "raw_delta_logit": raw,
            "prediction_logit": prediction_logit,
            "gate": gate,
            "prior_logit": prior_logit,
        }
