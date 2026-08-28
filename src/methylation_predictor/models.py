"""RNA-conditioned methylation models.

The canonical architecture is the variance-normalized residual model:

    RNA (25,017 genes)
      -> LayerNorm -> Linear(..., 256)
      -> RNA latent z

    [z, frozen NTv3 CpG embedding, proj(z) * proj(CpG)]
      -> MLP -> raw_delta

    beta_hat = sigmoid(logit(mu_i) + sigma_i * raw_delta)

The variability gate, mean-RNA anchor, direct-prediction branch and no-product
variants were research ablations.  They are intentionally not executable in
the canonical model. ``RNA2DNAmModel`` remains only as the historical flat-
residual compatibility baseline.
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
    """Canonical RNA-CpG interaction with an explicit multiplicative term.

    ``include_rna``/``include_cpg``/``include_product`` are architecture-
    simplification ablations: each independently controls whether that piece
    is part of the MLP's joint input ``[rna?, cpg?, product?]``. The product
    term (when included) is always computed from the *projected* rna/cpg,
    regardless of whether the raw rna/cpg pieces are also concatenated in --
    e.g. ``include_rna=False, include_cpg=True, include_product=True`` feeds
    the MLP only ``[cpg, proj(rna) * proj(cpg)]``. Canonical default is all
    three True. Not used by any canonical model path unless the config
    explicitly opts out of one (``model.interaction.include_*: false``).
    """

    def __init__(
        self,
        rna_dim: int,
        locus_dim: int,
        hidden_dim: int,
        dropout: float,
        include_rna: bool = True,
        include_cpg: bool = True,
        include_product: bool = True,
    ):
        super().__init__()
        self.include_rna = bool(include_rna)
        self.include_cpg = bool(include_cpg)
        self.include_product = bool(include_product)
        if not (self.include_rna or self.include_cpg or self.include_product):
            raise ValueError("ProductInteraction requires at least one of include_rna/include_cpg/include_product")
        product_dim = min(rna_dim, locus_dim)
        self.rna_product = nn.Linear(rna_dim, product_dim) if self.include_product else None
        self.locus_product = nn.Linear(locus_dim, product_dim) if self.include_product else None
        joint_dim = (
            (rna_dim if self.include_rna else 0)
            + (locus_dim if self.include_cpg else 0)
            + (product_dim if self.include_product else 0)
        )
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
        pieces = []
        if self.include_rna:
            pieces.append(rna.global_vector[:, None, :].expand(batch, n_loci, -1))
        if self.include_cpg:
            pieces.append(loci[None, :, :].expand(batch, n_loci, -1))
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


class RNA2DNAmModel(nn.Module):
    """Historical flat-residual compatibility baseline.

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


class VarianceNormalizedResidualModel(nn.Module):
    """canonical variance-normalized residual formulation: the residual is parametrized as
    ``sigma_i * raw_delta`` instead of a flat ``delta_logit``, so the network's
    raw output lives in a locus-variance-standardized space rather than
    directly in logit space.  ``sigma_i`` is the per-CpG inter-sample std of
    logit(beta) (exact for train CpGs, NTv3-probe-predicted for held-out/
    auxiliary CpGs -- see scripts/benchmark_methylprophet/prepare.py).

    Motivation (see docs/METHYLPROPHET_TABLE5.md V1 experiment note): under
    plain beta-MSE, a fixed absolute error contributes far more gradient on a
    high-variance locus than on a near-constant one, while the headline
    MAS-PCC metric weighs every locus equally regardless of its variance.
    Standardizing the residual target removes that scale mismatch.

    This is the current reference formulation. RNA2DNAmModel is retained
    only as a flat-residual ablation/control.  Reuses the same LinearRNAEncoder/ProductInteraction
    building blocks so results stay comparable to the frozen architecture.
    """

    def __init__(
        self,
        input_dim: int,
        locus_dim: int,
        config: ModelConfig,
        epsilon: float = 1e-4,
    ):
        super().__init__()
        if config.encoder.kind != "linear":
            raise ValueError("V1 variant requires model.encoder.kind='linear'")
        if config.interaction.kind != "concat":
            raise ValueError("V1 variant requires model.interaction.kind='concat'")
        if not config.zero_init_residual:
            raise ValueError("V1 variant requires zero_init_residual=true (starts exactly at the prior)")

        # 256 is the canonical/production RNA latent width. A different
        # config.encoder.latent_dim is an architecture-scaling ablation only
        # (does the model benefit from a wider RNA projection) -- not used by
        # any canonical model path unless the recipe explicitly opts in.
        latent_dim = config.encoder.latent_dim
        self.config = config
        self.epsilon = float(epsilon)
        self.rna_encoder = LinearRNAEncoder(
            input_dim=input_dim, latent_dim=latent_dim, layer_norm=config.encoder.layer_norm,
        )
        self.interaction = ProductInteraction(
            rna_dim=latent_dim, locus_dim=locus_dim,
            hidden_dim=config.interaction.hidden_dim, dropout=config.interaction.dropout,
            include_rna=config.interaction.include_rna,
            include_cpg=config.interaction.include_cpg,
            include_product=config.interaction.include_product,
        )
        self.interaction.zero_output()  # raw_delta = 0 at init => beta_hat = prior, same safe start as RNA2DNAmModel

    @property
    def supports_factorized_inference(self) -> bool:
        return False

    def forward(
        self,
        rna: torch.Tensor,
        loci: torch.Tensor,
        prior: torch.Tensor,
        sigma: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        representation = self.rna_encoder(rna)
        raw_delta = self.interaction(representation, loci)  # standardized-residual space
        sigma_safe = sigma.clamp_min(1e-6)  # numerical safety only, not the loss-side sigma_min floor
        delta_logit = sigma_safe.unsqueeze(0) * raw_delta
        prior = prior.clamp(self.epsilon, 1.0 - self.epsilon)
        prior_logit = torch.logit(prior)
        prediction_logit = prior_logit.unsqueeze(0) + delta_logit
        beta = torch.sigmoid(prediction_logit)

        gate = torch.ones(loci.shape[0], dtype=loci.dtype, device=loci.device)
        return {
            "beta": beta,
            "delta_logit": delta_logit,
            "raw_delta": raw_delta,
            "raw_delta_logit": delta_logit,
            "prediction_logit": prediction_logit,
            "gate": gate,
            "prior_logit": prior_logit,
        }


class RNAMethylationPredictor(VarianceNormalizedResidualModel):
    """Canonical public name for the frozen variance-normalized architecture.

    This subclass deliberately adds no parameters or buffers, so historical
    ``VarianceNormalizedResidualModel`` checkpoints load with identical state-
    dict keys.  The old class name remains a compatibility alias at call sites;
    new code should use ``RNAMethylationPredictor``.
    """
    pass
