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
import torch.utils.checkpoint  # noqa: F401 -- trunk gradient checkpointing

from .config import AxialConfig, EncoderConfig, InteractionConfig, ModelConfig, TrunkConfig


@dataclass
class RNARepresentation:
    global_vector: torch.Tensor
    # Architecture-novelty ablation only: a (batch, n_programs, program_dim) set of
    # gene-program tokens, emitted by ``LocusConditionedRNAEncoder`` so a CpG can
    # query the transcriptome instead of receiving one locus-invariant vector.
    # None for every canonical encoder, which keeps positional construction
    # ``RNARepresentation(vec)`` valid at all existing call sites.
    program_tokens: torch.Tensor | None = None


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


class FiLMInteraction(nn.Module):
    """Fusion-mechanism ablation: RNA modulates the CpG embedding via FiLM.

    RNA latent -> Linear -> (gamma, beta) of ``locus_dim``; the CpG embedding
    is affinely modulated (``loci' = loci * (1 + gamma) + beta``) before the
    joint MLP. Not used by any canonical model path -- see
    ``InteractionConfig.kind`` / docs/RNA_METHYLATION.md ablation note.
    """

    def __init__(self, rna_dim: int, locus_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.film = nn.Linear(rna_dim, 2 * locus_dim)
        joint_dim = rna_dim + locus_dim
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
        gamma, beta = self.film(rna.global_vector).chunk(2, dim=-1)  # each (batch, locus_dim)
        modulated = loci[None, :, :] * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        joint = torch.cat(
            [rna.global_vector[:, None, :].expand(batch, n_loci, -1), modulated], dim=-1
        )
        return self.network(joint).squeeze(-1)

    def zero_output(self) -> None:
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)


class CrossAttentionInteraction(nn.Module):
    """Fusion-mechanism ablation: single-head scaled dot-product cross attention.

    query = W_q(rna), key/value = W_k(loci)/W_v(loci); the (sample, locus)
    attention score is the scaled dot product of query and key (same pairwise
    shape as ``ProductInteraction``'s product term). Unlike sequence
    attention there is no token axis to pool over -- each locus is scored
    independently per sample within the Cartesian minibatch -- so the score
    gates the value via a sigmoid rather than a softmax. Not used by any
    canonical model path -- see ``InteractionConfig.kind`` /
    docs/RNA_METHYLATION.md ablation note.
    """

    def __init__(
        self,
        rna_dim: int,
        locus_dim: int,
        hidden_dim: int,
        dropout: float,
        attn_dim: int | None = None,
        n_heads: int = 1,
    ):
        super().__init__()
        attn_dim = attn_dim or min(rna_dim, locus_dim)
        if attn_dim % n_heads != 0:
            raise ValueError(f"attn_dim {attn_dim} must be divisible by attn_heads {n_heads}")
        self.attn_dim = attn_dim
        # n_heads > 1 splits the gate: each head scores its own attn_dim/n_heads
        # slice, so the sample-locus pair gets ``n_heads`` independent gates
        # instead of one scalar. The 2026-08 arm ran n_heads=1; this is the
        # capacity axis that arm never had, not a different mechanism.
        self.n_heads = int(n_heads)
        self.head_dim = attn_dim // self.n_heads
        self.query = nn.Linear(rna_dim, attn_dim)
        self.key = nn.Linear(locus_dim, attn_dim)
        self.value = nn.Linear(locus_dim, attn_dim)
        joint_dim = rna_dim + locus_dim + attn_dim
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
        q = self.query(rna.global_vector)  # (batch, attn_dim)
        k = self.key(loci)  # (n_loci, attn_dim)
        v = self.value(loci)  # (n_loci, attn_dim)
        if self.n_heads == 1:
            score = (q[:, None, :] * k[None, :, :]).sum(-1) / (self.attn_dim ** 0.5)  # (batch, n_loci)
            gate = torch.sigmoid(score)
            attended = gate[:, :, None] * v[None, :, :]  # (batch, n_loci, attn_dim)
        else:
            qh = q.view(batch, self.n_heads, self.head_dim)
            kh = k.view(n_loci, self.n_heads, self.head_dim)
            score = torch.einsum("bhd,lhd->bhl", qh, kh) / (self.head_dim ** 0.5)
            gate = torch.sigmoid(score)  # (batch, n_heads, n_loci)
            vh = v.view(n_loci, self.n_heads, self.head_dim)
            attended = (gate.permute(0, 2, 1)[..., None] * vh[None]).reshape(batch, n_loci, self.attn_dim)
        joint = torch.cat(
            [
                rna.global_vector[:, None, :].expand(batch, n_loci, -1),
                loci[None, :, :].expand(batch, n_loci, -1),
                attended,
            ],
            dim=-1,
        )
        return self.network(joint).squeeze(-1)

    def zero_output(self) -> None:
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)


class BilinearInteraction(nn.Module):
    """Fusion-mechanism ablation: learned low-rank bilinear RNA-CpG interaction.

    Generalizes ``ProductInteraction``'s elementwise product term (rank ==
    ``min(rna_dim, locus_dim)``, one shared projection pair) to an
    independent rank-``rank`` low-rank bilinear form: separate ``rna``/
    ``locus`` projections to ``rank`` dims, combined elementwise. This is the
    standard memory-efficient low-rank bilinear pooling factorization; a
    literal ``nn.Bilinear(rna_dim, locus_dim, rank)`` was tried first but its
    dense per-pair weight tensor OOMs on the large WGBS Cartesian blocks
    (batch*n_loci ~ 5*10^5) used here. Not used by any canonical model path --
    see ``InteractionConfig.kind`` / docs/RNA_METHYLATION.md ablation note.

    ``include_rna``/``include_cpg`` mirror ``ProductInteraction``'s flags: with
    both False (the "Bilinear RNA-CpG" baseline, see docs/PAPER_EXPERIMENTS.md),
    the joint MLP input is only the bilinear term itself -- a purer "shared
    latent space, interaction via dot product" baseline than the
    ``fusion_mechanism_2026_08`` ablation's hybrid (concat + bilinear) form.
    """

    def __init__(
        self,
        rna_dim: int,
        locus_dim: int,
        hidden_dim: int,
        dropout: float,
        rank: int = 64,
        include_rna: bool = True,
        include_cpg: bool = True,
    ):
        super().__init__()
        self.rank = rank
        self.include_rna = bool(include_rna)
        self.include_cpg = bool(include_cpg)
        self.rna_bilinear = nn.Linear(rna_dim, rank)
        self.locus_bilinear = nn.Linear(locus_dim, rank)
        joint_dim = (rna_dim if self.include_rna else 0) + (locus_dim if self.include_cpg else 0) + rank
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
        bilinear_features = (
            self.rna_bilinear(rna.global_vector)[:, None, :] * self.locus_bilinear(loci)[None, :, :]
        )  # (batch, n_loci, rank)
        pieces = []
        if self.include_rna:
            pieces.append(rna.global_vector[:, None, :].expand(batch, n_loci, -1))
        if self.include_cpg:
            pieces.append(loci[None, :, :].expand(batch, n_loci, -1))
        pieces.append(bilinear_features)
        joint = torch.cat(pieces, dim=-1)
        return self.network(joint).squeeze(-1)

    def zero_output(self) -> None:
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)


class GlobalShiftInteraction(nn.Module):
    """"Global RNA Shift" baseline: a single patient-specific correction.

    Ignores the CpG embedding entirely -- the joint MLP input is the RNA
    latent alone, so ``raw_delta`` is constant across every locus for a given
    sample. Combined with the canonical prior anchor
    (``logit(beta_hat) = logit(mu_i) + sigma_i * raw_delta``), this tests
    whether a single global per-patient shift is sufficient without any
    CpG-specific modeling of the RNA effect. See docs/PAPER_EXPERIMENTS.md.
    """

    def __init__(self, rna_dim: int, locus_dim: int, hidden_dim: int, dropout: float):
        del locus_dim
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(rna_dim),
            nn.Linear(rna_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, rna: RNARepresentation, loci: torch.Tensor) -> torch.Tensor:
        batch = rna.global_vector.shape[0]
        n_loci = loci.shape[0]
        shift = self.network(rna.global_vector).squeeze(-1)  # (batch,)
        return shift[:, None].expand(batch, n_loci)

    def zero_output(self) -> None:
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)


# InteractionConfig.kind -> interaction module constructor. "concat" is the
# canonical ProductInteraction (its own include_rna/include_cpg/include_product
# flags cover the concat/product-only ablation axis); the remaining kinds are
# fusion-mechanism ablations -- see docs/RNA_METHYLATION.md.
def build_interaction(config: "InteractionConfig", rna_dim: int, locus_dim: int) -> nn.Module:
    if config.kind == "concat":
        return ProductInteraction(
            rna_dim=rna_dim, locus_dim=locus_dim, hidden_dim=config.hidden_dim, dropout=config.dropout,
            include_rna=config.include_rna, include_cpg=config.include_cpg, include_product=config.include_product,
        )
    if config.kind == "film":
        return FiLMInteraction(rna_dim=rna_dim, locus_dim=locus_dim, hidden_dim=config.hidden_dim, dropout=config.dropout)
    if config.kind == "cross_attention":
        return CrossAttentionInteraction(
            rna_dim=rna_dim, locus_dim=locus_dim, hidden_dim=config.hidden_dim, dropout=config.dropout,
            n_heads=config.attn_heads,
        )
    if config.kind == "bilinear":
        return BilinearInteraction(
            rna_dim=rna_dim, locus_dim=locus_dim, hidden_dim=config.hidden_dim, dropout=config.dropout,
            rank=config.rank, include_rna=config.include_rna, include_cpg=config.include_cpg,
        )
    if config.kind == "global_shift":
        return GlobalShiftInteraction(rna_dim=rna_dim, locus_dim=locus_dim, hidden_dim=config.hidden_dim, dropout=config.dropout)
    raise ValueError(f"unknown interaction.kind: {config.kind!r}")


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
        # config.interaction.kind selects the fusion mechanism: "concat" is the
        # canonical ProductInteraction; "film"/"cross_attention"/"bilinear" are
        # fusion-mechanism ablations -- see build_interaction/docs/RNA_METHYLATION.md.
        self.interaction = build_interaction(config.interaction, rna_dim=latent_dim, locus_dim=locus_dim)
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


class DirectPredictionModel(nn.Module):
    """Architecture-ablation model: no CpG-statistics prior/anchor at all.

    ``prediction_logit = raw_delta`` directly (no ``prior_logit`` added, no
    sigma scaling) -- measures how much the mu/sigma prior contributes versus
    predicting methylation from RNA+CpG embedding alone. Selected only via
    ``ModelConfig.use_prior_anchor=False``, matched_chr1 engine only (see
    ``benchmark/methylprophet/trainer.py``). Not part of the canonical model;
    see docs/RNA_METHYLATION.md ablation note.
    """

    def __init__(
        self,
        input_dim: int,
        locus_dim: int,
        config: ModelConfig,
        epsilon: float = 1e-4,
    ):
        super().__init__()
        if config.use_prior_anchor:
            raise ValueError("DirectPredictionModel requires model.use_prior_anchor=false")
        if config.zero_init_residual:
            raise ValueError(
                "DirectPredictionModel requires model.zero_init_residual=false "
                "(there is no anchor to start safely 'at zero' relative to)"
            )
        if config.encoder.kind != "linear":
            raise ValueError("DirectPredictionModel requires model.encoder.kind='linear'")

        self.config = config
        self.epsilon = float(epsilon)
        self.rna_encoder = LinearRNAEncoder(
            input_dim=input_dim, latent_dim=config.encoder.latent_dim, layer_norm=config.encoder.layer_norm,
        )
        self.interaction = build_interaction(
            config.interaction, rna_dim=config.encoder.latent_dim, locus_dim=locus_dim
        )

    @property
    def supports_factorized_inference(self) -> bool:
        return False

    def forward(
        self,
        rna: torch.Tensor,
        loci: torch.Tensor,
        prior: torch.Tensor,
        sigma: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del sigma  # unused: no prior/sigma anchor in this ablation
        representation = self.rna_encoder(rna)
        raw_delta = self.interaction(representation, loci)
        prediction_logit = raw_delta
        beta = torch.sigmoid(prediction_logit)

        gate = torch.ones(loci.shape[0], dtype=loci.dtype, device=loci.device)
        return {
            "beta": beta,
            "delta_logit": prediction_logit,
            "raw_delta": raw_delta,
            "raw_delta_logit": prediction_logit,
            "prediction_logit": prediction_logit,
            "gate": gate,
            "prior_logit": None,
        }


class CpGTrunk(nn.Module):
    """Shared NTv3 -> locus representation ("locus CLS"): the part of
    ``FeatureFusionLocusCLSModel`` trained on the mean-prediction proxy task
    and then reused (not the raw NTv3 embedding) by the RNA-conditioned head.
    Mirrors the first
    layers of ``cpg_statistics.model._ScalarHead`` -- same shape/defaults, kept
    as a separate small module here rather than imported cross-package (see
    CLAUDE.md: the two model families stay architecturally decoupled on
    purpose)."""

    def __init__(self, input_dim: int = 1536, hidden_dim: int = 256, bottleneck_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureFusionLocusCLSModel(nn.Module):
    """The repo's primary/reference RNA-methylation architecture as of
    2026-09-03 (see docs/RNA_METHYLATION.md) -- selected via the
    ``shared_backbone_locus_cls_2026_09`` ablation ladder
    (``results/reference/ablations.yaml``) over the earlier two-stage
    frozen-prior + residual pipeline (``VarianceNormalizedResidualModel``/
    ``RNAMethylationPredictor``, kept below, frozen, for old-checkpoint
    compatibility only -- see CLAUDE.md's "Model compatibility note"). Trained
    via ``rna_training.locus_cls_trainer.LocusCLSJointTrainer``, wired into
    ``scripts/train.py --engine matched_chr1_shared_backbone``.

    Originated from a postdoc discussion (2026-09-01: use the mean-prediction
    task to learn *features* useful for the full prediction, single final
    head, no old-checkpoint warm start). The ladder's winning configuration
    (mean branch on, concatenation-only fusion, default init, no residual
    probe, no LR multiplier -- every other tested addition proved
    statistically indistinguishable from this simplest configuration, see the
    ablations.yaml entry) scored MAS-PCC 0.5627 on chr1's true official
    MethylProphet split, slightly above the two-stage model's 0.5613 (as a
    lower bound: that confirmation run was stopped at epoch 47/80, still
    improving).

    - "mean" branch (optional, ``use_mean_branch``): ``CpGTrunk`` on the NTv3
      embedding -> ``h_mean`` (locus-only features). A small auxiliary linear
      probe (``mean_head``) is trained against the mean-prediction proxy task
      purely to shape ``h_mean`` -- its scalar output (``mu_logit``) never
      enters the beta computation, only the trunk's *features* do (via
      concatenation into fusion).
    - "raw" branch (always present): the same ``[rna, cpg, proj(rna)*proj(cpg)]``
      joint input ``ProductInteraction`` uses, but stopped at its hidden layer
      instead of collapsing to a scalar, so its learned features (``h_raw``)
      can be reused downstream. A second auxiliary probe (``residual_head``),
      symmetric to ``mean_head``, is trained against the sample-locus residual
      (``logit(target_beta) - logit(target_mu)``, Huber not MSE -- WGBS's
      near-0/1 per-patient values make a plain MSE target blow up) so this
      branch gets its own direct gradient instead of only an indirect one
      through the fusion layer (a measured branch imbalance, chr1
      pair_complete, 2026-09-02).
    - fusion: the model's only prediction head -- a linear layer over
      ``[h_mean, h_raw]`` (``use_fusion_product`` also concatenates
      ``proj(h_mean)*proj(h_raw)``, mirroring the raw branch's own
      concat+product pattern) -> ``beta_logit`` -> sigmoid. No mu+residual
      sum, no second head exposed at inference.

    Trained end-to-end in one phase from random initialization (see
    ``rna_training.locus_cls_trainer.LocusCLSJointTrainer``) -- no separate
    pretraining stage, no warm start from another checkpoint."""

    def __init__(
        self,
        input_dim: int,
        cpg_input_dim: int,
        config: ModelConfig,
        trunk_hidden_dim: int = 256,
        bottleneck_dim: int = 64,
        trunk_dropout: float = 0.1,
        use_mean_branch: bool = True,
        use_fusion_product: bool = False,
        fusion_init_std: float = 0.01,
    ):
        super().__init__()
        if config.encoder.kind != "linear":
            raise ValueError("FeatureFusionLocusCLSModel requires model.encoder.kind='linear'")
        self.use_mean_branch = bool(use_mean_branch)
        self.use_fusion_product = bool(use_fusion_product)

        if self.use_mean_branch:
            self.trunk = CpGTrunk(cpg_input_dim, trunk_hidden_dim, bottleneck_dim, trunk_dropout)
            self.mean_head = nn.Linear(bottleneck_dim, 1)

        rna_dim = config.encoder.latent_dim
        raw_hidden_dim = config.interaction.hidden_dim
        dropout = config.interaction.dropout
        self.rna_encoder = LinearRNAEncoder(
            input_dim=input_dim, latent_dim=rna_dim, layer_norm=config.encoder.layer_norm,
        )
        product_dim = min(rna_dim, cpg_input_dim)
        self.rna_product = nn.Linear(rna_dim, product_dim)
        self.locus_product = nn.Linear(cpg_input_dim, product_dim)
        joint_dim = rna_dim + cpg_input_dim + product_dim
        self.raw_branch = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Linear(joint_dim, raw_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_head = nn.Linear(raw_hidden_dim, 1)

        fusion_dim = raw_hidden_dim + (bottleneck_dim if self.use_mean_branch else 0)
        if self.use_fusion_product:
            if not self.use_mean_branch:
                raise ValueError("use_fusion_product requires use_mean_branch (nothing to multiply h_raw against)")
            self.fusion_product_dim = min(bottleneck_dim, raw_hidden_dim)
            self.mean_fusion_proj = nn.Linear(bottleneck_dim, self.fusion_product_dim)
            self.raw_fusion_proj = nn.Linear(raw_hidden_dim, self.fusion_product_dim)
            fusion_dim += self.fusion_product_dim

        self.fusion = nn.Linear(fusion_dim, 1)
        # Small-but-nonzero by default (fusion_init_std=0.01, not exactly zero
        # like every other interaction head's zero_output() convention): a
        # hard-zero fusion layer blocks ALL gradient to h_raw/raw_branch at
        # step 0 (mean_head still gets a strong direct auxiliary-loss gradient
        # into h_mean regardless of fusion's weights), letting the trunk race
        # ahead early while the raw/RNA branch stays essentially untrained
        # until fusion's weights grow away from zero -- measured empirically
        # (chr1 pair_complete ablation, 2026-09-02) as a real MAS-PCC plateau.
        # fusion_init_std=0.0 reproduces the hard-zero start, for ablating this
        # choice specifically.
        if fusion_init_std > 0:
            nn.init.normal_(self.fusion.weight, std=fusion_init_std)
        else:
            nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    def forward(self, rna: torch.Tensor, cpg_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        representation = self.rna_encoder(rna)
        rna_vec = representation.global_vector
        batch = rna_vec.shape[0]
        n_loci = cpg_embedding.shape[0]
        product = self.rna_product(rna_vec)[:, None, :] * self.locus_product(cpg_embedding)[None, :, :]
        joint = torch.cat(
            [
                rna_vec[:, None, :].expand(batch, n_loci, -1),
                cpg_embedding[None, :, :].expand(batch, n_loci, -1),
                product,
            ],
            dim=-1,
        )
        h_raw = self.raw_branch(joint)  # (batch, n_loci, raw_hidden_dim)
        residual_logit = self.residual_head(h_raw).squeeze(-1)  # (batch, n_loci) auxiliary only

        if self.use_mean_branch:
            h_mean = self.trunk(cpg_embedding)  # (n_loci, bottleneck)
            mu_logit = self.mean_head(h_mean).squeeze(-1)  # (n_loci,) auxiliary only
            h_mean_b = h_mean[None, :, :].expand(batch, n_loci, -1)
            pieces = [h_mean_b, h_raw]
            if self.use_fusion_product:
                # (n_loci, fpd)[None] broadcasts against (batch, n_loci, fpd) directly.
                fusion_product = self.mean_fusion_proj(h_mean)[None, :, :] * self.raw_fusion_proj(h_raw)
                pieces.append(fusion_product)
            fused = torch.cat(pieces, dim=-1)
        else:
            mu_logit = None
            fused = h_raw

        prediction_logit = self.fusion(fused).squeeze(-1)  # (batch, n_loci)
        beta = torch.sigmoid(prediction_logit)
        return {
            "beta": beta,
            "delta_logit": prediction_logit,
            "raw_delta": prediction_logit,
            "prediction_logit": prediction_logit,
            "mu_logit": mu_logit,
            "residual_logit": residual_logit,
            "h_cpg": h_mean if self.use_mean_branch else None,
        }


# ---------------------------------------------------------------------------
# Architecture-novelty ablation suite (architecture_novelty_2026_09).
#
# Everything below is opt-in and unreachable from the canonical model classes
# above: it is selected only when a recipe sets a non-default
# ``model.encoder.kind``, ``model.trunk`` or ``model.axial`` block, which routes
# construction to ``ArchitectureVariantModel``. The canonical
# ``RNAMethylationPredictor`` keeps its exact parameter set and state-dict keys
# (CLAUDE.md's model-compatibility note), so historical checkpoints still load.
#
# Motivation for the suite (postdoc review, 2026-09-03): the canonical model is
# a single Linear(25017 -> 256) RNA projection feeding a depth-1 fusion head,
# and every drop-in fusion swap has already been measured and lost
# (results/reference/ablations.yaml::fusion_mechanism_2026_08). The two axes
# that were never touched are the RNA encoder itself and the depth/topology of
# the trunk -- which is also the prerequisite for Hyper-Connections/mHC to have
# anything to act on.
# ---------------------------------------------------------------------------


class MLPRNAEncoder(nn.Module):
    """Capacity control for the RNA branch: one hidden layer instead of none.

    The canonical ``LinearRNAEncoder`` has no nonlinearity at all, and the only
    encoder ablation ever run was its *width*
    (``ablations.yaml::interaction_concat_and_latent_dim_2026_08``, saturating
    at +0.002). This arm exists so that any gain from the fancier encoders below
    can be attributed to their structure rather than to simply having more
    encoder capacity.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, dropout: float, layer_norm: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim) if layer_norm else nn.Identity()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> RNARepresentation:
        return RNARepresentation(self.network(self.norm(x)))


class ProgramBottleneckEncoder(nn.Module):
    """Gene-expression -> ``n_programs`` normalized "programs" -> RNA latent.

    A narrow, explicitly normalized intermediate layer: each program is a
    learned linear functional of the transcriptome, LayerNorm'd across the
    program axis so no single program can dominate the latent by scale alone.
    Compared with ``MLPRNAEncoder`` this trades hidden width for an interpretable
    intermediate representation (a program's loading vector is directly readable
    as a gene weighting), while remaining a single locus-invariant vector -- the
    controlled step before ``LocusConditionedRNAEncoder`` makes it locus-specific.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_programs: int,
        dropout: float,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim) if layer_norm else nn.Identity()
        self.programs = nn.Linear(input_dim, n_programs)
        self.program_norm = nn.LayerNorm(n_programs)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(n_programs, latent_dim)

    def forward(self, x: torch.Tensor) -> RNARepresentation:
        p = self.program_norm(torch.nn.functional.gelu(self.programs(self.norm(x))))
        return RNARepresentation(self.projection(self.dropout(p)))


class LocusConditionedRNAEncoder(nn.Module):
    """RNA as a *set of gene-program tokens* that each CpG queries.

    This targets what is arguably the canonical architecture's single biggest
    structural limitation: ``z_s`` is identical for every CpG of a sample, so all
    locus specificity has to be squeezed through the 256-dim product term. Here
    the transcriptome becomes ``n_programs`` tokens of ``program_dim``, and the
    per-pair RNA vector is produced by genuine multi-head cross attention with
    the frozen NTv3 CpG embedding as the query -- i.e. the RNA representation
    itself becomes locus-specific.

    This is *not* a re-run of the losing ``cross_attention`` fusion arm
    (``ablations.yaml::fusion_mechanism_2026_08``, 0.5329). That arm produced a
    single scalar score gated by a sigmoid, precisely because -- as its own
    docstring says -- there was "no token axis to pool over". The token axis is
    the thing this encoder adds, so a softmax is meaningful here; the earlier
    failure is the motivation for this arm, not a precedent against it.

    Side benefit: ``attention`` is returned per forward, so locus -> gene-program
    weights are directly plottable, complementing the Expected-Gradients
    attribution in scripts/explain.py.
    """

    def __init__(
        self,
        input_dim: int,
        locus_dim: int,
        n_programs: int,
        program_dim: int,
        n_heads: int,
        dropout: float,
        layer_norm: bool = True,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        if program_dim % n_heads != 0:
            raise ValueError(f"encoder.program_dim {program_dim} must be divisible by encoder.n_heads {n_heads}")
        self.n_programs = int(n_programs)
        self.program_dim = int(program_dim)
        self.n_heads = int(n_heads)
        self.head_dim = program_dim // n_heads
        self.norm = nn.LayerNorm(input_dim) if layer_norm else nn.Identity()
        # The token bank is factorized through a shared transcriptome bottleneck
        # rather than built by a single Linear(input_dim -> n_programs*program_dim).
        # That direct form costs input_dim * n_programs * program_dim parameters --
        # 410M at K=64, d=256, i.e. 57x the whole canonical model -- so any win
        # would have been raw capacity rather than the locus-conditioning
        # structure this arm is meant to test. Factorized, the cost is
        # input_dim*bottleneck + n_programs*program_dim*bottleneck, which keeps the
        # arm within ~1.6x of canonical and comparable to the ``mlp`` capacity control.
        self.bottleneck = nn.Linear(input_dim, bottleneck_dim)
        self.bottleneck_norm = nn.LayerNorm(bottleneck_dim)
        # Program k reads the shared bottleneck through its own basis, so its gene
        # loading stays readable as ``program_basis[k] @ bottleneck.weight``.
        self.program_basis = nn.Parameter(torch.empty(n_programs, program_dim, bottleneck_dim))
        nn.init.normal_(self.program_basis, std=bottleneck_dim ** -0.5)
        self.token_norm = nn.LayerNorm(program_dim)
        self.query = nn.Linear(locus_dim, program_dim)
        self.key = nn.Linear(program_dim, program_dim)
        self.value = nn.Linear(program_dim, program_dim)
        self.out = nn.Linear(program_dim, program_dim)
        self.dropout = nn.Dropout(dropout)
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> RNARepresentation:
        shared = self.bottleneck_norm(torch.nn.functional.gelu(self.bottleneck(self.norm(x))))
        tokens = self.token_norm(torch.einsum("br,kdr->bkd", shared, self.program_basis))
        # The "global" vector keeps a locus-invariant summary available (used by
        # the product term and by any arm that also concatenates raw RNA), so
        # this encoder is a strict superset of the canonical information.
        return RNARepresentation(tokens.mean(dim=1), program_tokens=tokens)

    def attend(self, tokens: torch.Tensor, loci: torch.Tensor, store_attention: bool = False) -> torch.Tensor:
        """(batch, K, program_dim) x (n_loci, locus_dim) -> (batch, n_loci, program_dim)."""
        batch, n_programs, _ = tokens.shape
        n_loci = loci.shape[0]
        q = self.query(loci).view(n_loci, self.n_heads, self.head_dim)
        k = self.key(tokens).view(batch, n_programs, self.n_heads, self.head_dim)
        v = self.value(tokens).view(batch, n_programs, self.n_heads, self.head_dim)
        scores = torch.einsum("lhd,bkhd->bhlk", q, k) / (self.head_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        if store_attention:
            # Detached: diagnostics only, never part of the backward graph.
            self.last_attention = weights.detach()
        weights = self.dropout(weights)
        attended = torch.einsum("bhlk,bkhd->blhd", weights, v).reshape(batch, n_loci, self.program_dim)
        return self.out(attended)


def build_rna_encoder(config: "EncoderConfig", input_dim: int, locus_dim: int) -> nn.Module:
    """``EncoderConfig.kind`` dispatch. ``linear`` is canonical; the rest are
    architecture-novelty ablations reachable only via ``ArchitectureVariantModel``."""
    if config.kind == "linear":
        return LinearRNAEncoder(input_dim=input_dim, latent_dim=config.latent_dim, layer_norm=config.layer_norm)
    if config.kind == "mlp":
        return MLPRNAEncoder(
            input_dim=input_dim, latent_dim=config.latent_dim, hidden_dim=config.hidden_dim,
            dropout=config.dropout, layer_norm=config.layer_norm,
        )
    if config.kind == "program_bottleneck":
        return ProgramBottleneckEncoder(
            input_dim=input_dim, latent_dim=config.latent_dim, n_programs=config.n_programs,
            dropout=config.dropout, layer_norm=config.layer_norm,
        )
    if config.kind == "locus_attention":
        return LocusConditionedRNAEncoder(
            input_dim=input_dim, locus_dim=locus_dim, n_programs=config.n_programs,
            program_dim=config.program_dim, n_heads=config.n_heads, dropout=config.dropout,
            layer_norm=config.layer_norm, bottleneck_dim=config.latent_dim,
        )
    raise ValueError(f"unknown encoder.kind: {config.kind!r}")


def sinkhorn_knopp(logits: torch.Tensor, iterations: int) -> torch.Tensor:
    """Project a square matrix onto the Birkhoff polytope (doubly stochastic).

    Exactly the operator mHC (arXiv:2512.24880, Eq. 9) uses: exponentiate to make
    every entry positive, then alternately renormalize rows and columns. Run in
    float32 regardless of autocast -- the alternating normalization is
    ill-conditioned in bfloat16 and the whole point of the constraint is
    numerical conservation.
    """
    matrix = torch.exp(logits.float())
    for _ in range(max(1, iterations)):
        matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        matrix = matrix / matrix.sum(dim=-2, keepdim=True).clamp_min(1e-12)
    return matrix


class _ResidualFunction(nn.Module):
    """Pre-norm feed-forward residual function ``F`` shared by every trunk kind."""

    def __init__(self, width: int, expansion: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * expansion, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PlainTrunk(nn.Module):
    """Depth control: ``depth`` pre-norm residual blocks, one residual stream.

    This is the arm that separates "depth helped" from "multi-stream mixing
    helped". Without it, any HC/mHC gain over the canonical depth-1 head is
    confounded with simply having added layers.
    """

    n_streams = 1

    def __init__(self, width: int, depth: int, expansion: int, dropout: float, gradient_checkpointing: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList(_ResidualFunction(width, expansion, dropout) for _ in range(depth))
        self.out_norm = nn.LayerNorm(width)
        self.gradient_checkpointing = bool(gradient_checkpointing)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = x + torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = x + block(x)
        return self.out_norm(x)

    def composite_gain(self) -> dict[str, float]:
        # A single unconstrained identity stream: the composite residual mapping
        # is exactly 1 by construction, reported for schema symmetry with HC/mHC.
        return {"forward_amax_gain": 1.0, "backward_amax_gain": 1.0}


class HyperConnectionTrunk(nn.Module):
    """Hyper-Connections (``kind="hc"``) and Manifold-Constrained HC (``kind="mhc"``).

    Implements the single-layer HC recursion of arXiv:2409.19606 / arXiv:2512.24880
    Eq. 3::

        x_{l+1} = H^res x_l + (H^post)^T F(H^pre x_l)

    over ``n_streams`` parallel residual streams. ``mhc`` additionally constrains
    the mappings onto the manifold of Eq. 6-8: ``H^res`` is projected onto the
    Birkhoff polytope by Sinkhorn-Knopp (doubly stochastic, so its spectral norm
    is <= 1, the set is closed under composition, and the composite mapping
    across depth therefore preserves the signal mean instead of exploding), and
    ``H^pre``/``H^post`` are made non-negative via sigmoid to prevent signal
    cancellation.

    Only the *static* mappings are learned -- see ``TrunkConfig``'s docstring for
    why the dynamic, input-dependent form is not affordable on Cartesian blocks.

    ``stream_semantics`` is the novel arm: instead of ``n`` anonymous copies of
    the residual width, the caller seeds each stream with a different modality
    (RNA, CpG, RNA x CpG product, locus prior). Under the doubly stochastic
    constraint the residual mapping is then a *mass-conserving cross-modal
    exchange operator*: ``H^res`` is a convex combination of permutations of the
    modalities, readable per depth as an interpretable exchange matrix. That
    reading is only meaningful because of the manifold constraint -- an
    unconstrained HC matrix has no conservation interpretation.
    """

    def __init__(
        self,
        width: int,
        depth: int,
        expansion: int,
        dropout: float,
        n_streams: int,
        manifold: bool,
        sinkhorn_iters: int = 20,
        identity_init_scale: float = 4.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if n_streams < 1:
            raise ValueError("trunk.n_streams must be >= 1")
        self.n_streams = int(n_streams)
        self.manifold = bool(manifold)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.blocks = nn.ModuleList(_ResidualFunction(width, expansion, dropout) for _ in range(depth))

        n = self.n_streams
        eye = torch.eye(n)
        # Init near the identity mapping in both parameterizations, so the trunk
        # starts as ``depth`` independent plain-residual streams and any stream
        # mixing has to be learned (mHC section 4.1: n=1 degenerates to the
        # scalar 1, recovering the ordinary residual connection).
        self.res_param = nn.Parameter(eye.repeat(depth, 1, 1) * float(identity_init_scale))
        if manifold:
            pre_init = torch.logit(torch.full((depth, n), 1.0 / n))
            post_init = torch.zeros(depth, n)  # 2 * sigmoid(0) = 1
        else:
            pre_init = torch.full((depth, n), 1.0 / n)
            post_init = torch.ones(depth, n)
        self.pre_param = nn.Parameter(pre_init)
        self.post_param = nn.Parameter(post_init)
        self.out_norm = nn.LayerNorm(width)

    def _mappings(self, layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.manifold:
            h_res = sinkhorn_knopp(self.res_param[layer], self.sinkhorn_iters)
            h_pre = torch.sigmoid(self.pre_param[layer].float())
            h_post = 2.0 * torch.sigmoid(self.post_param[layer].float())
        else:
            h_res = self.res_param[layer].float()
            h_pre = self.pre_param[layer].float()
            h_post = self.post_param[layer].float()
        return h_res, h_pre, h_post

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_loci, n_streams, width)
        for layer, block in enumerate(self.blocks):
            h_res, h_pre, h_post = self._mappings(layer)
            dtype = x.dtype
            h_res, h_pre, h_post = h_res.to(dtype), h_pre.to(dtype), h_post.to(dtype)
            merged = torch.einsum("j,bljc->blc", h_pre, x)
            if self.gradient_checkpointing and self.training:
                update = torch.utils.checkpoint.checkpoint(block, merged, use_reentrant=False)
            else:
                update = block(merged)
            x = torch.einsum("ij,bljc->blic", h_res, x) + h_post[None, None, :, None] * update[:, :, None, :]
        return self.out_norm(x.mean(dim=2))

    @torch.no_grad()
    def composite_gain(self) -> dict[str, float]:
        """mHC's *Amax Gain Magnitude* of the composite residual mapping.

        The product of ``H^res`` across depth governs how a signal injected at
        one layer is scaled by the time it reaches the output (max abs row sum,
        forward) and how a gradient is scaled on the way back (max abs column
        sum). For a doubly stochastic composite both are exactly 1; the mHC paper
        measures peaks near 3000 for unconstrained HC, which is what its
        stability claim rests on. Logged per epoch so the stability figure exists
        even if the headline metric does not move.
        """
        depth = len(self.blocks)
        if depth == 0:
            return {"forward_amax_gain": 1.0, "backward_amax_gain": 1.0}
        composite = torch.eye(self.n_streams, device=self.res_param.device, dtype=torch.float32)
        for layer in range(depth):
            h_res, _, _ = self._mappings(layer)
            composite = h_res @ composite
        return {
            "forward_amax_gain": float(composite.abs().sum(dim=1).max()),
            "backward_amax_gain": float(composite.abs().sum(dim=0).max()),
        }

    @torch.no_grad()
    def residual_mappings(self) -> list[list[list[float]]]:
        """Per-depth ``H^res`` matrices, for the cross-modal exchange figure."""
        return [self._mappings(layer)[0].cpu().tolist() for layer in range(len(self.blocks))]


class AxialCpGAttention(nn.Module):
    """Windowed self-attention along the CpG axis of a Cartesian block.

    A training/evaluation block's CpG axis is a *contiguous* slice of the
    source's CpG index array (see ``CartesianSourceSchedule``), so neighbouring
    columns are neighbouring loci on the chromosome. Attending inside a window
    of that axis lets a prediction use its genomic neighbourhood -- co-methylation
    is one of the strongest known structures in the methylome, and a strictly
    per-pair model cannot express it.

    Full attention over the axis is not affordable (a WGBS block has 16,384
    CpGs), so attention runs inside non-overlapping windows of ``window``
    columns; cost is ``batch * n_loci * window * n_heads`` rather than
    ``batch * n_loci^2 * n_heads``.

    Two position signals are combined:
      * a learned bias over log-spaced buckets of the within-window offset
        ``|i - j|``, which needs no coordinates at all, and
      * when ``positions`` are supplied, a learned per-head decay in
        ``log1p(|pos_i - pos_j|)``. Positions are CpG *ordinals* in the canonical
        bundle ordering, which is monotone in genomic coordinate within a
        chromosome -- a proxy for base-pair distance, not base pairs.

    The output projection is zero-initialized, so the module starts as an exact
    identity and the arm degrades gracefully to its trunk-only control.
    """

    def __init__(self, width: int, n_heads: int, window: int, n_distance_buckets: int, dropout: float):
        super().__init__()
        if width % n_heads != 0:
            raise ValueError(f"axial: width {width} must be divisible by n_heads {n_heads}")
        self.n_heads = int(n_heads)
        self.head_dim = width // n_heads
        self.window = int(window)
        self.n_buckets = int(n_distance_buckets)
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.offset_bias = nn.Embedding(self.n_buckets, self.n_heads)
        nn.init.zeros_(self.offset_bias.weight)
        self.distance_scale = nn.Parameter(torch.zeros(self.n_heads))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("_bucket_ids", self._build_buckets(self.window, self.n_buckets), persistent=False)

    @staticmethod
    def _build_buckets(window: int, n_buckets: int) -> torch.Tensor:
        idx = torch.arange(window)
        distance = (idx[:, None] - idx[None, :]).abs().float()
        scaled = torch.log1p(distance) / torch.log1p(torch.tensor(float(max(window - 1, 1))))
        return (scaled * (n_buckets - 1)).round().long().clamp_(0, n_buckets - 1)

    def forward(self, h: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        batch, n_loci, width = h.shape
        window = min(self.window, n_loci)
        pad = (-n_loci) % window
        x = self.norm(h)
        if pad:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        n_windows = x.shape[1] // window
        qkv = self.qkv(x).view(batch, n_windows, window, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=3)
        scores = torch.einsum("bwihd,bwjhd->bwhij", q, k) / (self.head_dim ** 0.5)

        buckets = self._bucket_ids[:window, :window] if window == self.window else self._build_buckets(window, self.n_buckets).to(h.device)
        bias = self.offset_bias(buckets).permute(2, 0, 1)  # (heads, window, window)
        scores = scores + bias[None, None].to(scores.dtype)

        if positions is not None:
            pos = positions.to(torch.float32)
            if pad:
                # Pad with the last real coordinate; padded keys are masked out below.
                pos = torch.cat([pos, pos[-1:].expand(pad)])
            pos = pos.view(n_windows, window)
            delta = torch.log1p((pos[:, :, None] - pos[:, None, :]).abs())
            decay = -torch.nn.functional.softplus(self.distance_scale)[None, :, None, None] * delta[:, None, :, :]
            scores = scores + decay[None].to(scores.dtype)

        if pad:
            valid = torch.ones(n_windows * window, dtype=torch.bool, device=h.device)
            valid[n_loci:] = False
            valid = valid.view(n_windows, window)
            scores = scores.masked_fill(~valid[None, :, None, None, :], float("-inf"))

        weights = self.dropout(torch.softmax(scores, dim=-1))
        attended = torch.einsum("bwhij,bwjhd->bwihd", weights, v)
        attended = attended.reshape(batch, n_windows * window, width)[:, :n_loci]
        return h + self.out(attended)


class ArchitectureVariantModel(nn.Module):
    """Configurable model for the architecture-novelty suite.

    Keeps the canonical prediction contract exactly -- the RNA-conditioned
    network produces a variance-standardized residual that is anchored on the
    frozen CpG statistics::

        logit(beta_hat_{s,i}) = logit(mu_i) + sigma_i * raw_delta_{s,i}

    -- so every arm's MAS-PCC is directly comparable to the canonical 0.5613 and
    to the ``fusion_mechanism_2026_08`` arms, all of which ran on this same
    matched_chr1 engine and official split. What varies is only *how*
    ``raw_delta`` is computed:

      * ``model.encoder.kind`` -- the RNA branch (``linear`` canonical,
        ``mlp``/``program_bottleneck`` capacity controls, ``locus_attention``
        the locus-conditioned token-set encoder).
      * ``model.trunk`` -- depth and residual topology after fusion
        (``none`` canonical depth-1 head, ``plain`` the depth control, ``hc``
        and ``mhc`` the multi-stream arms).
      * ``model.axial`` -- optional windowed attention along the CpG axis.

    With all three at their defaults this class is architecturally identical to
    ``RNAMethylationPredictor`` (same joint input, same depth-1 head, same
    anchor), which makes it a valid same-code control -- but it is a separate
    class with separate state-dict keys on purpose: the canonical predictor must
    keep its exact parameter set for historical checkpoint loading.
    """

    def __init__(self, input_dim: int, locus_dim: int, config: ModelConfig, epsilon: float = 1e-4):
        super().__init__()
        if not config.zero_init_residual:
            raise ValueError("architecture-variant models require zero_init_residual=true (start exactly at the prior)")
        if not config.use_prior_anchor:
            raise ValueError(
                "architecture-variant models require use_prior_anchor=true; dropping the anchor is a separate, "
                "already-measured ablation (ablations.yaml::prior_anchor_2026_08)"
            )
        if not config.variance_normalized_residual:
            raise ValueError(
                "architecture-variant models require variance_normalized_residual=true so their MAS-PCC stays "
                "comparable to the canonical chr1 number"
            )
        self.config = config
        self.epsilon = float(epsilon)

        encoder_cfg, interaction_cfg = config.encoder, config.interaction
        trunk_cfg, axial_cfg = config.trunk, config.axial
        self.rna_encoder = build_rna_encoder(encoder_cfg, input_dim=input_dim, locus_dim=locus_dim)
        self.locus_conditioned = encoder_cfg.kind == "locus_attention"
        rna_pair_dim = encoder_cfg.program_dim if self.locus_conditioned else encoder_cfg.latent_dim

        self.include_rna = bool(interaction_cfg.include_rna)
        self.include_cpg = bool(interaction_cfg.include_cpg)
        self.include_product = bool(interaction_cfg.include_product)
        if not (self.include_rna or self.include_cpg or self.include_product):
            raise ValueError("at least one of interaction.include_rna/include_cpg/include_product must be true")
        product_dim = min(rna_pair_dim, locus_dim)
        self.rna_product = nn.Linear(rna_pair_dim, product_dim) if self.include_product else None
        self.locus_product = nn.Linear(locus_dim, product_dim) if self.include_product else None

        self.stream_semantics = bool(trunk_cfg.stream_semantics)
        piece_dims = {"rna": rna_pair_dim, "cpg": locus_dim, "product": product_dim, "prior": 2}
        joint_dim = (
            (rna_pair_dim if self.include_rna else 0)
            + (locus_dim if self.include_cpg else 0)
            + (product_dim if self.include_product else 0)
        )

        self.trunk_kind = trunk_cfg.kind
        if self.trunk_kind == "none":
            if self.stream_semantics:
                raise ValueError("trunk.stream_semantics requires an actual trunk (trunk.kind != 'none')")
            width = interaction_cfg.hidden_dim
            # Byte-for-byte the canonical head's feature stage, so the all-default
            # configuration of this class reproduces the canonical architecture.
            self.feature_stage = nn.Sequential(
                nn.LayerNorm(joint_dim),
                nn.Linear(joint_dim, width),
                nn.GELU(),
                nn.Dropout(interaction_cfg.dropout),
            )
            self.trunk = None
            self.stream_projections = None
        else:
            if trunk_cfg.depth < 1:
                raise ValueError(f"trunk.kind={trunk_cfg.kind!r} requires trunk.depth >= 1")
            width = trunk_cfg.width
            if self.stream_semantics:
                if trunk_cfg.kind == "plain":
                    raise ValueError("trunk.stream_semantics is only meaningful for a multi-stream trunk (hc/mhc)")
                if trunk_cfg.n_streams != 4:
                    raise ValueError(
                        "trunk.stream_semantics assigns one stream per modality (rna, cpg, product, prior); "
                        f"trunk.n_streams must be 4, got {trunk_cfg.n_streams}"
                    )
                if not (self.include_rna and self.include_cpg and self.include_product):
                    raise ValueError("trunk.stream_semantics needs all three interaction pieces enabled")
                self.feature_stage = None
                self.stream_projections = nn.ModuleDict(
                    {name: nn.Linear(dim, width) for name, dim in piece_dims.items()}
                )
            else:
                self.feature_stage = nn.Sequential(nn.LayerNorm(joint_dim), nn.Linear(joint_dim, width))
                self.stream_projections = None
            if trunk_cfg.kind == "plain":
                self.trunk = PlainTrunk(
                    width=width, depth=trunk_cfg.depth, expansion=trunk_cfg.expansion,
                    dropout=trunk_cfg.dropout, gradient_checkpointing=trunk_cfg.gradient_checkpointing,
                )
            elif trunk_cfg.kind in {"hc", "mhc"}:
                self.trunk = HyperConnectionTrunk(
                    width=width, depth=trunk_cfg.depth, expansion=trunk_cfg.expansion,
                    dropout=trunk_cfg.dropout, n_streams=trunk_cfg.n_streams,
                    manifold=(trunk_cfg.kind == "mhc"), sinkhorn_iters=trunk_cfg.sinkhorn_iters,
                    identity_init_scale=trunk_cfg.identity_init_scale,
                    gradient_checkpointing=trunk_cfg.gradient_checkpointing,
                )
            else:
                raise ValueError(f"unknown trunk.kind: {trunk_cfg.kind!r}")

        self.axial = (
            AxialCpGAttention(
                width=width, n_heads=axial_cfg.n_heads, window=axial_cfg.window,
                n_distance_buckets=axial_cfg.n_distance_buckets, dropout=axial_cfg.dropout,
            )
            if axial_cfg.enabled
            else None
        )
        self.head = nn.Linear(width, 1)
        self.concentration_head = nn.Linear(width, 1) if config.beta_likelihood_head else None
        self.zero_output()

    @property
    def supports_factorized_inference(self) -> bool:
        return False

    @property
    def requires_cpg_positions(self) -> bool:
        """True when the axial arm can use genomic ordinals (optional, not required)."""
        return self.axial is not None

    def zero_output(self) -> None:
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _pieces(self, rna: torch.Tensor, loci: torch.Tensor, prior_logit: torch.Tensor, sigma: torch.Tensor):
        representation = self.rna_encoder(rna)
        batch = rna.shape[0]
        n_loci = loci.shape[0]
        if self.locus_conditioned:
            assert representation.program_tokens is not None
            rna_pair = self.rna_encoder.attend(representation.program_tokens, loci)  # (batch, n_loci, program_dim)
        else:
            rna_pair = representation.global_vector[:, None, :].expand(batch, n_loci, -1)
        pieces = {"rna": rna_pair}
        pieces["cpg"] = loci[None, :, :].expand(batch, n_loci, -1)
        if self.include_product:
            assert self.rna_product is not None and self.locus_product is not None
            pieces["product"] = self.rna_product(rna_pair) * self.locus_product(loci)[None, :, :]
        pieces["prior"] = torch.stack([prior_logit, sigma], dim=-1)[None, :, :].expand(batch, n_loci, -1)
        return pieces

    def forward(
        self,
        rna: torch.Tensor,
        loci: torch.Tensor,
        prior: torch.Tensor,
        sigma: torch.Tensor,
        cpg_positions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        prior = prior.clamp(self.epsilon, 1.0 - self.epsilon)
        prior_logit = torch.logit(prior)
        sigma_safe = sigma.clamp_min(1e-6)
        pieces = self._pieces(rna, loci, prior_logit, sigma_safe)

        if self.stream_semantics:
            assert self.stream_projections is not None
            streams = torch.stack(
                [self.stream_projections[name](pieces[name]) for name in ("rna", "cpg", "product", "prior")],
                dim=2,
            )  # (batch, n_loci, 4, width)
            hidden = self.trunk(streams)
        else:
            selected = []
            if self.include_rna:
                selected.append(pieces["rna"])
            if self.include_cpg:
                selected.append(pieces["cpg"])
            if self.include_product:
                selected.append(pieces["product"])
            joint = torch.cat(selected, dim=-1)
            hidden = self.feature_stage(joint)
            if self.trunk is not None:
                if self.trunk.n_streams > 1:
                    hidden = hidden[:, :, None, :].expand(-1, -1, self.trunk.n_streams, -1)
                hidden = self.trunk(hidden)

        if self.axial is not None:
            hidden = self.axial(hidden, positions=cpg_positions)

        raw_delta = self.head(hidden).squeeze(-1)
        delta_logit = sigma_safe.unsqueeze(0) * raw_delta
        prediction_logit = prior_logit.unsqueeze(0) + delta_logit
        beta = torch.sigmoid(prediction_logit)

        outputs = {
            "beta": beta,
            "delta_logit": delta_logit,
            "raw_delta": raw_delta,
            "raw_delta_logit": delta_logit,
            "prediction_logit": prediction_logit,
            "gate": torch.ones(loci.shape[0], dtype=loci.dtype, device=loci.device),
            "prior_logit": prior_logit,
        }
        if self.concentration_head is not None:
            # Softplus keeps the Beta concentration positive; the loss applies the
            # configured floor (below ~2 the density turns U-shaped).
            outputs["concentration"] = torch.nn.functional.softplus(self.concentration_head(hidden).squeeze(-1))
        return outputs

    def diagnostics(self) -> dict[str, float]:
        """Per-epoch architecture diagnostics for history.json (never a loss term)."""
        if self.trunk is None or not hasattr(self.trunk, "composite_gain"):
            return {}
        return self.trunk.composite_gain()


def is_architecture_variant(config: ModelConfig) -> bool:
    """True when a recipe opted into the architecture-novelty suite.

    Kept as an explicit predicate (rather than a magic ``architecture:`` string)
    so a recipe that touches none of these blocks provably still builds the
    frozen canonical class.
    """
    return (
        config.encoder.kind != "linear"
        or config.trunk.kind != "none"
        or config.axial.enabled
        or config.beta_likelihood_head
    )


def architecture_variant_label(config: ModelConfig) -> str:
    """Compact, checkpoint-safe description of a variant's topology.

    Appended to the trainer's ``architecture_label`` so two different variants
    can never resume from each other's ``latest.pt``.
    """
    parts = [f"enc-{config.encoder.kind}"]
    if config.encoder.kind == "locus_attention":
        parts.append(f"k{config.encoder.n_programs}h{config.encoder.n_heads}d{config.encoder.program_dim}")
    if config.trunk.kind != "none":
        parts.append(f"trunk-{config.trunk.kind}d{config.trunk.depth}w{config.trunk.width}")
        if config.trunk.kind in {"hc", "mhc"}:
            parts.append(f"n{config.trunk.n_streams}" + ("sem" if config.trunk.stream_semantics else ""))
    if config.axial.enabled:
        parts.append(f"axial-w{config.axial.window}h{config.axial.n_heads}")
    if config.beta_likelihood_head:
        parts.append("betahead")
    return "_".join(parts)


class FeatureFusionArchitectureVariantModel(nn.Module):
    """Configurable extension of ``FeatureFusionLocusCLSModel`` for the
    architecture-novelty suite (``architecture_novelty_2026_09``), retargeted
    2026-09-04 after ``FeatureFusionLocusCLSModel`` was selected as the repo's
    primary/reference architecture (``shared_backbone_locus_cls_2026_09``,
    ``docs/RNA_METHYLATION.md``). The suite's earlier work
    (``ArchitectureVariantModel``, above) targeted the now-superseded two-stage
    frozen-prior model and stays only for old-checkpoint-compatible experiments.

    Same constructor signature as ``FeatureFusionLocusCLSModel`` (drop-in for
    ``LocusCLSJointTrainer``), same two branches -- a locus-only mean branch
    (``CpGTrunk``) and an RNA-conditioned raw branch -- but three axes become
    configurable through ``ModelConfig``:

      * ``config.encoder`` -- the RNA branch (``linear`` canonical,
        ``mlp``/``program_bottleneck`` capacity controls, ``locus_attention``
        makes the raw branch's RNA representation locus-specific).
      * ``config.trunk`` -- how the two branch embeddings are combined. This is
        the architecture's own core design question, and the postdoc's original
        suggestion applies most literally here: with
        ``trunk.stream_semantics=True`` (requires ``trunk.n_streams=2``), the
        mean and raw embeddings themselves become the two streams of an
        HC/mHC trunk -- they exchange information under a (for mHC) doubly
        stochastic, mass-conserving mixing matrix for ``trunk.depth`` steps,
        instead of being concatenated once into a single ``Linear``. Without
        ``stream_semantics``, the trunk instead deepens the already-concatenated
        joint representation (the depth control every HC/mHC claim needs).
      * ``config.axial`` / ``config.beta_likelihood_head`` -- unchanged from the
        two-stage suite's versions (windowed CpG-axis attention on the final
        joint representation; an additional Beta-concentration output head).

    At every default (``encoder.kind="linear"``, ``trunk.kind="none"``,
    ``axial.enabled=False``, no beta head) this reduces to
    ``FeatureFusionLocusCLSModel``'s exact forward computation -- verified
    numerically in tests -- so it is a valid same-code control for the current
    reference architecture, the same role ``ArchitectureVariantModel`` played
    for the retired one.

    Unlike the two-stage suite, there is no prior/sigma anchor here to give an
    exact "starts at the prior" contract: this architecture's own established
    convention (``fusion_init_std``, see ``FeatureFusionLocusCLSModel``'s
    docstring) is a small *nonzero* init, because a hard-zero final layer was
    measured to starve the raw branch of gradient. The exact, testable
    invariant this class keeps is ``fusion_init_std=0.0`` -> the final head's
    weight and bias are both exactly zero -> every prediction is exactly 0.5
    at step 0, regardless of which encoder/trunk/axial configuration is active.
    """

    def __init__(
        self,
        input_dim: int,
        cpg_input_dim: int,
        config: ModelConfig,
        trunk_hidden_dim: int = 256,
        bottleneck_dim: int = 64,
        trunk_dropout: float = 0.1,
        use_mean_branch: bool = True,
        use_fusion_product: bool = False,
        fusion_init_std: float = 0.01,
    ):
        super().__init__()
        self.config = config
        self.use_mean_branch = bool(use_mean_branch)
        self.use_fusion_product = bool(use_fusion_product)
        trunk_cfg, axial_cfg = config.trunk, config.axial
        self.stream_semantics = bool(trunk_cfg.stream_semantics)
        if self.stream_semantics:
            if not self.use_mean_branch:
                raise ValueError("trunk.stream_semantics needs both branches: use_mean_branch=True")
            if trunk_cfg.kind == "plain":
                raise ValueError("trunk.stream_semantics is only meaningful for a multi-stream trunk (hc/mhc)")
            if trunk_cfg.n_streams != 2:
                raise ValueError(
                    "trunk.stream_semantics assigns one stream per branch (mean, raw); "
                    f"trunk.n_streams must be 2, got {trunk_cfg.n_streams}"
                )

        if self.use_mean_branch:
            self.trunk_cpg = CpGTrunk(cpg_input_dim, trunk_hidden_dim, bottleneck_dim, trunk_dropout)
            self.mean_head = nn.Linear(bottleneck_dim, 1)

        encoder_cfg, interaction_cfg = config.encoder, config.interaction
        self.rna_encoder = build_rna_encoder(encoder_cfg, input_dim=input_dim, locus_dim=cpg_input_dim)
        self.locus_conditioned = encoder_cfg.kind == "locus_attention"
        rna_pair_dim = encoder_cfg.program_dim if self.locus_conditioned else encoder_cfg.latent_dim
        raw_hidden_dim = interaction_cfg.hidden_dim
        dropout = interaction_cfg.dropout
        product_dim = min(rna_pair_dim, cpg_input_dim)
        self.rna_product = nn.Linear(rna_pair_dim, product_dim)
        self.locus_product = nn.Linear(cpg_input_dim, product_dim)
        joint_dim = rna_pair_dim + cpg_input_dim + product_dim
        self.raw_branch = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Linear(joint_dim, raw_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_head = nn.Linear(raw_hidden_dim, 1)  # auxiliary probe only, mirrors FeatureFusionLocusCLSModel

        # ---- how the two branch embeddings are combined -------------------
        trunk_kind = trunk_cfg.kind
        if trunk_kind == "none":
            # Byte-for-byte FeatureFusionLocusCLSModel's own fusion stage, so the
            # all-default configuration reproduces it exactly.
            fusion_dim = raw_hidden_dim + (bottleneck_dim if self.use_mean_branch else 0)
            if self.use_fusion_product:
                if not self.use_mean_branch:
                    raise ValueError("use_fusion_product requires use_mean_branch")
                self.fusion_product_dim = min(bottleneck_dim, raw_hidden_dim)
                self.mean_fusion_proj = nn.Linear(bottleneck_dim, self.fusion_product_dim)
                self.raw_fusion_proj = nn.Linear(raw_hidden_dim, self.fusion_product_dim)
                fusion_dim += self.fusion_product_dim
            self.combine_stage = None
            self.trunk = None
            width = fusion_dim
        else:
            if trunk_cfg.depth < 1:
                raise ValueError(f"trunk.kind={trunk_kind!r} requires trunk.depth >= 1")
            width = trunk_cfg.width
            if self.stream_semantics:
                self.mean_stream_proj = nn.Linear(bottleneck_dim, width)
                self.raw_stream_proj = nn.Linear(raw_hidden_dim, width)
                self.combine_stage = None
            else:
                fusion_dim = raw_hidden_dim + (bottleneck_dim if self.use_mean_branch else 0)
                self.combine_stage = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, width))
            if trunk_kind == "plain":
                self.trunk = PlainTrunk(
                    width=width, depth=trunk_cfg.depth, expansion=trunk_cfg.expansion,
                    dropout=trunk_cfg.dropout, gradient_checkpointing=trunk_cfg.gradient_checkpointing,
                )
            elif trunk_kind in {"hc", "mhc"}:
                self.trunk = HyperConnectionTrunk(
                    width=width, depth=trunk_cfg.depth, expansion=trunk_cfg.expansion,
                    dropout=trunk_cfg.dropout, n_streams=trunk_cfg.n_streams,
                    manifold=(trunk_kind == "mhc"), sinkhorn_iters=trunk_cfg.sinkhorn_iters,
                    identity_init_scale=trunk_cfg.identity_init_scale,
                    gradient_checkpointing=trunk_cfg.gradient_checkpointing,
                )
            else:
                raise ValueError(f"unknown trunk.kind: {trunk_kind!r}")

        self.axial = (
            AxialCpGAttention(
                width=width, n_heads=axial_cfg.n_heads, window=axial_cfg.window,
                n_distance_buckets=axial_cfg.n_distance_buckets, dropout=axial_cfg.dropout,
            )
            if axial_cfg.enabled
            else None
        )
        self.fusion = nn.Linear(width, 1)
        self.concentration_head = nn.Linear(width, 1) if config.beta_likelihood_head else None

        # Same convention as FeatureFusionLocusCLSModel: fusion_init_std=0.0
        # reproduces a hard-zero start (every prediction exactly 0.5), the
        # default 0.01 avoids starving the raw/trunk parameters of gradient at
        # step 0 (measured empirically on that model -- see its docstring).
        if fusion_init_std > 0:
            nn.init.normal_(self.fusion.weight, std=fusion_init_std)
        else:
            nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    @property
    def requires_cpg_positions(self) -> bool:
        return self.axial is not None

    def diagnostics(self) -> dict[str, float]:
        if self.trunk is None or not hasattr(self.trunk, "composite_gain"):
            return {}
        return self.trunk.composite_gain()

    def forward(
        self, rna: torch.Tensor, cpg_embedding: torch.Tensor, cpg_positions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        representation = self.rna_encoder(rna)
        batch = rna.shape[0]
        n_loci = cpg_embedding.shape[0]
        if self.locus_conditioned:
            assert representation.program_tokens is not None
            rna_pair = self.rna_encoder.attend(representation.program_tokens, cpg_embedding)
        else:
            rna_pair = representation.global_vector[:, None, :].expand(batch, n_loci, -1)

        product = self.rna_product(rna_pair) * self.locus_product(cpg_embedding)[None, :, :]
        joint = torch.cat([rna_pair, cpg_embedding[None, :, :].expand(batch, n_loci, -1), product], dim=-1)
        h_raw = self.raw_branch(joint)  # (batch, n_loci, raw_hidden_dim)
        residual_logit = self.residual_head(h_raw).squeeze(-1)  # auxiliary probe only

        if self.use_mean_branch:
            h_mean = self.trunk_cpg(cpg_embedding)  # (n_loci, bottleneck_dim)
            mu_logit = self.mean_head(h_mean).squeeze(-1)  # auxiliary probe only
            h_mean_b = h_mean[None, :, :].expand(batch, n_loci, -1)
        else:
            mu_logit = None
            h_mean_b = None

        if self.trunk is None:
            pieces = [h_mean_b, h_raw] if self.use_mean_branch else [h_raw]
            if self.use_fusion_product:
                fusion_product = self.mean_fusion_proj(h_mean)[None, :, :] * self.raw_fusion_proj(h_raw)
                pieces.append(fusion_product)
            hidden = torch.cat(pieces, dim=-1)
        elif self.stream_semantics:
            mean_stream = self.mean_stream_proj(h_mean_b)
            raw_stream = self.raw_stream_proj(h_raw)
            streams = torch.stack([mean_stream, raw_stream], dim=2)  # (batch, n_loci, 2, width)
            hidden = self.trunk(streams)
        else:
            pieces = [h_mean_b, h_raw] if self.use_mean_branch else [h_raw]
            hidden = self.combine_stage(torch.cat(pieces, dim=-1))
            if self.trunk.n_streams > 1:
                hidden = hidden[:, :, None, :].expand(-1, -1, self.trunk.n_streams, -1)
            hidden = self.trunk(hidden)

        if self.axial is not None:
            hidden = self.axial(hidden, positions=cpg_positions)

        prediction_logit = self.fusion(hidden).squeeze(-1)
        beta = torch.sigmoid(prediction_logit)
        outputs = {
            "beta": beta,
            "delta_logit": prediction_logit,
            "raw_delta": prediction_logit,
            "prediction_logit": prediction_logit,
            "mu_logit": mu_logit,
            "residual_logit": residual_logit,
            "h_cpg": h_mean if self.use_mean_branch else None,
        }
        if self.concentration_head is not None:
            outputs["concentration"] = torch.nn.functional.softplus(self.concentration_head(hidden).squeeze(-1))
        return outputs


def feature_fusion_variant_label(config: ModelConfig) -> str:
    """Compact, checkpoint-safe description of a shared-backbone variant's
    topology -- mirrors ``architecture_variant_label`` for the two-stage suite."""
    parts = [f"enc-{config.encoder.kind}"]
    if config.encoder.kind == "locus_attention":
        parts.append(f"k{config.encoder.n_programs}h{config.encoder.n_heads}d{config.encoder.program_dim}")
    if config.trunk.kind != "none":
        parts.append(f"trunk-{config.trunk.kind}d{config.trunk.depth}w{config.trunk.width}")
        if config.trunk.kind in {"hc", "mhc"}:
            parts.append(f"n{config.trunk.n_streams}" + ("sem" if config.trunk.stream_semantics else ""))
    if config.axial.enabled:
        parts.append(f"axial-w{config.axial.window}h{config.axial.n_heads}")
    if config.beta_likelihood_head:
        parts.append("betahead")
    return "_".join(parts)
