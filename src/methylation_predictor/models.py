"""RNA-conditioned methylation models.

The reference architecture is the single-stage shared-backbone model,
``FeatureFusionArchitectureVariantModel`` (``encoder.kind=locus_attention``
selects its primary configuration; ``encoder.kind=linear`` reduces it to the
earlier ``FeatureFusionLocusCLSModel`` topology, kept as a comparison arm):

    CpG reference context -> frozen NTv3 embedding e_l -> CpGTrunk -> h_mean_l
    patient RNA x_s -> RNA encoder -> locus-conditioned representation r_s,l
    [r_s,l, e_l] -> raw interaction branch h_raw_s,l
    [h_mean_l, h_raw_s,l] -> fusion -> beta_hat_s,l

See ``docs/RNA_METHYLATION.md`` for the full architecture history and the
ongoing RNA-encoder comparison harness (``EncoderConfig.kind``/
``build_rna_encoder``). The earlier two-stage frozen-prior + residual
generation (``RNAMethylationPredictor``, ``VarianceNormalizedResidualModel``,
``RNA2DNAmModel``, ``ArchitectureVariantModel``, ``MethylProphetTrainer``) has
been retired; its frozen numbers remain under
``results/reference/methylprophet_comparison/`` for historical provenance.
"""
from __future__ import annotations

from dataclasses import dataclass

import json

from pathlib import Path

import numpy as np

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class _SparseMMFloat32(torch.autograd.Function):
    """CSR @ dense with an explicitly fp32 backward for CUDA AMP."""
    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.sparse = sparse
        return torch.sparse.mm(sparse, dense.float())

    @staticmethod
    def backward(ctx, grad_output):
        grad_dense = torch.sparse.mm(ctx.sparse.transpose(0, 1), grad_output.float())
        return None, grad_dense
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
# reference ProductInteraction (its own include_rna/include_cpg/include_product
# flags cover the concat/product-only ablation axis); "bilinear"/"global_shift"
# back the Bilinear RNA-CpG / Global RNA Shift required baselines
# (docs/PAPER_EXPERIMENTS.md) via FeatureFusionArchitectureVariantModel's
# ``interaction.kind`` axis.
def build_interaction(config: "InteractionConfig", rna_dim: int, locus_dim: int) -> nn.Module:
    if config.kind == "concat":
        return ProductInteraction(
            rna_dim=rna_dim, locus_dim=locus_dim, hidden_dim=config.hidden_dim, dropout=config.dropout,
            include_rna=config.include_rna, include_cpg=config.include_cpg, include_product=config.include_product,
        )
    if config.kind == "bilinear":
        return BilinearInteraction(
            rna_dim=rna_dim, locus_dim=locus_dim, hidden_dim=config.hidden_dim, dropout=config.dropout,
            rank=config.rank, include_rna=config.include_rna, include_cpg=config.include_cpg,
        )
    if config.kind == "global_shift":
        return GlobalShiftInteraction(rna_dim=rna_dim, locus_dim=locus_dim, hidden_dim=config.hidden_dim, dropout=config.dropout)
    raise ValueError(f"unknown interaction.kind: {config.kind!r}")


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
    frozen-prior + residual pipeline (retired; its frozen numbers remain
    under ``results/reference/methylprophet_comparison/``). Trained
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
# Everything below is opt-in and unreachable from the reference
# ``FeatureFusionLocusCLSModel`` above: it is selected only when a recipe sets
# a non-default ``model.encoder.kind``, ``model.trunk`` or ``model.axial``
# block, which routes construction to ``FeatureFusionArchitectureVariantModel``
# instead.
#
# Motivation for the suite (postdoc review, 2026-09-03): the reference model
# encoder was a single Linear RNA projection feeding a depth-1 fusion head, and
# every drop-in fusion swap had already been measured and lost
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



class GenePathwayEncoder(nn.Module):
    """Biologically sparse gene->pathway RNA encoder.

    MethylProphet's gene-encoding ablation replaces its Bottleneck MLP with the
    sparse pathway MLP of Jaume et al. / SurvPath. SurvPath uses a gene-pathway
    membership mask followed by a pathway-block-diagonal second layer with
    default widths 8 and 16. This implementation preserves that topology, but
    returns this repo's fixed-size RNA latent so CpG branch, mean proxy, fusion
    and methylation objective remain unchanged.

    The input is the same canonical z-scored RNA cache used by the other
    from-scratch encoder comparators. ``membership_path`` is generated by
    ``scripts/prepare_gene_pathway_membership.py`` in the current canonical RNA
    gene order.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        membership_path: str,
        dim1: int = 8,
        dim2: int = 16,
        dropout: float = 0.1,
        layer_norm: bool = True,
    ):
        super().__init__()
        if not membership_path:
            raise ValueError(
                "encoder.kind='gene_pathway' requires encoder.pathway_membership_path; "
                "run scripts/prepare_gene_pathway_membership.py first"
            )
        path = Path(membership_path)
        if not path.is_file():
            raise FileNotFoundError(f"gene-pathway membership file not found: {path}")
        data = np.load(path, allow_pickle=False)
        n_genes = int(np.asarray(data["n_genes"]).reshape(-1)[0])
        n_pathways = int(np.asarray(data["n_pathways"]).reshape(-1)[0])
        gene_idx = np.asarray(data["gene_idx"], dtype=np.int64)
        pathway_idx = np.asarray(data["pathway_idx"], dtype=np.int64)
        if n_genes != input_dim:
            raise ValueError(
                f"pathway membership was built for {n_genes} genes but RNA cache width is {input_dim}; "
                "regenerate the membership against this canonical bundle"
            )
        if gene_idx.ndim != 1 or pathway_idx.ndim != 1 or gene_idx.shape != pathway_idx.shape:
            raise ValueError("invalid gene-pathway membership: gene_idx/pathway_idx must be equal-length 1-D arrays")
        if len(gene_idx) == 0:
            raise ValueError("gene-pathway membership has zero edges")

        self.input_dim = int(input_dim)
        self.n_pathways = int(n_pathways)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.norm = nn.LayerNorm(input_dim) if layer_norm else nn.Identity()

        indices = np.stack([pathway_idx, gene_idx], axis=0)
        self.register_buffer("membership_indices", torch.from_numpy(indices).long(), persistent=True)
        n_edges = int(gene_idx.size)
        per_path = np.bincount(pathway_idx, minlength=n_pathways)
        scale = float(1.0 / max(np.sqrt(max(float(per_path.mean()), 1.0)), 1.0))
        self.edge_weight = nn.Parameter(torch.empty(self.dim1, n_edges))
        nn.init.normal_(self.edge_weight, std=scale)
        self.bias1 = nn.Parameter(torch.zeros(n_pathways, self.dim1))

        # Independent 8->16 transformation inside every pathway, equivalent to
        # SurvPath's second block-diagonal sparse mask without materialising it.
        self.weight2 = nn.Parameter(torch.empty(n_pathways, self.dim1, self.dim2))
        nn.init.xavier_normal_(self.weight2)
        self.bias2 = nn.Parameter(torch.zeros(n_pathways, self.dim2))

        flattened = n_pathways * self.dim2
        hidden = max(latent_dim, flattened // 4)
        self.to_latent = nn.Sequential(
            nn.LayerNorm(flattened),
            nn.Linear(flattened, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> RNARepresentation:
        x = self.norm(x)
        # Sparse mm is kept in fp32 for CUDA robustness; the dense projection
        # remains inside the trainer's normal autocast context. `.float()`
        # alone isn't enough here: CUDA sparse addmm has no bf16/fp16 kernel,
        # and an ambient AMP autocast region will still re-cast a plain
        # fp32 tensor down before dispatching a matmul-family op, so the
        # sparse mm itself must run with autocast explicitly disabled.
        x_t = x.float().transpose(0, 1).contiguous()  # genes x batch
        first_channels = []
        with torch.autocast(device_type=x.device.type, enabled=False):
            for channel in range(self.dim1):
                weight = torch.sparse_coo_tensor(
                    self.membership_indices,
                    self.edge_weight[channel].float(),
                    size=(self.n_pathways, self.input_dim),
                    device=x.device,
                ).coalesce()
                first_channels.append(torch.sparse.mm(weight, x_t).transpose(0, 1))
        h1 = torch.stack(first_channels, dim=-1) + self.bias1.float()[None, :, :]
        h2 = torch.einsum("bpi,pij->bpj", h1, self.weight2.float()) + self.bias2.float()[None, :, :]
        return RNARepresentation(self.to_latent(h2.reshape(h2.shape[0], -1)))


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
    weights are directly plottable for diagnostic analysis.
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
        mean_query_dim: int | None = None,
        query_source: str = "ntv3",
    ):
        super().__init__()
        if program_dim % n_heads != 0:
            raise ValueError(f"encoder.program_dim {program_dim} must be divisible by encoder.n_heads {n_heads}")
        allowed_query_sources = {"ntv3", "mean_only", "hybrid_detached", "hybrid_joint"}
        if query_source not in allowed_query_sources:
            raise ValueError(
                f"unknown locus-attention query_source={query_source!r}; "
                f"expected one of {sorted(allowed_query_sources)}"
            )
        if query_source != "ntv3" and mean_query_dim is None:
            raise ValueError(f"query_source={query_source!r} requires mean_query_dim")
        self.query_source = str(query_source)
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
        # Keep this parameter after every pre-existing locus-attention layer so
        # adding a hybrid query does not perturb the initialization of Q/K/V/out.
        # For hybrid modes it is zero-initialized: step 0 is exactly the current
        # raw-NTv3 query, and training must earn any contribution from h_mean.
        self.mean_query = None
        if self.query_source != "ntv3":
            self.mean_query = nn.Linear(int(mean_query_dim), program_dim, bias=False)
            if self.query_source.startswith("hybrid_"):
                nn.init.zeros_(self.mean_query.weight)
        self.dropout = nn.Dropout(dropout)
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> RNARepresentation:
        shared = self.bottleneck_norm(torch.nn.functional.gelu(self.bottleneck(self.norm(x))))
        tokens = self.token_norm(torch.einsum("br,kdr->bkd", shared, self.program_basis))
        # The "global" vector keeps a locus-invariant summary available (used by
        # the product term and by any arm that also concatenates raw RNA), so
        # this encoder is a strict superset of the canonical information.
        return RNARepresentation(tokens.mean(dim=1), program_tokens=tokens)

    def attend(
        self,
        tokens: torch.Tensor,
        loci: torch.Tensor,
        mean_features: torch.Tensor | None = None,
        store_attention: bool = False,
    ) -> torch.Tensor:
        """Query RNA program tokens with one of four controlled CpG representations.

        ``ntv3`` is the existing reference: q = W_e e_c.
        ``mean_only`` uses the CpGTrunk representation shaped by the mean-proxy task.
        ``hybrid_detached`` adds a zero-initialized W_mu stopgrad(h_mean), so the
        attention can use the methylation-aware representation without changing it.
        ``hybrid_joint`` removes stopgrad and lets the prediction task co-adapt h_mean.
        """
        batch, n_programs, _ = tokens.shape
        n_loci = loci.shape[0]
        q_ntv3 = self.query(loci)
        if self.query_source == "ntv3":
            q = q_ntv3
        else:
            if mean_features is None or self.mean_query is None:
                raise ValueError(f"query_source={self.query_source!r} requires mean_features")
            mean_input = mean_features.detach() if self.query_source == "hybrid_detached" else mean_features
            q_mean = self.mean_query(mean_input)
            q = q_mean if self.query_source == "mean_only" else q_ntv3 + q_mean
        q = q.view(n_loci, self.n_heads, self.head_dim)
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

    def query_diagnostics(self) -> dict[str, float]:
        if self.mean_query is None:
            return {}
        ntv3_norm = float(self.query.weight.detach().norm())
        mean_norm = float(self.mean_query.weight.detach().norm())
        return {
            "query_ntv3_weight_norm": ntv3_norm,
            "query_mean_weight_norm": mean_norm,
            "query_mean_to_ntv3_norm_ratio": mean_norm / max(ntv3_norm, 1e-12),
        }


class _BottleneckMLPBlock(nn.Module):
    """One pre-norm residual block: ``x + Linear(GELU(Linear(LayerNorm(x))))``."""

    def __init__(self, width: int, inner_width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, inner_width)
        self.fc2 = nn.Linear(inner_width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.dropout(torch.nn.functional.gelu(self.fc1(self.norm(x)))))
        return x + h


class BottleneckMLPEncoder(nn.Module):
    """MethylProphet-style RNA branch: input projection + N pre-norm residual
    bottleneck-MLP blocks.

    Reproduces the *architecture* of MethylProphet's published RNA encoder
    (bioRxiv 2025.02.05.636730, github.com/xk-huang/methylprophet,
    ``BottleneckMLP`` "B_6-Wi_1024": 6 blocks, width 1024, GELU, LayerNorm
    pre-norm, ``x = x + Linear(GELU(Linear(LayerNorm(x))))``). Deliberately
    *not* a full reproduction of their pipeline: (1) preprocessing -- they
    log-quantize raw counts to [0,1]; this encoder consumes the same frozen
    z-scored RNA cache as every other arm here, so it stays comparable to
    ``linear``/``mlp``/``program_bottleneck``/``locus_attention`` on the same
    ablation protocol rather than confounding architecture with preprocessing;
    (2) fusion -- they concatenate this branch's output as a token into a
    DistilBERT sequence, whereas here it plugs into this harness's existing
    fixed branch/fusion architecture as a locus-invariant global vector, the
    same contract ``MLPRNAEncoder`` already uses. This arm is trained from
    scratch (unlike ``frozen_embedding`` below), replacing the encoder only.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        n_blocks: int,
        mlp_ratio: int,
        dropout: float,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim) if layer_norm else nn.Identity()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            _BottleneckMLPBlock(hidden_dim, hidden_dim * mlp_ratio, dropout) for _ in range(n_blocks)
        )
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> RNARepresentation:
        h = self.input_proj(self.norm(x))
        for block in self.blocks:
            h = block(h)
        return RNARepresentation(self.output_proj(h))


class FrozenEmbeddingEncoder(nn.Module):
    """Thin trainable adapter over a precomputed, frozen embedding.

    Unlike every other encoder above, this one never sees raw/z-scored gene
    expression: ``x`` is expected to already be a per-sample embedding
    produced *offline* by a frozen pretrained transcriptome foundation model
    (e.g. BulkRNABert -- github.com/instadeepai/multiomics-open-research,
    CC BY-NC-SA 4.0, non-commercial; see
    ``scripts/prepare_bulkrnabert_embeddings.py`` and
    ``EncoderConfig.frozen_embedding_source`` for provenance) and cached in
    the same on-disk contract ``RNACache`` already reads (``storage.py``) --
    only ``input_dim`` differs from the reference 25017-gene width, which is
    why this always routes through ``FeatureFusionArchitectureVariantModel``
    rather than the fixed-25017 reference classes. Structurally identical to
    ``LinearRNAEncoder`` (LayerNorm + one Linear projection, no other trainable
    capacity); kept as its own class so the encoder registry documents, by
    name, that its input is a frozen embedding rather than expression, and so
    future frozen foundation-model arms (Geneformer, scGPT -- see
    docs/RNA_METHYLATION.md's "Forward direction") can share this one class
    via ``frozen_embedding_source`` rather than each needing a bespoke encoder.
    """

    def __init__(self, input_dim: int, latent_dim: int, layer_norm: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim) if layer_norm else nn.Identity()
        self.projection = nn.Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> RNARepresentation:
        return RNARepresentation(self.projection(self.norm(x)))


def build_rna_encoder(
    config: "EncoderConfig",
    input_dim: int,
    locus_dim: int,
    *,
    mean_query_dim: int | None = None,
    query_source: str = "ntv3",
) -> nn.Module:
    """``EncoderConfig.kind`` dispatch. ``linear`` is the reference encoder; the
    rest are architecture-novelty ablations reachable only via
    ``FeatureFusionArchitectureVariantModel``."""
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
            mean_query_dim=mean_query_dim, query_source=query_source,
        )
    if config.kind == "gene_pathway":
        return GenePathwayEncoder(
            input_dim=input_dim, latent_dim=config.latent_dim,
            membership_path=config.pathway_membership_path,
            dim1=config.pathway_dim1, dim2=config.pathway_dim2,
            dropout=config.dropout, layer_norm=config.layer_norm,
        )
    if config.kind == "bottleneck_mlp":
        return BottleneckMLPEncoder(
            input_dim=input_dim, latent_dim=config.latent_dim, hidden_dim=config.hidden_dim,
            n_blocks=config.n_blocks, mlp_ratio=config.mlp_ratio, dropout=config.dropout,
            layer_norm=config.layer_norm,
        )
    if config.kind == "frozen_embedding":
        return FrozenEmbeddingEncoder(input_dim=input_dim, latent_dim=config.latent_dim, layer_norm=config.layer_norm)
    raise ValueError(f"unknown encoder.kind: {config.kind!r}")


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

    This is the arm that separates "depth helped" from "the reference depth-1
    head is enough" -- see the architecture-novelty suite's `trunk_plain_dX`
    arms.
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
        or bool(config.functional_fusion_variant)
    )

class FeatureFusionArchitectureVariantModel(nn.Module):
    """Configurable extension of ``FeatureFusionLocusCLSModel`` for the
    architecture-novelty suite (``architecture_novelty_2026_09``), retargeted
    2026-09-04 after ``FeatureFusionLocusCLSModel`` was selected as the repo's
    primary/reference architecture (``shared_backbone_locus_cls_2026_09``,
    ``docs/RNA_METHYLATION.md``).

    Same constructor signature as ``FeatureFusionLocusCLSModel`` (drop-in for
    ``LocusCLSJointTrainer``), same two branches -- a locus-only mean branch
    (``CpGTrunk``) and an RNA-conditioned raw branch -- but three axes become
    configurable through ``ModelConfig``:

      * ``config.encoder`` -- the RNA branch (``linear`` canonical,
        ``mlp``/``program_bottleneck`` capacity controls, ``locus_attention``
        makes the raw branch's RNA representation locus-specific).
      * ``config.trunk`` -- optional extra depth after the two branch
        embeddings are concatenated (``trunk.kind="none"``: straight to the
        fusion head, the reference default; ``trunk.kind="plain"``: the
        concatenated joint representation is deepened by ``trunk.depth``
        pre-norm residual blocks first). Isolates whether depth alone helps
        beyond the reference's single fusion `Linear`.
      * ``config.axial`` / ``config.beta_likelihood_head`` -- unchanged from the
        two-stage suite's versions (windowed CpG-axis attention on the final
        joint representation; an additional Beta-concentration output head).

    A fourth axis, ``include_raw_rna``/``include_raw_cpg`` (constructor kwargs,
    not part of ``ModelConfig`` -- same convention as ``use_raw_product``),
    controls which raw pieces feed the raw branch's joint input independently
    of the product term. Combined with ``use_mean_branch=False`` these
    reproduce the paper-required simplified baselines
    (docs/PAPER_EXPERIMENTS.md) under this engine -- see the constructor's
    docstring-in-code for the exact per-baseline settings.

    At every default (``encoder.kind="linear"``, ``trunk.kind="none"``,
    ``axial.enabled=False``, no beta head) this reduces to
    ``FeatureFusionLocusCLSModel``'s exact forward computation -- verified
    numerically in tests -- so it is a valid same-code control for the
    reference architecture.

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
        use_raw_product: bool = True,
        include_raw_rna: bool = True,
        include_raw_cpg: bool = True,
        product_mlp: bool = False,
        fusion_init_std: float = 0.01,
        query_source: str = "ntv3",
        functional_conditioning: bool = False,
        functional_only: bool = False,
    ):
        super().__init__()
        self.config = config
        self.use_mean_branch = bool(use_mean_branch)
        self.query_source = str(query_source)
        # Ablation-only toggle (default True preserves every existing config/checkpoint
        # byte-for-byte): drops the raw branch's elementwise RNA*CpG product term
        # p_{s,l} = W_r r_{s,l} ⊙ W_e e_l, isolating whether that multiplicative
        # interaction -- inherited unconditionally from FeatureFusionLocusCLSModel --
        # contributes anything on top of locus-conditioned attention + plain concat.
        self.use_raw_product = bool(use_raw_product)
        # Ablation-only toggles (both default True, preserving every existing
        # config/checkpoint byte-for-byte): drop the raw branch's raw RNA and/or
        # raw CpG pieces from its joint input, independently of whether the
        # product term is included. With use_mean_branch=False these reproduce
        # the paper-required simplified baselines (docs/PAPER_EXPERIMENTS.md)
        # under this engine instead of the retired two-stage architecture's
        # GlobalShiftInteraction/BilinearInteraction:
        #   Global RNA Shift: include_raw_cpg=False, use_raw_product=False
        #     (joint = [rna] only -- ignores the CpG embedding entirely).
        #   Bilinear RNA-CpG: include_raw_rna=False, include_raw_cpg=False
        #     (joint = [product] only -- "shared latent space, interaction via
        #     dot product" with no raw pieces alongside it).
        #   MLP RNA-CpG: use_raw_product=False (joint = [rna, cpg], no product).
        # At least one of the three raw-branch pieces must remain enabled.
        self.include_raw_rna = bool(include_raw_rna)
        self.include_raw_cpg = bool(include_raw_cpg)
        if not (self.include_raw_rna or self.include_raw_cpg or self.use_raw_product):
            raise ValueError(
                "at least one of include_raw_rna/include_raw_cpg/use_raw_product must be true"
            )
        # Default False preserves every existing checkpoint's state-dict keys/shapes
        # byte-for-byte (rna_product/locus_product stay plain nn.Linear). True upgrades
        # both product-term projections to a small 2-layer MLP (Linear->GELU->Linear,
        # same in/out shape) -- the locus_attention_2026_09_05 reference recipe's choice:
        # every non-RNA-encoder Linear that isn't a final 1-unit output head becomes a
        # small MLP, matching CpGTrunk's existing shape. Meaningless when
        # use_raw_product=False (no product projections exist to upgrade).
        self.product_mlp = bool(product_mlp)
        self.use_fusion_product = bool(use_fusion_product)
        trunk_cfg, axial_cfg = config.trunk, config.axial

        if self.use_mean_branch:
            self.trunk_cpg = CpGTrunk(cpg_input_dim, trunk_hidden_dim, bottleneck_dim, trunk_dropout)
            self.mean_head = nn.Linear(bottleneck_dim, 1)

        encoder_cfg, interaction_cfg = config.encoder, config.interaction
        if self.query_source != "ntv3" and encoder_cfg.kind != "locus_attention":
            raise ValueError("non-NTv3 query_source requires encoder.kind='locus_attention'")
        if self.query_source != "ntv3" and not self.use_mean_branch:
            raise ValueError("mean-aware query_source requires use_mean_branch=true")
        self.rna_encoder = build_rna_encoder(
            encoder_cfg, input_dim=input_dim, locus_dim=cpg_input_dim,
            mean_query_dim=bottleneck_dim if self.use_mean_branch else None,
            query_source=self.query_source,
        )
        self.locus_conditioned = encoder_cfg.kind == "locus_attention"
        rna_pair_dim = encoder_cfg.program_dim if self.locus_conditioned else encoder_cfg.latent_dim
        raw_hidden_dim = interaction_cfg.hidden_dim
        dropout = interaction_cfg.dropout
        joint_dim = (rna_pair_dim if self.include_raw_rna else 0) + (cpg_input_dim if self.include_raw_cpg else 0)
        if self.use_raw_product:
            product_dim = min(rna_pair_dim, cpg_input_dim)
            if self.product_mlp:
                self.rna_product = nn.Sequential(
                    nn.Linear(rna_pair_dim, product_dim), nn.GELU(), nn.Linear(product_dim, product_dim),
                )
                self.locus_product = nn.Sequential(
                    nn.Linear(cpg_input_dim, product_dim), nn.GELU(), nn.Linear(product_dim, product_dim),
                )
            else:
                self.rna_product = nn.Linear(rna_pair_dim, product_dim)
                self.locus_product = nn.Linear(cpg_input_dim, product_dim)
            joint_dim += product_dim
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
            fusion_dim = raw_hidden_dim + (bottleneck_dim if self.use_mean_branch else 0)
            self.combine_stage = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, width))
            if trunk_kind == "plain":
                self.trunk = PlainTrunk(
                    width=width, depth=trunk_cfg.depth, expansion=trunk_cfg.expansion,
                    dropout=trunk_cfg.dropout, gradient_checkpointing=trunk_cfg.gradient_checkpointing,
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

        # Constructed after every pre-existing module so enabling this optional
        # residual cannot perturb initialization of parameters shared with the
        # control arm under the same seed.
        self.functional_conditioning = bool(functional_conditioning)
        self.functional_only = bool(functional_only)
        if self.functional_only and not self.functional_conditioning:
            raise ValueError("functional_only requires functional_conditioning")
        if self.functional_conditioning:
            cpu_rng_state = torch.get_rng_state()
            try:
                self.functional_peak_embedding = nn.EmbeddingBag(
                    4165, 64, mode="mean", include_last_offset=True,
                )
                self.functional_dense_encoder = nn.Sequential(
                    nn.LayerNorm(23), nn.Linear(23, 64), nn.GELU(),
                )
                self.functional_fusion = nn.Sequential(
                    nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU(),
                )
                self.functional_delta = nn.Linear(64, cpg_input_dim, bias=False)
                if not self.functional_only:
                    nn.init.zeros_(self.functional_delta.weight)
            finally:
                # Keep subsequent dropout/data-order RNG identical to the
                # control arm as well as preserving common initialization.
                torch.set_rng_state(cpu_rng_state)

    @property
    def requires_cpg_positions(self) -> bool:
        return self.axial is not None

    def diagnostics(self) -> dict[str, float]:
        diagnostics: dict[str, float] = {}
        if self.locus_conditioned and hasattr(self.rna_encoder, "query_diagnostics"):
            diagnostics.update(self.rna_encoder.query_diagnostics())
        if self.trunk is not None and hasattr(self.trunk, "composite_gain"):
            diagnostics.update(self.trunk.composite_gain())
        return diagnostics

    def forward(
        self, rna: torch.Tensor, cpg_embedding: torch.Tensor, cpg_positions: torch.Tensor | None = None,
        functional_track_indices: torch.Tensor | None = None,
        functional_offsets: torch.Tensor | None = None,
        functional_dense: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.functional_conditioning:
            if functional_track_indices is None or functional_offsets is None or functional_dense is None:
                raise ValueError("functional conditioning requires track indices, offsets, and dense features")
            peak_pattern = self.functional_peak_embedding(functional_track_indices, functional_offsets)
            dense_context = self.functional_dense_encoder(functional_dense)
            functional = self.functional_fusion(torch.cat([peak_pattern, dense_context], dim=-1))
            functional_embedding = self.functional_delta(functional)
            cpg_embedding = functional_embedding if self.functional_only else cpg_embedding + functional_embedding
        representation = self.rna_encoder(rna)
        batch = rna.shape[0]
        n_loci = cpg_embedding.shape[0]

        # Compute the mean-proxy representation before cross-attention so the
        # query experiment can reuse exactly the representation already shaped
        # by the auxiliary locus-mean task. The scalar mu_logit still never
        # enters beta_hat directly.
        if self.use_mean_branch:
            h_mean = self.trunk_cpg(cpg_embedding)  # (n_loci, bottleneck_dim)
            mu_logit = self.mean_head(h_mean).squeeze(-1)  # auxiliary probe only
            h_mean_b = h_mean[None, :, :].expand(batch, n_loci, -1)
        else:
            h_mean = None
            mu_logit = None
            h_mean_b = None

        if self.locus_conditioned:
            assert representation.program_tokens is not None
            rna_pair = self.rna_encoder.attend(
                representation.program_tokens, cpg_embedding, mean_features=h_mean,
            )
        else:
            rna_pair = representation.global_vector[:, None, :].expand(batch, n_loci, -1)

        pieces = []
        if self.include_raw_rna:
            pieces.append(rna_pair)
        if self.include_raw_cpg:
            pieces.append(cpg_embedding[None, :, :].expand(batch, n_loci, -1))
        if self.use_raw_product:
            product = self.rna_product(rna_pair) * self.locus_product(cpg_embedding)[None, :, :]
            pieces.append(product)
        joint = torch.cat(pieces, dim=-1)
        h_raw = self.raw_branch(joint)  # (batch, n_loci, raw_hidden_dim)
        residual_logit = self.residual_head(h_raw).squeeze(-1)  # auxiliary probe only

        if self.trunk is None:
            pieces = [h_mean_b, h_raw] if self.use_mean_branch else [h_raw]
            if self.use_fusion_product:
                fusion_product = self.mean_fusion_proj(h_mean)[None, :, :] * self.raw_fusion_proj(h_raw)
                pieces.append(fusion_product)
            hidden = torch.cat(pieces, dim=-1)
        else:
            pieces = [h_mean_b, h_raw] if self.use_mean_branch else [h_raw]
            hidden = self.combine_stage(torch.cat(pieces, dim=-1))
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
    topology. Appended to the trainer's ``architecture_label`` so two
    different variants can never resume from each other's ``latest.pt``."""
    parts = [f"enc-{config.encoder.kind}"]
    if config.encoder.kind == "locus_attention":
        parts.append(f"k{config.encoder.n_programs}h{config.encoder.n_heads}d{config.encoder.program_dim}")
    if config.encoder.kind == "gene_pathway":
        parts.append(f"p{config.encoder.pathway_dim1}x{config.encoder.pathway_dim2}")
    if config.encoder.kind == "bottleneck_mlp":
        parts.append(f"n{config.encoder.n_blocks}w{config.encoder.hidden_dim}r{config.encoder.mlp_ratio}")
    if config.encoder.kind == "frozen_embedding" and config.encoder.frozen_embedding_source:
        parts.append(f"src-{config.encoder.frozen_embedding_source}")
    if config.trunk.kind != "none":
        parts.append(f"trunk-{config.trunk.kind}d{config.trunk.depth}w{config.trunk.width}")
    if config.axial.enabled:
        parts.append(f"axial-w{config.axial.window}h{config.axial.n_heads}")
    if config.beta_likelihood_head:
        parts.append("betahead")
    return "_".join(parts)
