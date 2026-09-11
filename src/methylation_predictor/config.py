"""Shared architecture/loss/training configuration objects.

Used by both the generic scoped pipeline (rna_training/config.py,
cpg_statistics/*) and the MethylProphet-matched benchmark
(benchmark/methylprophet/config.py). Plain dataclasses so runs remain
portable on clusters without Hydra/OmegaConf.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EncoderConfig:
    # "linear" is the canonical single Linear(25017 -> latent_dim) projection and the
    # only kind the frozen canonical model classes accept. "mlp"/"program_bottleneck"/
    # "locus_attention"/"bottleneck_mlp"/"frozen_embedding" are architecture-novelty
    # ablations (an ongoing RNA-encoder-comparison harness, not a closed one-off study)
    # reachable only through ``models.FeatureFusionArchitectureVariantModel`` -- see
    # docs/RNA_METHYLATION.md's "Forward direction" section and
    # results/reference/ablations/architecture_novelty_2026_09/README.md.
    kind: str = "linear"
    # Canonical width of the RNA latent (LinearRNAEncoder output / ProductInteraction
    # rna_dim). A wider value is an architecture-scaling ablation only -- see
    # docs/RNA_METHYLATION.md ablation note before flipping this in a non-experimental recipe.
    latent_dim: int = 256
    layer_norm: bool = True
    # Ablation-only (kind != "linear"): hidden width of the "mlp" encoder, and the
    # per-block width of the "bottleneck_mlp" encoder (default 1024 matches both).
    hidden_dim: int = 1024
    dropout: float = 0.0
    # Ablation-only: number of gene-program units -- the bottleneck width for
    # "program_bottleneck", and the number of program *tokens* for
    # "locus_attention" (where the RNA representation becomes a K-token set that
    # the CpG embedding queries, instead of one locus-invariant vector).
    n_programs: int = 64
    # Ablation-only ("locus_attention"): per-token width and head count of the
    # locus -> gene-program cross attention.
    program_dim: int = 128
    n_heads: int = 4
    # Ablation-only ("bottleneck_mlp"): number of pre-norm residual blocks
    # (``x = x + Linear(GELU(Linear(LayerNorm(x))))``) and the inner expansion ratio
    # of each block's hidden layer (inner width = hidden_dim * mlp_ratio). Defaults
    # (6, 4) reproduce MethylProphet's published RNA-branch architecture (bioRxiv
    # 2025.02.05.636730, github.com/xk-huang/methylprophet, BottleneckMLP
    # "B_6-Wi_1024") -- architecture only, not their preprocessing or fusion; see
    # docs/RNA_METHYLATION.md.
    n_blocks: int = 6
    mlp_ratio: int = 4
    # Ablation-only ("frozen_embedding"): free-text label of which pretrained
    # transcriptome foundation model produced the embedding this encoder adapts
    # (e.g. "bulkrnabert_tcga") -- not read by any code path, only for
    # run/checkpoint provenance and W&B tagging.
    frozen_embedding_source: str = ""
    # Ablation-only ("gene_pathway"): generated sparse gene/pathway membership.
    pathway_membership_path: str = ""
    pathway_dim1: int = 8
    pathway_dim2: int = 16


@dataclass(slots=True)
class InteractionConfig:
    kind: str = "concat"
    hidden_dim: int = 128
    dropout: float = 0.1
    # Ablation-only: rank of BilinearInteraction's low-rank factorization. This was a
    # hardcoded 64 throughout fusion_mechanism_2026_08, which left that arm capacity-
    # starved next to the canonical product term (rank min(rna_dim, locus_dim) = 256).
    # Exposed so that arm can be retested at matched capacity.
    rank: int = 64
    # Ablation-only: heads for CrossAttentionInteraction. The 2026-08 arm was
    # single-head with a sigmoid gate because there was no token axis to pool over
    # (see CrossAttentionInteraction's docstring); kept configurable so the fusion
    # claim can be defended at matched capacity.
    attn_heads: int = 1
    # Canonical (all True): joint MLP input is [rna, cpg, projected_rna * projected_cpg].
    # Each flag is an independent architecture-simplification ablation switch -- setting
    # one to False drops that piece from the MLP's joint input (the product term, when
    # included, is always computed from the *projected* rna/cpg regardless of whether the
    # raw rna/cpg pieces are also included). At least one must stay True. See
    # docs/RNA_METHYLATION.md ablation note before flipping any of these in a
    # non-experimental recipe.
    include_rna: bool = True
    include_cpg: bool = True
    include_product: bool = True


@dataclass(slots=True)
class TrunkConfig:
    """Depth/topology of the post-fusion trunk (architecture-novelty ablation).

    The reference model has *no* trunk (``kind="none"``): the joint
    ``[rna, cpg, product]`` vector goes straight through one hidden layer to a
    scalar, i.e. depth 1. ``kind="plain"`` at several depths is the control
    that isolates whether depth alone helps beyond that single fusion layer.

    A Hyper-Connections/Manifold-Constrained-HC multi-stream trunk kind was
    tried and removed (architecture_novelty_2026_09, see
    `docs/RNA_METHYLATION.md` and CLAUDE.md's "Model compatibility note" --
    judged not worth the added complexity for the measured gain); this
    dataclass no longer has multi-stream fields.
    """

    kind: str = "none"  # none|plain
    depth: int = 0
    width: int = 128
    dropout: float = 0.1
    expansion: int = 2
    gradient_checkpointing: bool = False


@dataclass(slots=True)
class AxialConfig:
    """Windowed attention along the CpG axis of the Cartesian block.

    The trainer's Cartesian blocks are contiguous slices of the source's CpG
    index array, so a block's CpG axis is already a locally ordered stretch of
    the chromosome: attending within it lets a locus borrow evidence from its
    genomic neighbours (co-methylation), which a strictly per-pair model cannot
    represent. Methodological precedent is CpG Transformer (Bioinformatics
    2022), which uses axial attention for single-cell methylome *imputation*;
    here there is no measured methylation in the input at all.

    The sample axis is deliberately not attended over: cross-sample attention
    is transductive and would make the held-out evaluation contestable.
    """

    enabled: bool = False
    n_heads: int = 4
    # Attention is computed inside non-overlapping windows of this many CpGs;
    # full attention over a 16,384-CpG WGBS block is not affordable.
    window: int = 128
    # Learned relative-position bias over log-spaced |i - j| buckets.
    n_distance_buckets: int = 16
    dropout: float = 0.0


@dataclass(slots=True)
class ModelConfig:
    """RNA methylation architecture configuration."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    # Architecture-novelty ablation blocks. Both default to inert, and any
    # non-default value routes the run to
    # ``models.FeatureFusionArchitectureVariantModel`` instead of the reference
    # ``models.FeatureFusionLocusCLSModel``.
    trunk: TrunkConfig = field(default_factory=TrunkConfig)
    axial: AxialConfig = field(default_factory=AxialConfig)
    # Compatibility selector used by the current functional-locus candidates.
    # Paper-facing values are mas_concat_v3_purecontext (J0) and
    # mas_concat_v4_iterative (J1). Historical F/G ladder values are retired.
    functional_fusion_variant: str = ""
    # Emit a per-pair Beta concentration alongside the anchored mean, so
    # LossConfig.beta_nll_weight has something to score. Kept separate from the
    # loss weight so a misconfigured recipe fails loudly instead of silently
    # training with a no-op likelihood term.
    beta_likelihood_head: bool = False
    zero_init_residual: bool = True
    # Canonical model: logit(beta_hat) = logit(mu_i) + sigma_i * raw_delta.
    # False retains only the historical flat-residual compatibility baseline.
    variance_normalized_residual: bool = False
    # Architecture-ablation switch, matched_chr1 engine only (see
    # benchmark/methylprophet/trainer.py): False drops the CpG-statistics
    # prior/anchor entirely -- prediction_logit = raw_delta directly, with no
    # sigma scaling either (there is no anchor to take a standardized residual
    # against). Mutually exclusive with zero_init_residual=True (there is
    # nothing safe to start "at zero" relative to) and with
    # variance_normalized_residual=True. Canonical stays True; see
    # docs/RNA_METHYLATION.md ablation note before flipping this in a
    # non-experimental recipe.
    use_prior_anchor: bool = True


@dataclass(slots=True)
class LossConfig:
    beta_mse_weight: float = 1.0
    residual_huber_weight: float = 0.1
    residual_huber_delta: float = 1.0
    shrinkage_weight: float = 1e-4
    # V1: standardized-residual counterparts of residual_huber_weight/
    # shrinkage_weight above, operating on the model's raw (pre-sigma-scale)
    # output against (true_delta_logit / sigma) instead of the flat
    # delta_logit. Only meaningful when model.variance_normalized_residual=
    # true; a no-op (zero weight) and unused (sigma=None) otherwise.
    standardized_residual_huber_weight: float = 0.0
    standardized_residual_huber_delta: float = 1.0
    standardized_shrinkage_weight: float = 0.0
    # Floor applied to sigma_i when constructing the standardized residual
    # target r = true_delta_logit / max(sigma_i, sigma_min) -- prevents
    # near-constant loci (sigma_i -> 0) from producing an unbounded target.
    sigma_min: float = 0.05
    # MAS-PCC objective: Pearson correlation across samples, independently for
    # every CpG in the Cartesian minibatch. Disabled by default (zero weight).
    locus_pearson_weight: float = 0.0
    locus_min_observed_samples: int = 8
    locus_pearson_epsilon: float = 1e-8
    # Optional target-std eligibility floor for the Pearson objective.
    locus_pearson_min_target_std: float = 0.0
    # Sample-wise (MAC-direction: per-patient, across-CpG) Pearson objective --
    # the H0/H1/H2 sample-wise-Pearson experiment's L_sample_PCC term
    # (FunctionalConcatMASModel, losses.sample_correlation_loss). Disabled by
    # default (zero weight); distinct from locus_pearson_weight above, which
    # correlates in the opposite (per-CpG, across-sample/MAS) direction.
    sample_pearson_weight: float = 0.0
    sample_pearson_min_observed_cpgs: int = 8
    sample_pearson_epsilon: float = 1e-8
    # Within-locus (across-sample-centred) MSE -- forces the loss to reward
    # patient-specific (RNA-driven) deviation from the locus mean rather than
    # mostly the locus-level constant, which plain beta_mse under-weights
    # (diagnosed: target's between-locus variance is ~6x its within-locus
    # variance on chr1 Array). losses.within_locus_centered_mse_loss. Reuses
    # locus_min_observed_samples as its own per-locus observation floor.
    locus_centered_mse_weight: float = 0.0
    # Beta log-likelihood head (architecture-novelty ablation, opt-in). Methylation
    # beta values are bounded in [0, 1] and heteroscedastic -- their variance
    # collapses towards both boundaries -- which a plain MSE ignores. With a
    # nonzero weight the model additionally emits a per-pair concentration
    # ``phi`` and the objective gains ``-log Beta(y | mu*phi, (1-mu)*phi)``,
    # where ``mu`` is the existing anchored prediction. Additive on purpose: the
    # headline MAS-PCC stays comparable to every other arm.
    beta_nll_weight: float = 0.0
    # Targets are clamped into (eps, 1-eps) before the log-likelihood: WGBS beta
    # values hit exactly 0 and 1, where the Beta density diverges.
    beta_nll_epsilon: float = 1e-3
    # Floor on the predicted concentration; below ~2 the Beta density becomes
    # U-shaped and the NLL gradient destabilizes.
    concentration_min: float = 2.0


@dataclass(slots=True)
class TrainingConfig:
    seed: int = 17
    device: str = "auto"
    epochs: int = 100
    steps_per_epoch: int = 250
    sample_batch_size: int = 32
    cpg_batch_size: int = 256
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    scheduler: str = "constant"  # constant|cosine|cosine_warmup
    scheduler_horizon_epochs: int | None = None
    warmup_epochs: float = 0.0
    min_lr_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    amp: bool = True
    amp_dtype: str = "bfloat16"  # float16|bfloat16
    allow_tf32: bool = True
    matmul_precision: str = "high"  # highest|high|medium
    fused_adamw: bool = True
    patience: int = 15
    min_delta: float = 1e-5
    num_workers: int = 0
    # CPU/HDF5 input-pipeline controls. ``legacy_scattered`` preserves the
    # historical element-wise shuffled minibatches; ``contiguous_blocks``
    # shuffles whole blocks while retaining physical locality inside them.
    schedule_layout: str = "legacy_scattered"  # legacy_scattered|contiguous_blocks
    prefetch_depth: int = 2
    prefetch_workers: int = 1
    hdf5_cache_mb: int = 256
    checkpoint_every: int = 1
    compile: bool = False
    compile_mode: str = "default"
    validation_sample_split: str = "validation"
    validation_cpg_split: str = "validation"
    train_sample_split: str = "train"
    train_cpg_split: str = "train"
    checkpoint_metric: str = "mse"
    # Optional biological guardrails for checkpoint selection. A candidate that
    # violates any non-null bound is logged but cannot overwrite best.pt.
    checkpoint_min_global_skill: float | None = None
    checkpoint_min_amplitude_ratio: float | None = None
    checkpoint_max_amplitude_ratio: float | None = None
    validation_every: int = 1
    validation_max_cpgs: int | None = 512
    min_epochs: int = 0
    save_every_epoch: bool = False
    cpg_sampling: str = "uniform"  # uniform|balanced_tertiles|full_coverage
    # "best" (default): best.pt is overwritten only when checkpoint_metric improves
    # (standard early-stopping selection). "final": every validation epoch
    # overwrites best.pt unconditionally and resets the patience counter -- used
    # for a final refit trained for a fixed, externally-chosen epoch count where
    # early stopping must be structurally unreachable (e.g. re-running a
    # development-selected best_epoch on the full training pool without ever
    # touching held-out data again).
    checkpoint_selection: str = "best"  # best|final
    residual_learning_rate: float | None = None
    warm_start_checkpoint: str | None = None
    freeze_backbone_epochs: int = 0
    # True hard freeze (requires_grad_(False)) of the exact parameters loaded by warm
    # start, applied right after loading. Distinct from residual_learning_rate (which
    # only slows the backbone's updates but still lets its gradients enter the shared
    # clip_grad_norm_ call): frozen parameters never accumulate a gradient at all, so
    # they cannot distort the residual branch's effective clipped gradient.
    freeze_warm_start_params: bool = False
    # When True (and warm_start_checkpoint is set), seed best.pt/best_metric from the
    # pre-training validation pass instead of only from epochs actually trained. Lets a
    # residual branch that never improves validation simply "win" with epoch 0 (i.e.
    # reproduce the warm-started baseline exactly) instead of being forced to return
    # whichever trained epoch happened to be least-bad.
    seed_initial_checkpoint: bool = False


@dataclass(slots=True)
class TrackingConfig:
    backend: str = "none"  # none|wandb
    project: str = "MethylPredictor"
    entity: str | None = None
    group: str | None = None
    name: str | None = None
    job_type: str = "train"
    tags: list[str] = field(default_factory=list)
    mode: str = "online"  # online|offline|disabled
    log_every_steps: int = 25
    watch_model: bool = False
    log_checkpoint: bool = True
