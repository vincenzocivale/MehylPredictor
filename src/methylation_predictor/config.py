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
    # "locus_attention" are architecture-novelty ablations reachable only through
    # ``models.ArchitectureVariantModel`` -- see docs/RNA_METHYLATION.md and
    # results/reference/ablations/architecture_novelty_2026_09/README.md.
    kind: str = "linear"
    # Canonical width of the RNA latent (LinearRNAEncoder output / ProductInteraction
    # rna_dim). A wider value is an architecture-scaling ablation only -- see
    # docs/RNA_METHYLATION.md ablation note before flipping this in a non-experimental recipe.
    latent_dim: int = 256
    layer_norm: bool = True
    # Ablation-only (kind != "linear"): hidden width of the "mlp" encoder.
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

    The canonical model has *no* trunk (``kind="none"``): the joint
    ``[rna, cpg, product]`` vector goes straight through one hidden layer to a
    scalar, i.e. depth 1. Hyper-Connections and mHC are macro-design mechanisms
    for deep residual stacks, so they only have something to act on once a real
    trunk exists -- ``kind="plain"`` at several depths is the mandatory control
    that separates "depth helped" from "stream mixing helped".

    - ``plain``: pre-norm residual blocks, single residual stream.
    - ``hc``:    Hyper-Connections (Zhu et al., ICLR 2025, arXiv:2409.19606) --
                 ``n_streams`` parallel residual streams with unconstrained
                 learnable mixing.
    - ``mhc``:   Manifold-Constrained Hyper-Connections (DeepSeek,
                 arXiv:2512.24880) -- the same, with the residual mapping
                 projected onto the Birkhoff polytope (doubly stochastic) by
                 Sinkhorn-Knopp, restoring the identity-mapping/conservation
                 property that plain HC loses.

    Only the *static* (input-independent) mappings are learned. mHC's dynamic,
    input-dependent mappings would need one n x n matrix per (sample, locus)
    pair -- 262k matrices per array block, each Sinkhorn-normalized -- which is
    not affordable on Cartesian blocks. The paper's own component ablation
    (its Table 1) attributes -0.022 of the -0.027 total loss gap to the
    residual mapping alone, which is exactly the piece kept here.
    """

    kind: str = "none"  # none|plain|hc|mhc
    depth: int = 0
    width: int = 128
    dropout: float = 0.1
    expansion: int = 2
    # HC/mHC only: residual stream width (expansion rate n in the papers).
    n_streams: int = 4
    # mHC only: Sinkhorn-Knopp iterations for the doubly stochastic projection.
    # 20 is the value used in the mHC paper.
    sinkhorn_iters: int = 20
    # mHC/HC only: how strongly the residual mapping is initialized towards the
    # identity (logit scale before the exponent/Sinkhorn). At the default 4.0 with
    # n_streams=4 the projected matrix is ~0.948 on the diagonal and ~0.017 off
    # it -- near-independent streams, but not a hard identity, which would leave
    # the residual mapping with no gradient to move away from.
    identity_init_scale: float = 4.0
    # The novelty arm: instead of n anonymous copies of the residual width, give
    # each stream a modality identity (rna / cpg / product / prior), so the
    # doubly stochastic residual mapping is a mass-conserving *cross-modal
    # exchange* operator that can be read off per depth. Requires n_streams == 4.
    stream_semantics: bool = False
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
    # non-default value routes the run to ``models.ArchitectureVariantModel``
    # instead of the frozen canonical classes (whose parameters must not change
    # -- see CLAUDE.md's model-compatibility note).
    trunk: TrunkConfig = field(default_factory=TrunkConfig)
    axial: AxialConfig = field(default_factory=AxialConfig)
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
