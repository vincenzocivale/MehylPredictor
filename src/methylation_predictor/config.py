"""Shared configuration objects for retained MethylPredictor workflows."""
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
class ModelConfig:
    """Paper-facing RNA methylation architecture configuration."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    functional_fusion_variant: str = ""


@dataclass(slots=True)
class LossConfig:
    """Loss terms supported by the live RNA trainer."""

    beta_mse_weight: float = 1.0
    locus_pearson_weight: float = 0.0
    locus_min_observed_samples: int = 8
    locus_pearson_epsilon: float = 1e-8
    locus_pearson_min_target_std: float = 0.0
    sample_pearson_weight: float = 0.0
    sample_pearson_min_observed_cpgs: int = 8
    sample_pearson_epsilon: float = 1e-8
    locus_centered_mse_weight: float = 0.0

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
