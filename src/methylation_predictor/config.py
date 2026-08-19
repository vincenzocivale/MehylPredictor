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
    kind: str = "linear"
    latent_dim: int = 256
    layer_norm: bool = True


@dataclass(slots=True)
class InteractionConfig:
    kind: str = "concat"
    hidden_dim: int = 128
    dropout: float = 0.1


@dataclass(slots=True)
class ModelConfig:
    """RNA methylation architecture configuration."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    zero_init_residual: bool = True
    # Canonical model: logit(beta_hat) = logit(mu_i) + sigma_i * raw_delta.
    # False retains only the historical flat-residual compatibility baseline.
    variance_normalized_residual: bool = False


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
