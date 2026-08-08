"""Configuration objects for canonical RNA-to-DNAm training.

The module deliberately uses plain dataclasses and YAML dictionaries so runs remain
portable on clusters without Hydra/OmegaConf.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class MatrixConfig:
    path: str
    values_key: str
    row_ids_key: str
    col_ids_key: str | None = None


@dataclass(slots=True)
class TableConfig:
    path: str
    id_column: str


@dataclass(slots=True)
class DataConfig:
    rna: MatrixConfig
    methylation: MatrixConfig
    locus_embeddings: MatrixConfig
    locus_features: TableConfig
    sample_metadata: TableConfig
    cpg_splits: TableConfig
    gene_embeddings: MatrixConfig | None = None
    sample_id_column: str = "sample_idx"
    cpg_id_column: str = "cpg_idx"
    cancer_type_column: str = "cancer_type"
    split_column: str = "split"
    prior_column: str = "pred_ntv3_prior"
    embedding_key: str = "embeddings"
    locus_ids_key: str = "cpg_idx"
    variability_between_column: str = "pred_log_var_between"
    variability_within_column: str = "pred_log_var_within"
    rna_control: str = "real"  # real|mean|shuffle_global|shuffle_within_cancer|cancer_type_only
    # Backward compatible: when rna_transform is omitted, standardize_rna=True maps to zscore.
    # Explicit choices: none|zscore|methylprophet_quantile|rank|continuous_rank|continuous_binary.
    rna_transform: str | None = None
    rna_quantile_bins: int = 51
    rna_transform_seed: int = 17
    standardize_rna: bool = True
    # Optional frozen sample embeddings (BulkRNABert, scGPT, Geneformer, ...).
    # They are materialized offline and aligned by sample ID, avoiding repeated FM inference
    # inside the Cartesian CpG x patient training loop.
    pretrained_rna: MatrixConfig | None = None
    pretrained_mode: str = "none"  # none|replace|concat
    standardize_pretrained_rna: bool = True
    # Independent negative control for frozen sample embeddings.  Keeping this
    # separate from rna_control means R5 concat retains the true raw RNA while
    # only the foundation-model embedding is permuted.
    pretrained_control: str = "real"  # real|shuffle_global|shuffle_within_cancer
    pretrained_control_seed: int = 20260731
    clip_beta_epsilon: float = 1e-4
    allow_partial_overlap: bool = False
    # Nested, cancer-type-stratified subsampling of the training *sampling pool*
    # only. Preprocessing (standardizer), validation/test splits,
    # and the gate's variability-tertile calibration always see the full train
    # pool regardless of this fraction -- only which patients get selected into
    # SGD minibatches during train() is affected. fraction=1.0 is a no-op.
    train_sample_fraction: float = 1.0
    train_sample_fraction_seed: int = 20260731
    # Same idea for CpGs, nested/stratified by variability tertile.
    train_cpg_fraction: float = 1.0
    train_cpg_fraction_seed: int = 20260731


@dataclass(slots=True)
class EncoderConfig:
    kind: str = "mlp"  # linear|mlp|...|gene_token_perceiver|global_experts|module_tokens|gene_tokens
    latent_dim: int = 64
    hidden_dims: list[int] = field(default_factory=lambda: [1024, 256])
    dropout: float = 0.1
    input_dropout: float = 0.0
    layer_norm: bool = True
    activation: str = "gelu"  # gelu|silu|relu
    width: int = 512
    num_blocks: int = 2
    expansion_factor: int = 4
    residual_scale_init: float = 1.0
    zero_init_encoder_residual: bool = True
    value_encoding: str = "linear"  # linear|fourier (perceiver)
    fourier_frequencies: int = 8
    token_dim: int = 128  # perceiver token width, or linear_tokens' d (K*d == total capacity)
    num_latents: int = 32  # perceiver latent count, or linear_tokens' K
    num_heads: int = 8
    num_self_attention_blocks: int = 2
    # Structured gene-token encoder (unused by the canonical model).
    # `gene_identity_source` controls only the gene identity signal;
    # expression encoding and Perceiver pooling remain identical across variants.
    gene_identity_source: str = "learned"  # learned|ntv3|ntv3_permuted
    gene_token_fusion: str = "film"  # add|film|concat
    gene_embedding_permutation_seed: int = 20260730
    freeze_gene_embeddings: bool = True
    # CpG-conditioned expert / biological-token experiments.
    num_experts: int = 8
    expert_dim: int = 32
    module_weights_path: str | None = None  # npz/npy, shape [modules, genes]
    module_weights_key: str = "weights"
    gene_embedding_path: str | None = None  # optional npz/npy, shape [genes, token_dim]
    gene_embedding_key: str = "embeddings"


@dataclass(slots=True)
class InteractionConfig:
    kind: str = "bilinear"
    # Only "concat" is accepted by the canonical model (models.py raises otherwise).
    hidden_dim: int = 128
    dropout: float = 0.1
    num_heads: int = 8
    mlp_hidden_dims: list[int] = field(default_factory=lambda: [128, 32])
    expert_temperature: float = 1.0
    token_top_k: int = 128
    token_residual_scale_init: float = 1.0
    # Region-aware adapter (unused by the canonical model). Regional input features
    # are appended to the frozen locus embedding; the first ``base_locus_dim``
    # columns remain bit-identical to the canonical locus input.
    base_locus_dim: int | None = None
    regional_rank: int = 16
    # Width of the appended feature prefix that is identical for every CpG in
    # the same region (projected regional NTv3 mean + region count/span).
    regional_context_dim: int | None = None
    regional_gate_kind: str = "fixed"  # fixed|learned
    regional_gate_hidden_dim: int = 32
    # Index inside the appended regional feature block. -1 selects the final
    # feature, which the preparation command reserves for the regulatory mask.
    regional_mask_index: int = -1
    regional_residual_scale_init: float = 1.0


@dataclass(slots=True)
class GateConfig:
    kind: str = "variability"  # none|global|locus|variability
    hidden_dim: int = 64
    dropout: float = 0.0
    initial_global_scale: float = 1.0


@dataclass(slots=True)
class ModelConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    anchor_to_mean_rna: bool = True
    zero_init_residual: bool = True


@dataclass(slots=True)
class LossConfig:
    beta_mse_weight: float = 1.0
    beta_huber_weight: float = 0.0
    beta_huber_delta: float = 0.05
    residual_huber_weight: float = 0.1
    residual_huber_delta: float = 1.0
    shrinkage_weight: float = 1e-4
    beta_macro_weight: float = 0.0
    # MAS-PCC-oriented objectives. Correlations are computed across samples for
    # every CpG in the Cartesian minibatch. Disabled by default so all existing
    # experiments remain numerically unchanged.
    locus_pearson_weight: float = 0.0
    locus_lower_tail_weight: float = 0.0
    locus_lower_tail_fraction: float = 0.60
    locus_min_observed_samples: int = 8
    locus_pearson_epsilon: float = 1e-8
    # Pairwise sample differences remove the locus-static prior exactly and force
    # the dynamic branch to preserve inter-patient direction and amplitude.
    pairwise_difference_weight: float = 0.0
    pairwise_huber_delta: float = 0.05
    pairwise_pairs_per_batch: int = 512
    # Quantitative biological-fidelity loss objectives.
    # The global ratio keeps absolute accuracy competitive with the fixed prior,
    # while the locus terms prevent that global objective from being dominated by
    # easy/static CpGs.
    global_prior_ratio_weight: float = 0.0
    global_prior_ratio_epsilon: float = 1e-8
    global_prior_ratio_clip: float = 4.0
    locus_skill_weight: float = 0.0
    locus_ccc_weight: float = 0.0
    within_cancer_dynamic_weight: float = 0.0
    locus_min_target_std: float = 0.05
    locus_skill_denominator_epsilon: float = 1e-4
    locus_skill_loss_clip: float = 4.0
    within_cancer_min_samples: int = 4
    # Optional deterministic MSE warm-up before the configured objective becomes
    # active. This supports a fair P5 test without chaining checkpoints between
    # grid jobs. Epochs are one-indexed; 0 disables the schedule.
    objective_warmup_epochs: int = 0
    warmup_beta_mse_weight: float = 1.0
    warmup_beta_huber_weight: float = 0.0
    warmup_residual_huber_weight: float = 0.1
    warmup_shrinkage_weight: float = 1e-4
    # Linearly interpolate every non-warmup objective (including the biological-
    # fidelity terms above) from zero to its configured weight during these
    # epochs, via scheduled_loss_config. 0 keeps the historical hard switch. This
    # is independent of the separate ramp_epochs/ramp_start_epoch mechanism below,
    # which ramps warmup_locus_pearson_weight/etc instead.
    objective_ramp_epochs: int = 0
    # Value-space, scale-sensitive complement to the (scale-invariant) Pearson
    # terms above. Per-locus centering removes exactly what a locus-static prior
    # already explains, but unlike locus_pearson_weight, squared error on the
    # centred residual grows monotonically if the model shrinks amplitude toward
    # zero -- it cannot be gamed by amplitude collapse the way raw correlation can.
    centered_mse_weight: float = 0.0
    # Direct regularizer on the per-locus predicted/true amplitude ratio
    # (std(pred_centred) / std(true_centred)), pushed toward 1. Targets the same
    # failure mode as centered_mse_weight from the opposite direction: an
    # explicit penalty on amplitude collapse rather than an implicit one.
    amplitude_weight: float = 0.0
    amplitude_epsilon: float = 1e-6
    # Gradual linear ramp of the objective between the
    # warmup_* weights and the configured (target) weights, instead of
    # scheduled_loss_config's instant switch. ramp_epochs <= 0 disables the ramp
    # entirely (trainer falls back to scheduled_loss_config / objective_warmup_epochs).
    # lambda(e) = warmup + (target - warmup) * clamp((e - ramp_start_epoch) / ramp_epochs, 0, 1)
    ramp_start_epoch: int = 0
    ramp_epochs: int = 0
    warmup_locus_pearson_weight: float = 0.0
    warmup_locus_lower_tail_weight: float = 0.0
    warmup_pairwise_difference_weight: float = 0.0
    warmup_centered_mse_weight: float = 0.0
    warmup_amplitude_weight: float = 0.0


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
class EvaluationConfig:
    sample_chunk_size: int = 128
    cpg_chunk_size: int = 256
    max_cpgs_per_panel: int | None = None
    save_predictions: bool = False
    prediction_format: str = "npz"
    panels: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            "in_distribution": {"sample_split": "train", "cpg_split": "train"},
            "sample_ood": {"sample_split": "test", "cpg_split": "train"},
            "locus_ood": {"sample_split": "train", "cpg_split": "test"},
            "double_ood": {"sample_split": "test", "cpg_split": "test"},
        }
    )


@dataclass(slots=True)
class TrackingConfig:
    backend: str = "none"  # none|wandb
    project: str = "MethylationPredictor"
    entity: str | None = None
    group: str | None = None
    name: str | None = None
    job_type: str = "train"
    tags: list[str] = field(default_factory=list)
    mode: str = "online"  # online|offline|disabled
    log_every_steps: int = 25
    watch_model: bool = False
    log_checkpoint: bool = True


@dataclass(slots=True)
class RunConfig:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    output_dir: str = "artifacts/train/default"
    run_name: str = "train"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _matrix_config(value: dict[str, Any]) -> MatrixConfig:
    return MatrixConfig(**value)


def _table_config(value: dict[str, Any]) -> TableConfig:
    return TableConfig(**value)


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a mapping: {path}")

    data_raw = dict(raw["data"])
    data_raw["rna"] = _matrix_config(data_raw["rna"])
    if data_raw.get("pretrained_rna") is not None:
        data_raw["pretrained_rna"] = _matrix_config(data_raw["pretrained_rna"])
    data_raw["methylation"] = _matrix_config(data_raw["methylation"])
    data_raw["locus_embeddings"] = _matrix_config(data_raw["locus_embeddings"])
    data_raw["locus_features"] = _table_config(data_raw["locus_features"])
    data_raw["sample_metadata"] = _table_config(data_raw["sample_metadata"])
    data_raw["cpg_splits"] = _table_config(data_raw["cpg_splits"])
    if data_raw.get("gene_embeddings") is not None:
        data_raw["gene_embeddings"] = _matrix_config(data_raw["gene_embeddings"])
    data = DataConfig(**data_raw)

    model_raw = dict(raw.get("model", {}))
    model = ModelConfig(
        encoder=EncoderConfig(**model_raw.get("encoder", {})),
        interaction=InteractionConfig(**model_raw.get("interaction", {})),
        gate=GateConfig(**model_raw.get("gate", {})),
        anchor_to_mean_rna=model_raw.get("anchor_to_mean_rna", True),
        zero_init_residual=model_raw.get("zero_init_residual", True),
    )

    return RunConfig(
        data=data,
        model=model,
        loss=LossConfig(**raw.get("loss", {})),
        training=TrainingConfig(**raw.get("training", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        tracking=TrackingConfig(**raw.get("tracking", {})),
        output_dir=raw.get("output_dir", "artifacts/train/default"),
        run_name=raw.get("run_name", path.stem),
    )


def save_config(config: RunConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config.as_dict(), sort_keys=False))
