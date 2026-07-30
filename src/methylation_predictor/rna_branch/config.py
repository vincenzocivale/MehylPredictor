"""Configuration objects for RNA-branch experiments.

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
    standardize_rna: bool = True
    clip_beta_epsilon: float = 1e-4
    allow_partial_overlap: bool = False
    # Nested, cancer-type-stratified subsampling of the training *sampling pool*
    # only (Stage E2.5-A). Preprocessing (standardizer), validation/test splits,
    # and the gate's variability-tertile calibration always see the full train
    # pool regardless of this fraction -- only which patients get selected into
    # SGD minibatches during train() is affected. fraction=1.0 is a no-op.
    train_sample_fraction: float = 1.0
    train_sample_fraction_seed: int = 20260731
    # Same idea for CpGs (Stage E2.5-B), nested/stratified by variability tertile.
    train_cpg_fraction: float = 1.0
    train_cpg_fraction_seed: int = 20260731


@dataclass(slots=True)
class EncoderConfig:
    kind: str = "mlp"  # linear|mlp|bottleneck_mlp|linear_residual|gated_residual|perceiver|linear_tokens|gene_token_perceiver
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
    # Structured gene-token encoder (Stage T).  `gene_identity_source` controls
    # only the gene identity signal; expression encoding and Perceiver pooling
    # remain identical across T1/T2/T3.
    gene_identity_source: str = "learned"  # learned|ntv3|ntv3_permuted
    gene_token_fusion: str = "film"  # add|film|concat
    gene_embedding_permutation_seed: int = 20260730
    freeze_gene_embeddings: bool = True


@dataclass(slots=True)
class InteractionConfig:
    kind: str = "bilinear"
    # bilinear|interaction_mlp|multihead_bilinear|between_within|concat|raw_concat|film|
    # film_locus|cross_attention|linear_token_cross_attention|concat_only|product_only|
    # concat_product_linear|gene_token_cross_attention
    hidden_dim: int = 128
    dropout: float = 0.1
    num_heads: int = 8
    mlp_hidden_dims: list[int] = field(default_factory=lambda: [128, 32])


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
    residual_huber_weight: float = 0.1
    residual_huber_delta: float = 1.0
    shrinkage_weight: float = 1e-4
    beta_macro_weight: float = 0.0


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
    patience: int = 15
    min_delta: float = 1e-5
    num_workers: int = 0
    validation_sample_split: str = "validation"
    validation_cpg_split: str = "validation"
    train_sample_split: str = "train"
    train_cpg_split: str = "train"
    checkpoint_metric: str = "mse"
    validation_every: int = 1
    validation_max_cpgs: int | None = 512
    min_epochs: int = 0
    save_every_epoch: bool = False
    cpg_sampling: str = "uniform"  # uniform|balanced_tertiles
    residual_learning_rate: float | None = None
    warm_start_checkpoint: str | None = None
    freeze_backbone_epochs: int = 0


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
class RunConfig:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output_dir: str = "artifacts/rna_branch/screening/default"
    run_name: str = "rna_branch"

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
        output_dir=raw.get("output_dir", "artifacts/rna_branch/screening/default"),
        run_name=raw.get("run_name", path.stem),
    )


def save_config(config: RunConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config.as_dict(), sort_keys=False))
