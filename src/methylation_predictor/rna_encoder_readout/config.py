"""Configuration schema for the frozen-encoder readout search."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TokenCacheConfig:
    path: str
    layers: list[int] = field(default_factory=lambda: [2])
    tokens_key_template: str = "tokens_layer{layer}"
    sample_ids_key: str = "sample_idx"
    gene_ids_key: str = "gene_ids"
    expression_key: str = "expression_log2p1"
    normalized_expression_key: str = "normalized_expression"
    token_ids_key: str = "token_ids"
    split_key: str = "split"
    cancer_key: str = "cancer_type"
    augmentation_path: str | None = None


@dataclass
class RNAConfig:
    path: str
    values_key: str = "X"
    row_ids_key: str = "sample_idx"
    col_ids_key: str = "gene_ids"


@dataclass
class ModelConfig:
    kind: str = "mean"
    primary_layer: int = 2
    output_dim: int = 256
    hidden_dim: int = 128
    num_heads: int = 4
    num_queries: int = 4
    dropout: float = 0.1
    weight_mode: str = "within_variance"
    continuous_mode: str = "none"
    fourier_frequencies: int = 8
    module_membership_path: str | None = None
    perceiver_self_attention_layers: int = 1


@dataclass
class ObjectiveConfig:
    target_gene_count: int = 4096
    target_selection: str = "stratified_variance"
    selection_seed: int = 17
    total_weight: float = 1.0
    within_weight: float = 1.0
    consistency_weight: float = 0.0
    input_target_overlap_allowed: bool = False


@dataclass
class TrainingConfig:
    seed: int = 17
    batch_size: int = 16
    num_workers: int = 0
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    gradient_clip_norm: float = 1.0
    patience: int = 8
    min_delta: float = 1e-5
    device: str = "auto"
    mixed_precision: bool = True
    warm_start_checkpoint: str | None = None
    warm_start_mode: str = "decoder_only"
    seed_initial_checkpoint: bool = True
    decoder_mode: str = "sgd"
    ridge_refit_interval: int = 1
    ridge_alpha_total: float = 1.0
    ridge_alpha_within: float = 1.0


@dataclass
class ReadoutConfig:
    run_name: str
    output_dir: str
    token_cache: TokenCacheConfig
    rna: RNAConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        if not self.run_name:
            raise ValueError("run_name must not be empty")
        if not self.token_cache.layers:
            raise ValueError("token_cache.layers must contain at least one layer")
        if self.model.primary_layer not in self.token_cache.layers:
            raise ValueError("model.primary_layer must be present in token_cache.layers")
        if self.model.output_dim <= 0 or self.model.hidden_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if self.model.num_queries <= 0 or self.model.num_heads <= 0:
            raise ValueError("num_queries and num_heads must be positive")
        if self.model.output_dim % self.model.num_heads != 0:
            raise ValueError("output_dim must be divisible by num_heads")
        if self.objective.target_gene_count <= 0:
            raise ValueError("target_gene_count must be positive")
        if self.objective.target_selection != "stratified_variance":
            raise ValueError("only target_selection=stratified_variance is currently supported")
        if self.training.batch_size <= 0 or self.training.epochs < 0:
            raise ValueError("invalid training schedule")
        if self.training.warm_start_mode not in {"decoder_only", "compatible", "full"}:
            raise ValueError("warm_start_mode must be decoder_only, compatible or full")
        if self.training.decoder_mode not in {"sgd", "ridge_alternating"}:
            raise ValueError("decoder_mode must be sgd or ridge_alternating")
        if self.training.ridge_refit_interval <= 0:
            raise ValueError("ridge_refit_interval must be positive")
        if self.model.kind in {"pma", "perceiver_lite", "module_pma"}:
            if self.model.output_dim % self.model.num_queries != 0:
                raise ValueError("PMA output_dim must be divisible by num_queries")
            if (self.model.output_dim // self.model.num_queries) % self.model.num_heads != 0:
                raise ValueError("PMA latent_dim must be divisible by num_heads")
        if self.objective.consistency_weight > 0 and not self.token_cache.augmentation_path:
            raise ValueError("consistency_weight > 0 requires token_cache.augmentation_path")
        valid_kinds = {
            "mean", "mean_std", "variance_weighted", "learned_gene_weights",
            "gated_attentive_stats", "pma", "scalar_mix", "concat_layer_means",
            "perceiver_lite", "module_pma",
        }
        if self.model.kind not in valid_kinds:
            raise ValueError(f"unsupported model.kind={self.model.kind!r}; choose from {sorted(valid_kinds)}")
        if self.model.kind in {"scalar_mix", "concat_layer_means"} and len(self.token_cache.layers) < 2:
            raise ValueError(f"{self.model.kind} requires at least two cached layers")
        if self.model.kind == "module_pma" and not self.model.module_membership_path:
            raise ValueError("module_pma requires model.module_membership_path")


def _construct(cls: type[Any], values: dict[str, Any] | None) -> Any:
    return cls(**(values or {}))


def load_config(path: str | Path) -> ReadoutConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    config = ReadoutConfig(
        run_name=str(raw["run_name"]),
        output_dir=str(raw["output_dir"]),
        token_cache=_construct(TokenCacheConfig, raw.get("token_cache")),
        rna=_construct(RNAConfig, raw.get("rna")),
        model=_construct(ModelConfig, raw.get("model")),
        objective=_construct(ObjectiveConfig, raw.get("objective")),
        training=_construct(TrainingConfig, raw.get("training")),
    )
    config.validate()
    return config
