"""Run-config loader for the MethylProphet-matched (chr1) benchmark path.

The composite ``RunConfig``/``DataConfig`` shape here is specific to
``MethylProphetTrainer``'s single YAML config file
(configs/benchmark_methylprophet/reference.yaml and friends); the generic
scoped pipeline uses the lighter ``rna_training.config.RNARecipe`` instead,
built directly from the shared dataclasses in ``...config``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ...config import EncoderConfig, InteractionConfig, LossConfig, ModelConfig, TrainingConfig, TrackingConfig


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
    """Data-source block for the MethylProphet-matched (``load_config``) path.

    Only ``clip_beta_epsilon`` is read by ``MethylProphetTrainer`` (it gets
    its actual data via explicit cache paths, not these matrix/table
    locations); the rest is kept because
    ``scripts/benchmark_methylprophet/prepare.py``'s ``build_eval_adapter``
    still emits a ``data:`` block in this exact shape for provenance, and
    existing reference configs already carry it.
    """

    rna: MatrixConfig
    methylation: MatrixConfig
    locus_embeddings: MatrixConfig
    locus_features: TableConfig
    sample_metadata: TableConfig
    cpg_splits: TableConfig
    rna_control: str = "real"  # real|mean|shuffle_global|shuffle_within_cancer|cancer_type_only
    rna_transform: str | None = None
    standardize_rna: bool = True
    clip_beta_epsilon: float = 1e-4
    allow_partial_overlap: bool = False


@dataclass(slots=True)
class RunConfig:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
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
    data_raw["methylation"] = _matrix_config(data_raw["methylation"])
    data_raw["locus_embeddings"] = _matrix_config(data_raw["locus_embeddings"])
    data_raw["locus_features"] = _table_config(data_raw["locus_features"])
    data_raw["sample_metadata"] = _table_config(data_raw["sample_metadata"])
    data_raw["cpg_splits"] = _table_config(data_raw["cpg_splits"])
    data = DataConfig(**data_raw)

    model_raw = dict(raw.get("model", {}))
    retired = sorted(set(model_raw) & {"gate", "anchor_to_mean_rna", "prediction_mode"})
    interaction_raw = dict(model_raw.get("interaction", {}))
    if "include_product" in interaction_raw:
        retired.append("interaction.include_product")
    if retired:
        raise ValueError(
            "retired architecture-ablation fields are not accepted by the canonical model: "
            + ", ".join(retired)
        )
    model = ModelConfig(
        encoder=EncoderConfig(**model_raw.get("encoder", {})),
        interaction=InteractionConfig(**interaction_raw),
        zero_init_residual=model_raw.get("zero_init_residual", True),
        variance_normalized_residual=model_raw.get(
            "variance_normalized_residual", False
        ),
    )

    return RunConfig(
        data=data,
        model=model,
        loss=LossConfig(**raw.get("loss", {})),
        training=TrainingConfig(**raw.get("training", {})),
        tracking=TrackingConfig(**raw.get("tracking", {})),
        output_dir=raw.get("output_dir", "artifacts/train/default"),
        run_name=raw.get("run_name", path.stem),
    )
