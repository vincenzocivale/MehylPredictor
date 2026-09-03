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

from ...config import (
    AxialConfig,
    EncoderConfig,
    InteractionConfig,
    LossConfig,
    ModelConfig,
    TrackingConfig,
    TrainingConfig,
    TrunkConfig,
)


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
    # interaction.include_product was retired here after the
    # interaction_concat_and_latent_dim_2026_08 ablation (chr1, generic engine,
    # inner double-OOD dev split) concluded with no permanent architecture
    # change adopted -- see results/reference/ablations.yaml. It is
    # re-permitted as of the "MLP RNA-CpG" baseline
    # (configs/models/baselines/baseline_mlp_rna_cpg.yaml,
    # docs/PAPER_EXPERIMENTS.md), which deliberately trains that exact
    # architecture-simplification arm on this frozen matched_chr1 path for a
    # formal, permanent paper-comparison number, not a casual re-run.
    if retired:
        raise ValueError(
            "retired architecture-ablation fields are not accepted by the canonical model: "
            + ", ".join(retired)
        )
    # ModelConfig is built field by field on purpose (no **model_raw): an unknown
    # or misspelled key must fail here rather than silently train a different
    # architecture. New architecture-novelty blocks are threaded through the same
    # way -- see docs/RNA_METHYLATION.md and
    # results/reference/ablations/architecture_novelty_2026_09/README.md.
    model = ModelConfig(
        encoder=EncoderConfig(**model_raw.get("encoder", {})),
        interaction=InteractionConfig(**interaction_raw),
        trunk=TrunkConfig(**model_raw.get("trunk", {})),
        axial=AxialConfig(**model_raw.get("axial", {})),
        beta_likelihood_head=model_raw.get("beta_likelihood_head", False),
        zero_init_residual=model_raw.get("zero_init_residual", True),
        variance_normalized_residual=model_raw.get(
            "variance_normalized_residual", False
        ),
        use_prior_anchor=model_raw.get("use_prior_anchor", True),
    )
    unknown_model_keys = sorted(
        set(model_raw)
        - {
            "encoder", "interaction", "trunk", "axial", "beta_likelihood_head",
            "zero_init_residual", "variance_normalized_residual", "use_prior_anchor",
        }
    )
    if unknown_model_keys:
        raise ValueError(f"unknown model configuration keys: {', '.join(unknown_model_keys)}")
    if not model.use_prior_anchor:
        if model.variance_normalized_residual:
            raise ValueError(
                "use_prior_anchor=false is mutually exclusive with variance_normalized_residual=true "
                "(there is no anchor to take a sigma-standardized residual against)"
            )
        if model.zero_init_residual:
            raise ValueError(
                "use_prior_anchor=false requires zero_init_residual=false "
                "(there is no anchor to start safely 'at zero' relative to)"
            )

    loss = LossConfig(**raw.get("loss", {}))
    if loss.beta_nll_weight != 0.0 and not model.beta_likelihood_head:
        raise ValueError(
            "loss.beta_nll_weight is nonzero but model.beta_likelihood_head is false, so no concentration "
            "would be predicted and the term would silently be a no-op"
        )
    if model.beta_likelihood_head and loss.beta_nll_weight == 0.0:
        raise ValueError(
            "model.beta_likelihood_head is true but loss.beta_nll_weight is zero, which trains an unused head"
        )

    return RunConfig(
        data=data,
        model=model,
        loss=loss,
        training=TrainingConfig(**raw.get("training", {})),
        tracking=TrackingConfig(**raw.get("tracking", {})),
        output_dir=raw.get("output_dir", "artifacts/train/default"),
        run_name=raw.get("run_name", path.stem),
    )
