"""Compact recipe loader for the refactored RNA training workflow."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..config import EncoderConfig, InteractionConfig, LossConfig, ModelConfig, TrainingConfig, TrackingConfig


@dataclass(slots=True)
class RNARecipe:
    raw: dict[str, Any]
    model: ModelConfig
    loss: LossConfig
    training: TrainingConfig
    tracking: TrackingConfig
    batching: dict[str, dict[str, int]]
    schedule_policy: str
    structured_loss_sources: set[str]
    exclude_official_val_from_auxiliary: bool


def load_rna_recipe(path: str | Path) -> RNARecipe:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    model_raw = dict(raw.get("model", {}))
    interaction_raw = dict(model_raw.get("interaction", {}))
    model = ModelConfig(
        encoder=EncoderConfig(**model_raw.get("encoder", {})),
        interaction=InteractionConfig(**interaction_raw),
        zero_init_residual=bool(model_raw.get("zero_init_residual", True)),
        variance_normalized_residual=bool(model_raw.get("variance_normalized_residual", True)),
    )
    if not model.variance_normalized_residual:
        raise ValueError("refactored RNA workflow requires variance_normalized_residual=true")
    training = TrainingConfig(**raw.get("training", {}))
    batching = raw.get("batching", {})
    defaults = {
        "array": {"sample_size": 512, "cpg_size": 512},
        "epic": {"sample_size": 128, "cpg_size": 4096},
        "wgbs": {"sample_size": 32, "cpg_size": 16384},
    }
    resolved_batching = {name: {**defaults[name], **dict(batching.get(name, {}))} for name in defaults}
    schedule_policy = str(raw.get("schedule_policy", "axis_full_coverage"))
    if schedule_policy not in {"pair_complete", "axis_full_coverage"}:
        raise ValueError("schedule_policy must be pair_complete or axis_full_coverage")
    return RNARecipe(
        raw=raw,
        model=model,
        loss=LossConfig(**raw.get("loss", {})),
        training=training,
        tracking=TrackingConfig(**raw.get("tracking", {})),
        batching=resolved_batching,
        schedule_policy=schedule_policy,
        structured_loss_sources=set(raw.get("structured_loss_sources", ["array", "epic", "wgbs"])),
        exclude_official_val_from_auxiliary=bool(raw.get("exclude_official_val_from_auxiliary", True)),
    )
