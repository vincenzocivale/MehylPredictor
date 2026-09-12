"""Shared training-pool primitives for RNA methylation training."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

_STRUCTURED_LOSS_FIELDS = ("locus_pearson_weight",)


def loss_config_for_source(config, source_name: str, enabled: set[str]):
    if source_name in enabled:
        return config
    return replace(
        config,
        **{name: 0.0 for name in _STRUCTURED_LOSS_FIELDS},
    )


@dataclass(slots=True)
class TrainingPool:
    name: str
    row_positions: np.ndarray
    sample_idx: np.ndarray
    cpg_idx: np.ndarray
