"""Shared training-pool primitives for the RNA-methylation pipeline.

``TrainingPool`` and ``loss_config_for_source`` are reused by
``rna_training.locus_cls_trainer.LocusCLSJointTrainer``, the sole RNA-methylation
trainer (``scripts/train.py``). The earlier scope-general two-stage engine
(``ScopedRNATrainer``) that used to live in this module has been retired
alongside the rest of the frozen-prior + residual architecture generation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

_STRUCTURED_LOSS_FIELDS = ("locus_pearson_weight",)


def loss_config_for_source(config, source_name: str, enabled: set[str]):
    if source_name in enabled:
        return config
    return replace(config, **{name: 0.0 for name in _STRUCTURED_LOSS_FIELDS})


@dataclass(slots=True)
class TrainingPool:
    name: str
    row_positions: np.ndarray
    sample_idx: np.ndarray
    cpg_idx: np.ndarray
