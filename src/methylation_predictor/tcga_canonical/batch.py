"""Model-facing batch schema.

Nothing in these dataclasses (or in anything that builds them) leaks an
HDF5 path, an Array/EPIC/WGBS distinction beyond the `source` label, a
MethylProphet split file, or a per-source local column index. The model
consumes `RNAFeatures`, `sample_idx`, `cpg_idx` (always the official global
namespace), `beta`, `source`, `measurement_idx` -- flat, aligned arrays,
one row per (measurement, CpG) training pair.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RNAFeatures:
    """Raw RNA expression, aligned to `sample_idx`, gene order = source `gene_ids`."""

    values: np.ndarray  # (n_pairs, n_genes) float32
    gene_ids: np.ndarray  # (n_genes,)


@dataclass
class TrainingBatch:
    rna: RNAFeatures
    sample_idx: np.ndarray  # (n_pairs,) int64 -- TCGA patient id
    cpg_idx: np.ndarray  # (n_pairs,) int64 -- official MethylProphet global CpG id
    beta: np.ndarray  # (n_pairs,) float32 -- always finite (sampler drops NaNs)
    source: np.ndarray  # (n_pairs,) object -- "array" | "epic" | "wgbs"
    measurement_idx: np.ndarray  # (n_pairs,) int64 -- per-source row id, != sample_idx

    def __len__(self) -> int:
        return len(self.sample_idx)
