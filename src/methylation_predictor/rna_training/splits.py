"""Inner-development splits that never touch official benchmark validation labels."""
from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ..storage import SortedIndex


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in values], dtype=object)


def stratified_sample_split(
    *, canonical_root: str | Path, sample_ids: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(canonical_root) / "rna" / "tcga_rna_official_full.h5"
    with h5py.File(path, "r") as h:
        ids = np.asarray(h["sample_idx"][...], np.int64)
        if "tissue_name" not in h:
            raise RuntimeError("canonical RNA needs tissue_name for stratified development split")
        labels = _decode(h["tissue_name"][...])
    rows = SortedIndex(ids, "RNA sample ids").positions_of(sample_ids)
    labels = labels[rows]
    train, val = [], []
    rng = np.random.default_rng(seed)
    for label in np.unique(labels):
        group = np.asarray(sample_ids[labels == label], np.int64)
        group = group[rng.permutation(len(group))]
        n_val = max(1, int(round(len(group) * val_fraction))) if len(group) > 1 else 0
        val.extend(group[:n_val].tolist()); train.extend(group[n_val:].tolist())
    train = np.asarray(sorted(train), np.int64); val = np.asarray(sorted(val), np.int64)
    if not len(train) or not len(val) or np.intersect1d(train, val).size:
        raise RuntimeError("invalid inner sample split")
    return train, val


def blocked_cpg_split(
    *, registry: str | Path, cpg_ids: np.ndarray, val_fraction: float, seed: int, block_bp: int = 5_000_000
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(registry)
    chrom_col = next((c for c in ("chrom", "chr", "chromosome") if c in frame.columns), None)
    pos_col = next((c for c in ("pos", "position", "start") if c in frame.columns), None)
    if chrom_col is None or pos_col is None or "cpg_idx" not in frame.columns:
        raise ValueError("registry needs cpg_idx, chromosome and position columns")
    indexed = frame.set_index("cpg_idx").loc[cpg_ids]
    keys = np.asarray([
        f"{chrom}:{int(pos)//block_bp}" for chrom, pos in zip(indexed[chrom_col].astype(str), indexed[pos_col])
    ], dtype=object)
    unique = np.unique(keys)
    values = np.asarray([
        int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big") / 2**64
        for key in unique
    ])
    val_blocks = set(unique[values < val_fraction].tolist())
    mask = np.asarray([x in val_blocks for x in keys], dtype=bool)
    if not mask.any() or mask.all():
        raise RuntimeError("invalid inner genomic-block split")
    return np.asarray(cpg_ids[~mask], np.int64), np.asarray(cpg_ids[mask], np.int64)
