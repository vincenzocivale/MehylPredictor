"""Pull the "unseen CpGs, seen patients" masked-recovery view for a scope.

Reuses the existing `val_cpg_x_train_sample` view -- `array_train_sample_idx` paired
against `array_val_cpg_idx` -- already defined per-scope under
`.../protocols/tcga_mix_<scope>/` (chr1: the matched, split-verified cache at
`derived/methylprophet_table5_tcga_chr1/table5_protocol/`; chr123: `datasets/
methylprophet_repro_v1/protocols/tcga_mix_chr123/`, CpG axis verified exact against the
released MethylProphet chr123 artifact, sample axis reused from chr1 -- see
`docs/BENCHMARK_METHYLPROPHET.md`'s "chr123: verified" section). No new split logic: this
module only adds the genomic-position lookup a foundation-model adapter needs (CpG row
index -> (chrom, pos)) that these protocol directories don't carry directly, plus the
vocabulary-overlap plumbing MethylGPT's fixed vocabulary requires.

**Reads the 4 core `array_*_idx.npy` files directly, not via
`benchmark.methylprophet.protocol.Table5Protocol`** -- that class's `.load()` also requires
`epic_train_cpg_idx.npy`/`wgbs_train_cpg_idx.npy` (this module never uses either) and its
`validate()` checks chr1-specific expected counts (`TABLE5_EXPECTED`), which a different
scope's protocol directory (e.g. chr123's, with ~78k/~14.9k train/val CpGs vs. chr1's
~33.9k/~6.7k) would always fail. Reading the two files this module actually needs directly
keeps these functions scope-agnostic despite the historical `_chr1` suffix in their names
(kept for call-site compatibility -- see `evaluator.py`, `scripts/benchmark_foundation_models/
{check_crosswalk_coverage,prepare_cpgpt_chr1_context}.py`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_val_cpg_positions_chr1(
    protocol_root: str | Path,
    array_cpg_registry_parquet: str | Path,
) -> list[tuple[str, int]]:
    """A scope's `val_cpg_x_train_sample` view's held-out CpGs as (chrom, pos) tuples.

    `array_val_cpg_idx` (read directly from `protocol_root/array_val_cpg_idx.npy`) is a row
    position into the canonical Array HDF5 matrix, not a genomic coordinate --
    `array_cpg_registry_parquet` (`cpg/registries/array_cpg_map.parquet`, columns
    `cpg_idx`/`chr`/`pos`, genome-wide -- one file covers every scope) is the crosswalk. Note
    this repo's registry has no Illumina probe-ID column, so this (chrom, pos) list -- not a
    probe ID list -- is as far as this repo's own data gets you toward a MethylGPT
    vocabulary-overlap check; see `vocabulary_overlap` in `verify.py` and
    `crosswalk.IlluminaCrosswalk` for the resolution.
    """
    val_cpg_idx = np.load(Path(protocol_root) / "array_val_cpg_idx.npy").astype(np.int64)
    registry = pd.read_parquet(array_cpg_registry_parquet, columns=["cpg_idx", "chr", "pos"])
    registry = registry.set_index("cpg_idx")

    rows = registry.loc[val_cpg_idx]
    return list(zip(rows["chr"].tolist(), rows["pos"].astype(int).tolist()))


def load_train_sample_context_ids_chr1(protocol_root: str | Path) -> np.ndarray:
    """A scope's `val_cpg_x_train_sample` view's context samples (train-set patients).

    These are the samples visible to a foundation model as context (their train-CpG beta
    values are true, real data); `load_val_cpg_positions_chr1` above gives the masked target
    CpGs for those same samples.
    """
    return np.load(Path(protocol_root) / "array_train_sample_idx.npy").astype(np.int64)


def load_train_cpg_positions(
    protocol_root: str | Path,
    array_cpg_registry_parquet: str | Path,
) -> list[tuple[str, int]]:
    """A scope's known/context train CpGs as (chrom, pos) tuples (see module docstring)."""
    train_cpg_idx = np.load(Path(protocol_root) / "array_train_cpg_idx.npy").astype(np.int64)
    registry = pd.read_parquet(array_cpg_registry_parquet, columns=["cpg_idx", "chr", "pos"])
    registry = registry.set_index("cpg_idx")

    rows = registry.loc[train_cpg_idx]
    return list(zip(rows["chr"].tolist(), rows["pos"].astype(int).tolist()))
