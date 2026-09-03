"""Pull the "unseen CpGs, seen patients" masked-recovery view for a scope.

Reuses the existing `val_cpg_x_train_sample` view -- `array_train_sample_idx` paired
against `array_val_cpg_idx` -- already defined by
`benchmark.methylprophet.protocol.Table5Protocol` for chr1 (the matched, split-
verified cache at `derived/methylprophet_table5_tcga_chr1/table5_protocol/`). No new
split logic: this module only adds the genomic-position lookup a foundation-model
adapter needs (CpG row index -> (chrom, pos)) that `Table5Protocol` itself doesn't
carry, plus the vocabulary-overlap plumbing MethylGPT's fixed vocabulary requires.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from methylation_predictor.benchmark.methylprophet.protocol import Table5Protocol


def load_val_cpg_positions_chr1(
    table5_protocol_root: str | Path,
    array_cpg_registry_parquet: str | Path,
) -> list[tuple[str, int]]:
    """The chr1 `val_cpg_x_train_sample` view's held-out CpGs as (chrom, pos) tuples.

    `array_cpg_idx` in `Table5Protocol` is a row position into the canonical Array
    HDF5 matrix, not a genomic coordinate -- `array_cpg_registry_parquet`
    (`cpg/registries/array_cpg_map.parquet`, columns `cpg_idx`/`chr`/`pos`) is the
    crosswalk. Note this repo's registry has no Illumina probe-ID column, so this
    (chrom, pos) list -- not a probe ID list -- is as far as this repo's own data
    gets you toward a MethylGPT vocabulary-overlap check; see `vocabulary_overlap`
    in `verify.py` and this package's docstring in `PAPER_EXPERIMENTS.md`.
    """
    protocol = Table5Protocol.load(table5_protocol_root)
    registry = pd.read_parquet(array_cpg_registry_parquet, columns=["cpg_idx", "chr", "pos"])
    registry = registry.set_index("cpg_idx")

    val_cpg_idx = np.asarray(protocol.array_val_cpg_idx, dtype=np.int64)
    rows = registry.loc[val_cpg_idx]
    return list(zip(rows["chr"].tolist(), rows["pos"].astype(int).tolist()))


def load_train_sample_context_ids_chr1(table5_protocol_root: str | Path) -> np.ndarray:
    """The chr1 `val_cpg_x_train_sample` view's context samples (train-set patients).

    These are the samples visible to a foundation model as context (their train-CpG
    beta values are true, real data); `load_val_cpg_positions_chr1` above gives the
    masked target CpGs for those same samples.
    """
    protocol = Table5Protocol.load(table5_protocol_root)
    return np.asarray(protocol.array_train_sample_idx, dtype=np.int64)
