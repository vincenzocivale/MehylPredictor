"""Balanced (measurement/sample, CpG) pair sampling for the mix protocols.

## MethylProphet's own "balanced" semantics (source-verified)

MethylProphet's released checkpoint config sets
``sampling_method: balanced`` on its data loader. That string is **not** a
MethylProphet-authored class-balancing routine -- it is passed straight
through to the upstream `mosaicml-streaming` library's
`StreamingDataset(sampling_method=...)` parameter:

    # methylprophet/src/data/dataset.py (xk-huang/methylprophet)
    def create_methylformer_streaming_dataset(
        ..., sampling_method="balanced", sampling_granularity=1, ...
    ):
        dataset = MethylformerStreamingDataset(..., sampling_method=sampling_method, ...)

`mosaicml-streaming`'s own docs define the two possible values precisely:
"balanced" (the default) means that when an epoch's `epoch_size` is smaller
than the underlying (already source-mixed) shard pool, each epoch draws a
*fresh random* subset of that size from the full pool; "fixed" means the
same subset is reused every epoch. In the released MethylProphet configs
`epoch_size` is left unset (commented out in every `*.cfg` this repo could
find), so "balanced" here mainly governs reshuffling behavior across
epochs -- it says nothing about weighting array vs. EPIC vs. WGBS relative
to each other. Any array/EPIC/WGBS mixing ratio in the original training
run was baked in *offline*, at MDS-shard build time, via the mosaicml
`Stream(proportion=...|repeat=...|choose=...)` mechanism -- values not
recoverable from the released checkpoint config.

## What this module actually implements

Given the above, this sampler does not attempt to reproduce a specific
undocumented array:EPIC:WGBS mixing ratio, and never calls one
`methylprophet_exact`. Instead the mixing weight is a named, swappable
*policy* (`Protocol.SOURCE_SAMPLING_POLICIES`, set via a protocol run
config's `source_sampling: {policy: ..., weights: ...}` block):
`explicit_balanced` (default: equal weight per configured source,
independent of each source's pool size -- "equal-source" in the ablation
sense) or `proportional_to_measurements` (weight = pool row count). See
docs/data/METHYLPROPHET_PROTOCOLS.md's "two-level comparison policy":
published-model benchmarking compares on the shared Array evaluation
views only and never depends on this choice; matched-source training
treats the mixing policy as an explicit, cheap ablation axis (equal-source
/ array-heavy / proportional-to-measurements) instead of a single arbitrary
guess. Each draw:

1. picks one source according to the resolved per-source weights
   (renormalized over whichever sources the active protocol includes);
2. samples a Cartesian (rows x CpGs) mini-block from that source's train
   pool without replacement within the block;
3. flattens the block to individual (sample, CpG) pairs and keeps only the
   finite ones -- MethylProphet's own `non_nan` masking; NaNs are never
   imputed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SourceSamplingPool:
    """One source's eligible pool for the sampler to draw (row, CpG) blocks from."""

    name: str
    row_positions: np.ndarray  # HDF5 row positions eligible for training
    sample_idx: np.ndarray  # sample_idx aligned 1:1 with row_positions
    measurement_idx: np.ndarray  # measurement_idx aligned 1:1 with row_positions
    cpg_idx_pool: np.ndarray  # candidate cpg_idx (global namespace)
    weight: float = 1.0

    def __post_init__(self) -> None:
        if len(self.row_positions) == 0:
            raise ValueError(f"source {self.name!r} has an empty row pool")
        if len(self.cpg_idx_pool) == 0:
            raise ValueError(f"source {self.name!r} has an empty CpG pool")
        if not (len(self.row_positions) == len(self.sample_idx) == len(self.measurement_idx)):
            raise ValueError(f"source {self.name!r}: row/sample/measurement pools must be aligned")


class BalancedPairSampler:
    """Draws (source, row_positions, cpg_idx) mini-blocks, source chosen first."""

    def __init__(self, pools: list[SourceSamplingPool], seed: int = 0):
        if not pools:
            raise ValueError("BalancedPairSampler requires at least one source pool")
        self.pools = pools
        self._rng = np.random.default_rng(seed)
        weights = np.array([p.weight for p in pools], dtype=np.float64)
        if np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError("source weights must be non-negative with a positive sum")
        self._probs = weights / weights.sum()

    def choose_source(self) -> SourceSamplingPool:
        index = self._rng.choice(len(self.pools), p=self._probs)
        return self.pools[index]

    def draw_block(
        self, pool: SourceSamplingPool, n_rows: int, n_cpgs: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (row_positions, sample_idx, measurement_idx, cpg_idx) for one
        Cartesian mini-block, sampled without replacement within each axis."""
        n_rows = min(n_rows, len(pool.row_positions))
        n_cpgs = min(n_cpgs, len(pool.cpg_idx_pool))
        row_slots = self._rng.choice(len(pool.row_positions), size=n_rows, replace=False)
        cpg_slots = self._rng.choice(len(pool.cpg_idx_pool), size=n_cpgs, replace=False)
        return (
            pool.row_positions[row_slots],
            pool.sample_idx[row_slots],
            pool.measurement_idx[row_slots],
            pool.cpg_idx_pool[cpg_slots],
        )
