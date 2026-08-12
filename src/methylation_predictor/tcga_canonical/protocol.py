"""MethylProphet-compatible protocols on top of `TCGACanonicalBundle`.

A `Protocol` is the only thing a training script should need: it resolves
the exact sample/CpG ID sets MethylProphet's released splits use (read
verbatim from `official_training_data/protocols/<name>/`, never
regenerated -- see docs/data/METHYLPROPHET_PROTOCOLS.md) and exposes them
as a `train_dataset()` sampler and `evaluation_views()`. The model layer
never sees an HDF5 path, an Array/EPIC/WGBS distinction beyond `source`, or
a local column index.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .batch import RNAFeatures, TrainingBatch
from .bundle import TCGACanonicalBundle
from .sampler import BalancedPairSampler, SourceSamplingPool

PROTOCOLS_DIRNAME = "protocols"
SOURCE_VIEWS_DIRNAME = "_source_views"

# Named source-mixing policies for `Protocol.train_dataset` / `_source_pools`.
# None of these claims to reproduce MethylProphet's own (unrecoverable, see
# sampler.py) internal MDS mixing ratio -- see
# docs/data/METHYLPROPHET_PROTOCOLS.md's "two-level comparison policy":
# published-model benchmarking never depends on this choice (it evaluates on
# the shared Array evaluation views only); matched-source training uses one
# of these as an explicit, named, and swappable ablation axis.
SOURCE_SAMPLING_POLICIES = ("explicit_balanced", "proportional_to_measurements")

KNOWN_PROTOCOLS = (
    "tcga_array_chr1",
    "tcga_array_epic_chr1",
    "tcga_array_wgbs_chr1",
    "tcga_mix_chr1",
    "tcga_mix_chr123",
    "array_genomewide",
)


def _load_cpg_idx(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.int64)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)["cpg_idx"].to_numpy(dtype=np.int64)
    raise ValueError(f"unsupported CpG index file: {path}")


@dataclass
class EvaluationView:
    """One (sample x CpG) evaluation panel, IDs taken verbatim from the protocol."""

    name: str
    source: str
    sample_idx: np.ndarray
    cpg_idx: np.ndarray


@dataclass
class Protocol:
    name: str
    bundle: TCGACanonicalBundle
    metadata: dict
    sources: tuple[str, ...]
    chromosomes: tuple[str, ...]
    array_train_sample_idx: np.ndarray
    array_val_sample_idx: np.ndarray
    array_train_cpg_idx: np.ndarray
    array_val_cpg_idx: np.ndarray
    auxiliary_cpg_idx: dict[str, np.ndarray] = field(default_factory=dict)  # epic/wgbs -> pool

    # -- training -----------------------------------------------------
    def _build_pools(self) -> list[SourceSamplingPool]:
        """Build one pool per configured source, weight left at the default
        (1.0) -- weights are assigned afterwards by `_resolve_weights`."""
        pools: list[SourceSamplingPool] = []

        if "array" in self.sources:
            source = self.bundle.sources["array"]
            train_mask = np.isin(source.sample_idx, self.array_train_sample_idx)
            pools.append(
                SourceSamplingPool(
                    name="array",
                    row_positions=np.flatnonzero(train_mask).astype(np.int64),
                    sample_idx=source.sample_idx[train_mask],
                    measurement_idx=source.measurement_idx[train_mask],
                    cpg_idx_pool=self.array_train_cpg_idx,
                )
            )
        for name in ("epic", "wgbs"):
            if name not in self.sources or name not in self.auxiliary_cpg_idx:
                continue
            source = self.bundle.sources[name]
            pools.append(
                SourceSamplingPool(
                    name=name,
                    row_positions=np.arange(source.n_rows, dtype=np.int64),
                    sample_idx=source.sample_idx,
                    measurement_idx=source.measurement_idx,
                    cpg_idx_pool=self.auxiliary_cpg_idx[name],
                )
            )
        if not pools:
            raise ValueError(f"protocol {self.name!r}: no configured source resolved to a training pool")
        return pools

    @staticmethod
    def _resolve_weights(
        policy: str, weights: dict[str, float] | None, pools: list[SourceSamplingPool]
    ) -> dict[str, float]:
        if policy == "explicit_balanced":
            # Equal representation per configured source by default -- this is
            # "equal-source" in the ablation sense, not a claim about
            # MethylProphet's own (unrecoverable) internal mixing ratio.
            return dict(weights) if weights else {pool.name: 1.0 for pool in pools}
        if policy == "proportional_to_measurements":
            if weights:
                raise ValueError("proportional_to_measurements computes weights itself; do not pass `weights`")
            return {pool.name: float(len(pool.row_positions)) for pool in pools}
        raise ValueError(f"unknown source_sampling policy {policy!r}; expected one of {SOURCE_SAMPLING_POLICIES}")

    def _source_pools(
        self, policy: str = "explicit_balanced", weights: dict[str, float] | None = None
    ) -> list[SourceSamplingPool]:
        pools = self._build_pools()
        resolved = self._resolve_weights(policy, weights, pools)
        for pool in pools:
            pool.weight = resolved.get(pool.name, 0.0)
        pools = [pool for pool in pools if pool.weight > 0]
        if not pools:
            raise ValueError(f"protocol {self.name!r}: no source has positive sampling weight")
        return pools

    def train_dataset(
        self,
        batch_sample_size: int = 32,
        batch_cpg_size: int = 256,
        seed: int = 0,
        source_sampling_policy: str = "explicit_balanced",
        source_weights: dict[str, float] | None = None,
    ) -> "ProtocolTrainDataset":
        """`source_sampling_policy` is a named, swappable mixing-ratio ablation
        axis (see `SOURCE_SAMPLING_POLICIES` / docs/data/METHYLPROPHET_PROTOCOLS.md),
        not a claim to reproduce MethylProphet's own internal mixing ratio.
        `source_weights` is only meaningful for `explicit_balanced`."""
        pools = self._source_pools(source_sampling_policy, source_weights)
        sampler = BalancedPairSampler(pools, seed=seed)
        return ProtocolTrainDataset(
            bundle=self.bundle,
            sampler=sampler,
            batch_sample_size=batch_sample_size,
            batch_cpg_size=batch_cpg_size,
        )

    # -- evaluation -----------------------------------------------------
    def evaluation_views(self) -> dict[str, EvaluationView]:
        """The three official Array evaluation panels, IDs exact per protocol."""
        return {
            "train_cpg_x_val_sample": EvaluationView(
                "train_cpg_x_val_sample", "array", self.array_val_sample_idx, self.array_train_cpg_idx
            ),
            "val_cpg_x_train_sample": EvaluationView(
                "val_cpg_x_train_sample", "array", self.array_train_sample_idx, self.array_val_cpg_idx
            ),
            "val_cpg_x_val_sample": EvaluationView(
                "val_cpg_x_val_sample", "array", self.array_val_sample_idx, self.array_val_cpg_idx
            ),
        }

    def evaluation_finite_counts(self) -> dict[str, int]:
        source = self.bundle.sources["array"]
        counts = {}
        for view_name, view in self.evaluation_views().items():
            rows = source.rows_of_samples(view.sample_idx)
            counts[view_name] = source.finite_count(rows, view.cpg_idx)
        return counts


@dataclass
class ProtocolTrainDataset:
    """Iterable training batches: sample a Cartesian mini-block, keep finite cells."""

    bundle: TCGACanonicalBundle
    sampler: BalancedPairSampler
    batch_sample_size: int = 32
    batch_cpg_size: int = 256
    max_resample_attempts: int = 8

    def sample_batch(self) -> TrainingBatch:
        """Draw one training batch, retrying the block draw if it happens to
        contain zero finite cells (possible for a small/sparse block)."""
        for _ in range(self.max_resample_attempts):
            pool = self.sampler.choose_source()
            row_positions, sample_idx, measurement_idx, cpg_idx = self.sampler.draw_block(
                pool, self.batch_sample_size, self.batch_cpg_size
            )
            source = self.bundle.sources[pool.name]
            block = source.block(row_positions, cpg_idx)  # (n_rows, n_cpgs), NaNs preserved
            finite_rows, finite_cols = np.nonzero(np.isfinite(block))
            if len(finite_rows):
                break
        else:
            raise RuntimeError(
                f"no finite cells found after {self.max_resample_attempts} draws from source {pool.name!r}"
            )

        pair_sample_idx = sample_idx[finite_rows]
        pair_measurement_idx = measurement_idx[finite_rows]
        pair_cpg_idx = cpg_idx[finite_cols]
        pair_beta = block[finite_rows, finite_cols].astype(np.float32)

        rna_values = self.bundle.rna.rows(pair_sample_idx)
        return TrainingBatch(
            rna=RNAFeatures(values=rna_values, gene_ids=self.bundle.rna.gene_ids),
            sample_idx=pair_sample_idx,
            cpg_idx=pair_cpg_idx,
            beta=pair_beta,
            source=np.full(len(pair_sample_idx), pool.name, dtype=object),
            measurement_idx=pair_measurement_idx,
        )


def load_protocol(name: str, bundle: TCGACanonicalBundle, root: str | Path | None = None) -> Protocol:
    if name not in KNOWN_PROTOCOLS:
        raise ValueError(f"unknown protocol {name!r}; expected one of {KNOWN_PROTOCOLS}")
    protocol_root = Path(root) if root is not None else bundle.root
    protocol_dir = protocol_root / PROTOCOLS_DIRNAME / name
    if not protocol_dir.is_dir():
        raise FileNotFoundError(f"protocol directory not found: {protocol_dir}")
    metadata = json.loads((protocol_dir / "protocol.json").read_text())
    sources = tuple(metadata["sources"])
    chromosomes = tuple(metadata["chromosomes"])

    array_source = bundle.sources["array"]
    train_sample_path = protocol_dir / "array_train_sample_idx.npy"
    val_sample_path = protocol_dir / "array_val_sample_idx.npy"
    if train_sample_path.is_file():
        array_train_sample_idx = np.load(train_sample_path).astype(np.int64)
        array_val_sample_idx = np.load(val_sample_path).astype(np.int64)
    else:
        # No protocol-specific sample split file (e.g. chr123): the sample-axis
        # (patient) split is genome-wide, not chromosome-specific, and is
        # already carried verbatim on the array source's own sample_split field.
        split = array_source.sample_split
        array_train_sample_idx = array_source.sample_idx[split == "train"]
        array_val_sample_idx = array_source.sample_idx[split == "val"]

    train_cpg_path = protocol_dir / "array_train_cpg_idx.npy"
    val_cpg_path = protocol_dir / "array_val_cpg_idx.npy"
    if not train_cpg_path.is_file():
        train_cpg_path = protocol_dir / "array_train_cpg_idx.parquet"
        val_cpg_path = protocol_dir / "array_val_cpg_idx.parquet"
    array_train_cpg_idx = _load_cpg_idx(train_cpg_path)
    array_val_cpg_idx = _load_cpg_idx(val_cpg_path)

    chrom_tag = "chr1" if chromosomes == ("chr1",) else "chr" + "".join(c.removeprefix("chr") for c in chromosomes)
    source_views_dir = protocol_root / PROTOCOLS_DIRNAME / SOURCE_VIEWS_DIRNAME
    auxiliary_cpg_idx: dict[str, np.ndarray] = {}
    for aux_name in ("epic", "wgbs"):
        if aux_name not in sources:
            continue
        view_path = source_views_dir / f"{aux_name}_{chrom_tag}_cpg_idx.npy"
        if view_path.is_file():
            auxiliary_cpg_idx[aux_name] = np.load(view_path).astype(np.int64)

    return Protocol(
        name=name,
        bundle=bundle,
        metadata=metadata,
        sources=sources,
        chromosomes=chromosomes,
        array_train_sample_idx=array_train_sample_idx,
        array_val_sample_idx=array_val_sample_idx,
        array_train_cpg_idx=array_train_cpg_idx,
        array_val_cpg_idx=array_val_cpg_idx,
        auxiliary_cpg_idx=auxiliary_cpg_idx,
    )
