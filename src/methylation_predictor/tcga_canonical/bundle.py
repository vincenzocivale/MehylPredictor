"""Lazy, read-only access to the MethylProphet-compatible canonical TCGA bundle.

Everything under ``TCGACanonicalBundle`` treats the on-disk artifacts
(``tcga_rna_official_full.h5``, ``methylation/{tcga_array_official_full,
epic_full,wgbs_full}.h5``, ``registries/*_cpg_map.parquet``) as frozen and
read-only -- see docs/data/TCGA_CANONICAL_DATA.md. Nothing here writes to
the data root or loads a full multi-hundred-million-cell matrix into RAM;
HDF5 datasets stay open and are read in row/column chunks on demand.

``cpg_idx`` everywhere in this module is the official MethylProphet global
CpG namespace (as stored in each source's own ``cpg_idx`` HDF5 dataset).
Per-source HDF5 column positions are an internal implementation detail,
never exposed to callers -- see ``MethylationSource._cpg_index``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .ids import GroupIndex, UniqueIndex

try:
    import h5py
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError("h5py is required for the TCGA canonical bundle") from exc

SOURCE_FILES = {
    "array": "methylation/tcga_array_official_full.h5",
    "epic": "methylation/epic_full.h5",
    "wgbs": "methylation/wgbs_full.h5",
}
RNA_FILE = "tcga_rna_official_full.h5"

EXPECTED_SHAPES = {
    "rna": (10916, 25017),
    "array": (9178, 408399),
    "epic": (1706, 740296),
    "wgbs": (32, 23047052),
}


def _decode(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind in {"S", "O"}:
        return np.asarray(
            [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values.tolist()],
            dtype=object,
        )
    return values


def _read_rows(dataset: "h5py.Dataset", positions: np.ndarray) -> np.ndarray:
    """Read (possibly-duplicated, unordered) row positions with one h5py call."""
    positions = np.asarray(positions, dtype=np.int64)
    if positions.size == 0:
        return np.empty((0, dataset.shape[1]), dtype=np.float32)
    unique_pos, inverse = np.unique(positions, return_inverse=True)
    data = np.asarray(dataset[unique_pos, :], dtype=np.float32)
    return data[inverse]


def _read_cols(dataset: "h5py.Dataset", positions: np.ndarray) -> np.ndarray:
    """Read (possibly-duplicated, unordered) column positions, all rows."""
    positions = np.asarray(positions, dtype=np.int64)
    if positions.size == 0:
        return np.empty((dataset.shape[0], 0), dtype=np.float32)
    unique_pos, inverse = np.unique(positions, return_inverse=True)
    data = np.asarray(dataset[:, unique_pos], dtype=np.float32)
    return data[:, inverse]


@dataclass
class RNASource:
    """The single canonical RNA matrix -- all 25,017 genes, `gene_ids` order."""

    path: Path
    h5: "h5py.File"
    gene_ids: np.ndarray
    sample_index: UniqueIndex

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.h5["X"].shape)

    def has_sample(self, sample_idx) -> np.ndarray:
        return self.sample_index.contains(sample_idx)

    def rows(self, sample_idx: Sequence[int]) -> np.ndarray:
        """Return (n, 25017) float32 RNA rows, raising if any sample_idx is missing.

        Requirement: every measurement a protocol uses must resolve to a real
        RNA row -- this is the enforcement point (KeyError from UniqueIndex).
        """
        positions = self.sample_index.positions_of(sample_idx)
        return _read_rows(self.h5["X"], positions)

    def close(self) -> None:
        self.h5.close()


@dataclass
class MethylationSource:
    """One methylation source (array/epic/wgbs): lazy HDF5 + precomputed maps.

    `sample_idx`/`measurement_idx` are 1:1 with HDF5 rows and are *not* the
    same axis: `sample_idx` is the TCGA patient id (shared with RNA),
    `measurement_idx` is this source's own per-row id (distinct even when
    the same sample_idx appears twice, as WGBS does once).
    """

    name: str
    path: Path
    h5: "h5py.File"
    sample_idx: np.ndarray
    measurement_idx: np.ndarray
    sample_split: np.ndarray | None  # only populated for "array"
    _cpg_index: UniqueIndex
    _measurement_index: UniqueIndex
    _sample_groups: GroupIndex

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.h5["beta"].shape)

    @property
    def n_rows(self) -> int:
        return self.shape[0]

    def _column_major(self) -> bool:
        """WGBS is chunked (32, 8192) -- whole-column-band chunks, cheap to read
        by column. Array/EPIC are chunked one-row-per-chunk -- cheap by row."""
        chunks = self.h5["beta"].chunks
        return chunks is not None and chunks[0] >= self.n_rows

    def has_cpg(self, cpg_idx) -> np.ndarray:
        return self._cpg_index.contains(cpg_idx)

    def rows_of_measurements(self, measurement_idx: Sequence[int]) -> np.ndarray:
        return self._measurement_index.positions_of(measurement_idx)

    def rows_of_sample(self, sample_idx: int) -> np.ndarray:
        """All row positions for one sample_idx (>1 only for WGBS's duplicate)."""
        return self._sample_groups.positions_of(sample_idx)

    def rows_of_samples(self, sample_idx: Sequence[int]) -> np.ndarray:
        """One row position per queried sample_idx; errors on an ambiguous
        (duplicated) sample_idx -- callers that need every row for a
        duplicated sample must use `rows_of_sample` instead."""
        out = np.empty(len(sample_idx), dtype=np.int64)
        for i, value in enumerate(sample_idx):
            positions = self.rows_of_sample(int(value))
            if len(positions) != 1:
                raise ValueError(
                    f"{self.name}: sample_idx={value} maps to {len(positions)} rows, "
                    "use rows_of_sample() for sources with duplicated samples"
                )
            out[i] = positions[0]
        return out

    def block(self, row_positions: Sequence[int], cpg_idx_query: Sequence[int]) -> np.ndarray:
        """Dense (len(rows), len(cpgs)) float32 block, NaNs preserved."""
        cols = self._cpg_index.positions_of(cpg_idx_query)
        rows = np.asarray(row_positions, dtype=np.int64)
        dataset = self.h5["beta"]
        if self._column_major():
            data = _read_cols(dataset, cols)
            return data[rows, :]
        data = _read_rows(dataset, rows)
        return data[:, cols]

    def finite_count(
        self, row_positions: Sequence[int], cpg_idx_query: Sequence[int], row_chunk: int = 256
    ) -> int:
        """Count finite (non-NaN) cells in the (rows x cpgs) view without
        materializing the whole block at once."""
        cols = self._cpg_index.positions_of(cpg_idx_query)
        rows = np.asarray(row_positions, dtype=np.int64)
        dataset = self.h5["beta"]
        if self._column_major():
            data = _read_cols(dataset, cols)
            return int(np.isfinite(data[rows, :]).sum())
        unique_rows = np.unique(rows)
        total = 0
        for start in range(0, len(unique_rows), row_chunk):
            chunk_rows = unique_rows[start : start + row_chunk]
            chunk = np.asarray(dataset[chunk_rows, :], dtype=np.float32)[:, cols]
            total += int(np.isfinite(chunk).sum())
        return total

    def close(self) -> None:
        self.h5.close()


@dataclass
class TCGACanonicalBundle:
    """Root handle for the frozen canonical TCGA bundle.

    Path is resolved from (in order): explicit `root` argument, the
    `TCGA_CANONICAL_ROOT` environment variable, or the caller's config file --
    never hardcoded here. See configs/data/tcga_canonical.yaml.
    """

    root: Path
    rna: RNASource
    sources: dict[str, MethylationSource]

    @classmethod
    def from_root(cls, root: str | Path, validate_shapes: bool = True) -> "TCGACanonicalBundle":
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"canonical bundle root does not exist: {root}")

        rna_path = root / RNA_FILE
        rna_h5 = h5py.File(rna_path, "r")
        rna = RNASource(
            path=rna_path,
            h5=rna_h5,
            gene_ids=_decode(rna_h5["gene_ids"][...]),
            sample_index=UniqueIndex(rna_h5["sample_idx"][...], name="RNA sample_idx"),
        )

        sources: dict[str, MethylationSource] = {}
        for name, relative_path in SOURCE_FILES.items():
            path = root / relative_path
            h5f = h5py.File(path, "r")
            sample_idx = np.asarray(h5f["sample_idx"][...], dtype=np.int64)
            measurement_idx = np.asarray(h5f["measurement_idx"][...], dtype=np.int64)
            sample_split = _decode(h5f["sample_split"][...]) if "sample_split" in h5f else None
            sources[name] = MethylationSource(
                name=name,
                path=path,
                h5=h5f,
                sample_idx=sample_idx,
                measurement_idx=measurement_idx,
                sample_split=sample_split,
                _cpg_index=UniqueIndex(h5f["cpg_idx"][...], name=f"{name} cpg_idx"),
                _measurement_index=UniqueIndex(measurement_idx, name=f"{name} measurement_idx"),
                _sample_groups=GroupIndex(sample_idx, name=f"{name} sample_idx"),
            )

        bundle = cls(root=root, rna=rna, sources=sources)
        if validate_shapes:
            bundle.validate_shapes()
        return bundle

    def validate_shapes(self) -> None:
        actual = {"rna": self.rna.shape, **{name: src.shape for name, src in self.sources.items()}}
        mismatches = {k: (v, EXPECTED_SHAPES[k]) for k, v in actual.items() if v != EXPECTED_SHAPES[k]}
        if mismatches:
            raise ValueError(f"canonical bundle shape mismatch (actual, expected): {mismatches}")

    def close(self) -> None:
        self.rna.close()
        for source in self.sources.values():
            source.close()

    def __enter__(self) -> "TCGACanonicalBundle":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
