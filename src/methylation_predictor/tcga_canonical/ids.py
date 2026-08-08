"""Vectorized id -> position lookups, without materializing a huge Python dict.

The largest registry in the canonical bundle (WGBS, ~23M CpGs) makes a plain
``{id: position}`` dict too memory-heavy (tens of bytes per Python int pair,
~GB scale). Both lookup classes here instead sort the id array once and
resolve queries with ``np.searchsorted`` -- O(n log n) to build, O(m log n)
per vectorized batch query, backed by two int64 arrays.
"""
from __future__ import annotations

import numpy as np


class UniqueIndex:
    """id -> single position, for id arrays known to have no duplicates."""

    def __init__(self, ids: np.ndarray, name: str = "id"):
        ids = np.asarray(ids, dtype=np.int64)
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        if len(sorted_ids) > 1 and bool(np.any(sorted_ids[1:] == sorted_ids[:-1])):
            dupe_mask = np.r_[False, sorted_ids[1:] == sorted_ids[:-1]]
            examples = np.unique(sorted_ids[dupe_mask])[:5].tolist()
            raise ValueError(f"duplicate {name} values, e.g. {examples}")
        self._sorted_ids = sorted_ids
        self._positions = order
        self.name = name

    def __len__(self) -> int:
        return len(self._sorted_ids)

    def contains(self, query_ids) -> np.ndarray:
        query = np.asarray(query_ids, dtype=np.int64)
        if query.size == 0:
            return np.zeros(0, dtype=bool)
        loc = np.clip(np.searchsorted(self._sorted_ids, query), 0, len(self._sorted_ids) - 1)
        return self._sorted_ids[loc] == query

    def positions_of(self, query_ids) -> np.ndarray:
        query = np.asarray(query_ids, dtype=np.int64)
        if query.size == 0:
            return np.zeros(0, dtype=np.int64)
        loc = np.clip(np.searchsorted(self._sorted_ids, query), 0, len(self._sorted_ids) - 1)
        found = self._sorted_ids[loc] == query
        if not bool(found.all()):
            missing = np.unique(query[~found])
            raise KeyError(f"{len(missing)} {self.name} values not found, e.g. {missing[:5].tolist()}")
        return self._positions[loc]


class GroupIndex:
    """id -> all positions sharing that id (e.g. WGBS's one duplicated sample_idx).

    Built as a plain dict; only ever used on small (row-count-sized) axes in
    this codebase, never on the multi-million-row CpG axis.
    """

    def __init__(self, ids: np.ndarray, name: str = "id"):
        ids = np.asarray(ids, dtype=np.int64)
        groups: dict[int, list[int]] = {}
        for position, value in enumerate(ids.tolist()):
            groups.setdefault(value, []).append(position)
        self._groups = {key: np.asarray(value, dtype=np.int64) for key, value in groups.items()}
        self.name = name

    def __len__(self) -> int:
        return len(self._groups)

    def positions_of(self, query_id: int) -> np.ndarray:
        try:
            return self._groups[int(query_id)]
        except KeyError as exc:
            raise KeyError(f"{self.name} value not found: {query_id}") from exc

    def keys(self) -> np.ndarray:
        return np.asarray(sorted(self._groups), dtype=np.int64)
