from __future__ import annotations

import numpy as np


class SortedIndex:
    """Compact int64 id -> row-position map using sort/searchsorted."""

    def __init__(self, ids: np.ndarray, name: str = "ids") -> None:
        self.ids = np.asarray(ids, dtype=np.int64)
        self.name = name
        self.order = np.argsort(self.ids, kind="mergesort")
        self.sorted_ids = self.ids[self.order]
        if len(self.sorted_ids) and np.any(self.sorted_ids[1:] == self.sorted_ids[:-1]):
            raise ValueError(f"{name} contains duplicate ids")

    def contains(self, query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=np.int64)
        pos = np.searchsorted(self.sorted_ids, query)
        ok = pos < len(self.sorted_ids)
        if len(self.sorted_ids):
            clipped = np.minimum(pos, len(self.sorted_ids) - 1)
            ok &= self.sorted_ids[clipped] == query
        return ok

    def positions_of(self, query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=np.int64)
        pos = np.searchsorted(self.sorted_ids, query)
        if len(self.sorted_ids) == 0:
            raise KeyError(f"{self.name} is empty")
        clipped = np.minimum(pos, len(self.sorted_ids) - 1)
        ok = (pos < len(self.sorted_ids)) & (self.sorted_ids[clipped] == query)
        if not np.all(ok):
            missing = query[~ok]
            raise KeyError(f"{self.name} missing {len(missing)} ids; examples={missing[:10].tolist()}")
        return self.order[pos]
