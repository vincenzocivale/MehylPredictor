"""Explicit pair-complete and scalable axis-full-coverage schedules."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SourceSchedule:
    n_rows: int
    n_cpgs: int
    row_block: int
    cpg_block: int
    epoch: int
    seed: int
    policy: str
    layout: str = "legacy_scattered"

    def __post_init__(self):
        if min(self.n_rows,self.n_cpgs,self.row_block,self.cpg_block) < 1:
            raise ValueError("schedule dimensions must be positive")
        rng = np.random.default_rng([self.seed,self.epoch,443])
        if self.layout == "legacy_scattered":
            row_order = rng.permutation(self.n_rows)
            cpg_order = rng.permutation(self.n_cpgs)
            self.rows = [row_order[i:i+self.row_block] for i in range(0,self.n_rows,self.row_block)]
            self.cpgs = [cpg_order[i:i+self.cpg_block] for i in range(0,self.n_cpgs,self.cpg_block)]
        elif self.layout == "contiguous_blocks":
            # Keep each physical HDF5/mmap selection contiguous. Randomness is
            # retained by shuffling the block grid below, rather than by turning
            # every minibatch into a whole-file point selection.
            self.rows = [np.arange(i,min(i+self.row_block,self.n_rows),dtype=np.int64)
                         for i in range(0,self.n_rows,self.row_block)]
            self.cpgs = [np.arange(i,min(i+self.cpg_block,self.n_cpgs),dtype=np.int64)
                         for i in range(0,self.n_cpgs,self.cpg_block)]
        else:
            raise ValueError("layout must be legacy_scattered or contiguous_blocks")
        if self.policy == "pair_complete":
            grid = [(r,c) for r in range(len(self.rows)) for c in range(len(self.cpgs))]
            order = rng.permutation(len(grid)); self.plan=[grid[int(i)] for i in order]
        elif self.policy == "axis_full_coverage":
            n=max(len(self.rows),len(self.cpgs)); offset=self.epoch % len(self.rows)
            base=[((i+offset)%len(self.rows), i%len(self.cpgs)) for i in range(n)]
            if self.layout == "contiguous_blocks":
                order=rng.permutation(n); self.plan=[base[int(i)] for i in order]
            else:
                self.plan=base
        else:
            raise ValueError("policy must be pair_complete or axis_full_coverage")

    def __len__(self): return len(self.plan)
    def __getitem__(self, i):
        r,c=self.plan[i]; return self.rows[r], self.cpgs[c]

    def report(self):
        return {"policy": self.policy, "layout": self.layout, "row_blocks": len(self.rows), "cpg_blocks": len(self.cpgs), "steps": len(self.plan), "sample_axis_coverage": 1.0, "cpg_axis_coverage": 1.0, "pair_complete": self.policy=="pair_complete"}


def interleave(schedules: list[SourceSchedule], *, seed: int, epoch: int) -> list[tuple[int,int]]:
    plan=[(source,step) for source,schedule in enumerate(schedules) for step in range(len(schedule))]
    rng=np.random.default_rng([seed,epoch,991]); order=rng.permutation(len(plan))
    return [plan[int(i)] for i in order]
