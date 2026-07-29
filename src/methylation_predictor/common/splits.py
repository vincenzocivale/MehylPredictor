"""Split invariants shared by lightweight validation code."""
from __future__ import annotations

import pandas as pd


def assert_disjoint(frame: pd.DataFrame, key: str, split: str) -> None:
    counts = frame.groupby(key, dropna=False)[split].nunique()
    if (counts > 1).any():
        raise ValueError(f"{key} occurs in multiple split partitions")
