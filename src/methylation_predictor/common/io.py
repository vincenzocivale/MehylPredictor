from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)
