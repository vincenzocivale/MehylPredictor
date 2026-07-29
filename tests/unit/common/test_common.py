from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from methylation_predictor.common.hashing import sha256
from methylation_predictor.common.metrics import mae, mse
from methylation_predictor.common.schemas import require_columns
from methylation_predictor.common.splits import assert_disjoint


def test_metrics_ignore_nan_pairs() -> None:
    truth = np.array([1.0, np.nan, 3.0])
    pred = np.array([2.0, 9.0, 1.0])
    assert mse(truth, pred) == 2.5
    assert mae(truth, pred) == 1.5


def test_hash_and_schema_are_reproducible(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    path.write_text("stable\n")
    assert sha256(path) == sha256(path)
    require_columns(pd.DataFrame({"cpg_idx": [1]}), {"cpg_idx"}, "fixture")
    with pytest.raises(ValueError, match="missing columns"):
        require_columns(pd.DataFrame({"other": [1]}), {"cpg_idx"}, "fixture")


def test_disjoint_split_guard() -> None:
    assert_disjoint(pd.DataFrame({"cpg_idx": [1, 2], "split": ["train", "test"]}), "cpg_idx", "split")
    with pytest.raises(ValueError, match="multiple"):
        assert_disjoint(pd.DataFrame({"cpg_idx": [1, 1], "split": ["train", "test"]}), "cpg_idx", "split")
