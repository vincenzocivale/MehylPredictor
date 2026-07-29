import numpy as np
import pandas as pd
import pytest

from methylation_predictor.diagnostics.methylprophet.stage_d_matched import (
    calibrate_global,
    load_matched,
    paired_bootstrap,
)


def _frame(prediction):
    return pd.DataFrame({
        "sample_idx": ["s1", "s1", "s2", "s2"],
        "cpg_idx": ["a", "b", "a", "b"],
        "cancer_type": ["x", "x", "y", "y"],
        "target": [.1, np.nan, .3, .4],
        "prediction": prediction,
    })


def test_matched_loader_rejects_prediction_mask_mismatch(tmp_path):
    left = tmp_path / "left.parquet"
    right = tmp_path / "right.parquet"
    _frame([.1, .2, .3, .4]).to_parquet(left, index=False)
    _frame([.1, .2, np.nan, .4]).to_parquet(right, index=False)
    with pytest.raises(ValueError, match="missing predictions"):
        load_matched({"left": left, "right": right})


def test_global_calibration_is_fit_from_validation_only(tmp_path):
    source = tmp_path / "validation.parquet"
    output = tmp_path / "calibrated.parquet"
    frame = _frame([.2, .2, .6, .8])
    frame["target"] = [.15, np.nan, .35, .45]
    frame["static_prediction"] = [.1, .1, .1, .1]
    frame.to_parquet(source, index=False)
    calibrate_global(source, output)
    got = pd.read_parquet(output)
    # alpha = 1/2 from finite validation rows: dynamic prediction is twice truth.
    assert np.allclose(got.prediction[[0, 2, 3]], [.15, .35, .45])


def test_paired_bootstrap_reports_all_requested_resampling_units():
    target = np.array([[.1, .2, .3, .4], [.2, .3, .4, .5]])
    prior = np.zeros(4)
    reference = target + .1
    candidate = target + .05
    result = paired_bootstrap(target, prior, candidate, reference, np.array(["x", "y"]),
                              [np.array([0, 1]), np.array([2, 3])], 20, 1)
    assert result["point"]["delta_mse"] < 0
    assert set(result) == {"point", "patients", "genomic_blocks", "hierarchical"}
    assert "delta_within_cancer_skill_ci_low" in result["hierarchical"]
