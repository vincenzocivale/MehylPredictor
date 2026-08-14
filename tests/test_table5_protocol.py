import numpy as np

from methylation_predictor.benchmark.table5.protocol import (
    TABLE5_EXPECTED,
    TABLE5_PUBLISHED_METHYLPROPHET,
    published_delta,
)


def test_table5_training_pair_total_is_published_total():
    assert (
        TABLE5_EXPECTED["array_train_observed"]
        + TABLE5_EXPECTED["epic_train_observed"]
        + TABLE5_EXPECTED["wgbs_train_observed"]
        == TABLE5_EXPECTED["total_train_observed"]
        == 454_931_749
    )


def test_table5_published_delta_orientation():
    ours = {"mas_pcc": 0.60, "mac_pcc": 0.95, "mse": 0.018, "mae": 0.08}
    delta = published_delta(ours, "train_cpg_x_val_sample")
    assert np.isclose(delta["mas_pcc"], 0.60 - TABLE5_PUBLISHED_METHYLPROPHET["train_cpg_x_val_sample"]["mas_pcc"])
    assert delta["mse"] < 0  # negative is better for an error metric
