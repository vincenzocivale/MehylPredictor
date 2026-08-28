from methylation_predictor.benchmark.methylprophet.protocol import (
    SOURCE_EXPECTED_OBSERVED,
    TABLE5_PUBLISHED_METHYLPROPHET,
    TABLE7_PUBLISHED_METHYLPROPHET,
    published_delta,
)


def test_published_delta_defaults_to_table5_reference():
    ours = {"mas_pcc": 0.60, "mac_pcc": 0.95, "mse": 0.015, "mae": 0.06}
    delta = published_delta(ours, "train_cpg_x_val_sample")
    reference = TABLE5_PUBLISHED_METHYLPROPHET["train_cpg_x_val_sample"]
    for metric in ("mas_pcc", "mac_pcc", "mse", "mae"):
        assert delta[metric] == ours[metric] - reference[metric]


def test_published_delta_accepts_table7_reference():
    ours = {"mas_pcc": 0.45, "mac_pcc": 0.90, "mse": 0.03, "mae": 0.11}
    reference = TABLE7_PUBLISHED_METHYLPROPHET["train_array"]
    delta = published_delta(ours, "val_cpg_x_val_sample", reference)
    expected = reference["val_cpg_x_val_sample"]
    for metric in ("mas_pcc", "mac_pcc", "mse", "mae"):
        assert delta[metric] == ours[metric] - expected[metric]


def test_table7_reference_rows_have_all_three_views():
    for row in ("train_array", "train_array_wgbs", "train_array_epic", "train_array_epic_wgbs"):
        views = TABLE7_PUBLISHED_METHYLPROPHET[row]
        assert set(views) == {
            "train_cpg_x_val_sample",
            "val_cpg_x_train_sample",
            "val_cpg_x_val_sample",
        }
        for metrics in views.values():
            assert set(metrics) == {"mas_pcc", "mac_pcc", "mse", "mae"}


def test_source_expected_observed_subsets_sum_correctly():
    array_only = {k: v for k, v in SOURCE_EXPECTED_OBSERVED.items() if k in {"array"}}
    array_wgbs = {k: v for k, v in SOURCE_EXPECTED_OBSERVED.items() if k in {"array", "wgbs"}}
    array_epic = {k: v for k, v in SOURCE_EXPECTED_OBSERVED.items() if k in {"array", "epic"}}
    assert sum(array_only.values()) == 275_093_377
    assert sum(array_wgbs.values()) == 275_093_377 + 63_982_272
    assert sum(array_epic.values()) == 275_093_377 + 115_856_100


def test_table7_train_array_epic_wgbs_matches_table5_published():
    assert TABLE7_PUBLISHED_METHYLPROPHET["train_array_epic_wgbs"] == TABLE5_PUBLISHED_METHYLPROPHET
