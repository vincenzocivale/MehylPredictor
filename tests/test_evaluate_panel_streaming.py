from __future__ import annotations

import numpy as np

from methylation_predictor.rna_branch.config import load_config
from methylation_predictor.rna_branch.trainer import ExperimentRunner

EXACT_KEYS = [
    "mse", "mae", "skill_vs_prior", "dynamic_skill", "dynamic_pearson",
    "dynamic_calibration_alpha", "dynamic_amplitude_ratio",
    "patient_dynamic_pearson_median", "locus_dynamic_pearson_median",
    "within_cancer_skill", "sample_win_fraction", "cpg_win_fraction",
]


def test_evaluate_panel_streaming_matches_predict_panel(synthetic_project):
    config = load_config(synthetic_project)
    runner = ExperimentRunner(config)
    try:
        non_streaming = runner.predict_panel("test", "test", keep_predictions=False)
        streaming = runner.evaluate_panel_streaming("test", "test", cpg_chunk_size=1)
        for key in EXACT_KEYS:
            assert np.isclose(streaming[key], non_streaming.metrics[key], atol=1e-6, rtol=1e-5, equal_nan=True), (
                f"{key}: streaming={streaming[key]!r} non_streaming={non_streaming.metrics[key]!r}"
            )
        assert streaming["samples"] == non_streaming.metrics["samples"]
        assert streaming["cpgs"] == non_streaming.metrics["cpgs"]

        # override params: union of validation+test splits, matching the
        # official-val-panel use case (validation ∪ test = official val_cpg).
        union_sample = np.union1d(runner.bundle.sample_indices("validation"), runner.bundle.sample_indices("test"))
        union_cpg = np.union1d(runner.bundle.cpg_indices("validation"), runner.bundle.cpg_indices("test"))
        override_streaming = runner.evaluate_panel_streaming(
            None, None, sample_indices_override=union_sample, cpg_indices_override=union_cpg,
        )
        override_reference = runner.predict_panel(
            None, None, sample_indices_override=union_sample, cpg_indices_override=union_cpg,
        )
        assert override_streaming["samples"] == len(union_sample)
        assert override_streaming["cpgs"] == len(union_cpg)
        for key in EXACT_KEYS:
            assert np.isclose(
                override_streaming[key], override_reference.metrics[key], atol=1e-6, rtol=1e-5, equal_nan=True
            ), f"override {key}: streaming={override_streaming[key]!r} reference={override_reference.metrics[key]!r}"
    finally:
        runner.close()
