from __future__ import annotations

import numpy as np
import pytest

from methylation_predictor.rna_branch.metrics import evaluate_predictions
from methylation_predictor.rna_branch.streaming_metrics import StreamingPanelMetrics

EXACT_KEYS = [
    "mse", "mae", "prior_mse", "skill_vs_prior", "dynamic_skill",
    "dynamic_pearson", "dynamic_calibration_alpha", "dynamic_amplitude_ratio",
    "patient_dynamic_pearson_median", "locus_dynamic_pearson_median",
    "within_cancer_skill", "within_cancer_pearson",
    "sample_win_fraction", "cpg_win_fraction",
    "macro_cancer_mse", "macro_cancer_skill_vs_prior",
]


def _synthetic_panel(seed: int, n_samples: int = 40, n_cpgs: int = 300):
    rng = np.random.default_rng(seed)
    prior = rng.uniform(0.05, 0.95, size=n_cpgs)
    locus_effect = rng.normal(0, 0.05, size=n_cpgs)
    sample_effect = rng.normal(0, 0.05, size=n_samples)
    target = np.clip(
        prior[None, :] + locus_effect[None, :] + sample_effect[:, None]
        + rng.normal(0, 0.02, size=(n_samples, n_cpgs)),
        0.001, 0.999,
    )
    noise = rng.normal(0, 0.03, size=(n_samples, n_cpgs))
    prediction = np.clip(target + noise, 0.001, 0.999)
    # sprinkle missingness, matching the real beta matrix's NaN-as-missing convention
    missing = rng.random(size=(n_samples, n_cpgs)) < 0.05
    target = target.copy()
    target[missing] = np.nan

    cancer_types = np.asarray([f"CANCER_{i % 4}" for i in range(n_samples)], dtype=object)
    proxy = rng.normal(size=n_cpgs)
    thresholds = np.quantile(proxy, [1 / 3, 2 / 3])
    tertiles = np.digitize(proxy, thresholds, right=True).astype(np.int64)
    chromosomes = np.asarray([f"chr{1 + (i % 5)}" for i in range(n_cpgs)], dtype=object)
    return target, prediction, prior, cancer_types, tertiles, chromosomes


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("chunk_size", [37, 128, 300])
def test_streaming_matches_non_streaming_exact_metrics(seed, chunk_size):
    target, prediction, prior, cancer_types, tertiles, chromosomes = _synthetic_panel(seed)
    reference = evaluate_predictions(target, prediction, prior, cancer_types, cpg_tertiles=tertiles)

    acc = StreamingPanelMetrics(n_samples=target.shape[0], cancer_type_names=sorted(set(cancer_types.tolist())))
    n_cpgs = target.shape[1]
    for start in range(0, n_cpgs, chunk_size):
        sl = slice(start, start + chunk_size)
        acc.add_cpg_chunk(
            target[:, sl], prediction[:, sl], prior[sl], cancer_types,
            chromosome_codes=chromosomes[sl], tertile_codes=tertiles[sl],
        )
    streamed = acc.finalize()

    for key in EXACT_KEYS:
        assert np.isclose(streamed[key], reference[key], atol=1e-8, rtol=1e-6), (
            f"{key}: streamed={streamed[key]!r} reference={reference[key]!r}"
        )

    assert streamed["rows"] == reference["rows"]
    assert streamed["samples"] == reference["samples"]
    assert streamed["cpgs"] == reference["cpgs"]

    for name in ("low", "mid", "high"):
        ref_block = reference["per_variability_tertile"].get(name)
        streamed_block = streamed["per_variability_tertile"].get(name)
        if ref_block is None:
            continue
        assert streamed_block is not None
        # _tertile_metrics (metrics.py) only ever computed this subset per tertile;
        # the streaming implementation returns a superset (also dynamic_skill/
        # calibration_alpha/amplitude_ratio per tertile), which has nothing to
        # cross-check against since the original never computed those per-tertile.
        for key in ["mse", "prior_mse", "skill_vs_prior", "dynamic_pearson",
                    "patient_dynamic_pearson_median", "locus_dynamic_pearson_median"]:
            assert np.isclose(streamed_block[key], ref_block[key], atol=1e-8, rtol=1e-6), (
                f"tertile {name} {key}: streamed={streamed_block[key]!r} reference={ref_block[key]!r}"
            )

    for cancer_type, ref_block in reference["per_cancer"].items():
        streamed_block = streamed["per_cancer"][cancer_type]
        assert np.isclose(streamed_block["mse"], ref_block["mse"], atol=1e-8, rtol=1e-6)
        assert np.isclose(streamed_block["skill_vs_prior"], ref_block["skill_vs_prior"], atol=1e-8, rtol=1e-6)
