from __future__ import annotations

import time

import numpy as np

from methylation_predictor.diagnostics.methylprophet.stage_d1 import bootstrap, quick_metrics


def _synthetic(seed, n_samples=200, n_cpgs=500, n_cancer_types=32, missing_frac=0.05):
    rng = np.random.default_rng(seed)
    target = rng.uniform(0.05, 0.95, size=(n_samples, n_cpgs))
    prediction = target + rng.normal(0, 0.05, size=target.shape)
    missing = rng.random(size=target.shape) < missing_frac
    target = target.copy()
    target[missing] = np.nan
    prior = rng.uniform(0.05, 0.95, size=n_cpgs)
    cancer = np.array([f"CANCER_{i % n_cancer_types:02d}" for i in range(n_samples)], dtype=object)
    return target, prediction, prior, cancer


def test_bootstrap_delta_shares_target_slice_and_matches_reference():
    """`bootstrap()`'s `delta()` closure slices target/prior/cancer once and
    reuses them for the reference and every candidate's quick_metrics call
    (previously re-sliced per call -- a redundant full fancy-index copy of
    the panel per candidate at genome-wide scale). This must not change the
    result: the point estimate must match quick_metrics() called directly
    with the naive (per-call) slicing pattern."""
    target, prediction, prior, cancer = _synthetic(11, n_samples=80, n_cpgs=120)
    candidate = prediction
    reference_pred = target + np.random.default_rng(5).normal(0, 0.08, size=target.shape)
    blocks = [np.arange(120)]  # single block -> genomic_blocks/hierarchical column-resampling is a no-op here

    result = bootstrap(target, prior, cancer, blocks, [candidate], reference_pred, replicates=5, seed=42)

    rows = np.arange(target.shape[0])
    cols = np.arange(target.shape[1])
    r = quick_metrics(target[np.ix_(rows, cols)], reference_pred[np.ix_(rows, cols)], prior[cols], cancer[rows])
    c = quick_metrics(target[np.ix_(rows, cols)], candidate[np.ix_(rows, cols)], prior[cols], cancer[rows])
    expected_point = {
        "delta_mse": c["mse"] - r["mse"],
        "delta_skill": c["skill_vs_prior"] - r["skill_vs_prior"],
        "delta_within_cancer_skill": c["within_cancer_skill"] - r["within_cancer_skill"],
    }
    for key, value in expected_point.items():
        assert np.isclose(result["point"][key], value, atol=1e-10, rtol=1e-10)


def test_delta_avoids_duplicate_target_slicing():
    """Direct measurement: delta()'s single shared target/prior/cancer slice
    must be faster than the naive pattern of re-slicing target per candidate,
    at a scale where the fancy-index copy itself is the dominant cost (many
    columns, few candidates -- exactly the genome-wide official-val panel
    shape that motivated this fix)."""
    target, prediction, prior, cancer = _synthetic(0, n_samples=800, n_cpgs=40_000, n_cancer_types=32)
    reference_pred = target + np.random.default_rng(1).normal(0, 0.05, size=target.shape)
    rows = np.arange(target.shape[0])
    cols = np.arange(target.shape[1])
    idx = np.ix_(rows, cols)

    def naive():
        r = quick_metrics(target[idx], reference_pred[idx], prior[cols], cancer[rows])
        c = quick_metrics(target[idx], prediction[idx], prior[cols], cancer[rows])
        return r, c

    def shared():
        target_sub = target[idx]
        prior_sub = prior[cols]
        cancer_sub = cancer[rows]
        r = quick_metrics(target_sub, reference_pred[idx], prior_sub, cancer_sub)
        c = quick_metrics(target_sub, prediction[idx], prior_sub, cancer_sub)
        return r, c

    started = time.perf_counter()
    naive()
    naive_seconds = time.perf_counter() - started

    started = time.perf_counter()
    shared()
    shared_seconds = time.perf_counter() - started

    assert shared_seconds < naive_seconds
