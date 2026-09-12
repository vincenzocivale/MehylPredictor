from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from methylation_predictor.rna_training.metrics import ArrayMomentMetrics
from methylation_predictor.storage import LocusPriorCache


ROOT = Path(__file__).resolve().parents[1]


def test_prior_cache_needs_no_genomic_embedding_file(tmp_path):
    np.save(
        tmp_path / "cpg_idx.npy",
        np.array([11, 7, 19], dtype=np.int64),
    )
    np.save(
        tmp_path / "prior.npy",
        np.array([0.2, 0.4, 0.8], dtype=np.float32),
    )

    cache = LocusPriorCache(tmp_path)
    got = cache.get(np.array([19, 11], dtype=np.int64))

    np.testing.assert_allclose(
        got,
        np.array([0.8, 0.2], dtype=np.float32),
    )
    assert not (tmp_path / "embeddings.f16.npy").exists()
    assert not (tmp_path / "sigma.npy").exists()


def test_metrics_can_skip_prior_without_changing_prediction_metrics():
    target = np.array(
        [[0.1, 0.8], [0.3, 0.6]],
        dtype=np.float32,
    )
    pred = np.array(
        [[0.2, 0.7], [0.4, 0.5]],
        dtype=np.float32,
    )

    metrics = ArrayMomentMetrics(2, 2, track_prior=False)
    metrics.add(0, 0, target, pred)
    out = metrics.finalize()

    assert np.isclose(out["mse"], np.mean((pred - target) ** 2))
    assert np.isclose(out["mae"], np.mean(np.abs(pred - target)))
    assert math.isnan(out["prior_mse"])
    assert math.isnan(out["skill_vs_prior"])


def test_prior_metric_matches_legacy_formula():
    target = np.array(
        [[0.1, 0.8], [0.3, 0.6]],
        dtype=np.float32,
    )
    pred = np.array(
        [[0.2, 0.7], [0.4, 0.5]],
        dtype=np.float32,
    )
    prior = np.array([0.25, 0.65], dtype=np.float32)

    metrics = ArrayMomentMetrics(2, 2)
    metrics.add(0, 0, target, pred, prior)
    out = metrics.finalize()

    mse = float(np.mean((pred - target) ** 2))
    prior_pred = np.broadcast_to(prior[None, :], target.shape)
    prior_mse = float(np.mean((prior_pred - target) ** 2))

    assert np.isclose(out["prior_mse"], prior_mse)
    assert np.isclose(
        out["skill_vs_prior"],
        1.0 - mse / prior_mse,
    )


def test_rna_train_cli_no_longer_exposes_feature_cache():
    text = (ROOT / "scripts/train.py").read_text()
    assert "--feature-cache" not in text
    assert "--prior-cache" in text


def test_active_experiment_scripts_do_not_pass_feature_cache():
    stale = []
    for path in (ROOT / "scripts/experiments").iterdir():
        if (
            path.is_file()
            and path.suffix in {".py", ".sh"}
            and "--feature-cache" in path.read_text()
        ):
            stale.append(path.name)
    assert stale == []
