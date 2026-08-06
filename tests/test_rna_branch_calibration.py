from __future__ import annotations

import numpy as np

from methylation_predictor.rna_branch.calibration import (
    apply_calibration,
    fit_alpha,
    logit,
    sigmoid,
)


def test_logit_sigmoid_roundtrip() -> None:
    p = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
    assert np.allclose(sigmoid(logit(p)), p, atol=1e-6)


def test_apply_calibration_identity_at_alpha_one_and_prior_at_alpha_zero() -> None:
    rng = np.random.default_rng(0)
    prior = rng.uniform(0.2, 0.8, size=5)
    raw = np.clip(prior[None, :] + rng.normal(scale=0.1, size=(6, 5)), 0.05, 0.95)
    calibrated_one = apply_calibration(prior, raw, alpha=1.0)
    calibrated_zero = apply_calibration(prior, raw, alpha=0.0)
    assert np.allclose(calibrated_one, raw, atol=1e-6)
    assert np.allclose(calibrated_zero, np.broadcast_to(prior, raw.shape), atol=1e-6)


def test_fit_alpha_mse_recovers_known_shrinkage_factor() -> None:
    rng = np.random.default_rng(17)
    n_samples, n_cpgs = 200, 30
    prior = rng.uniform(0.3, 0.7, size=n_cpgs)
    true_delta_logit = rng.normal(scale=1.0, size=(n_samples, n_cpgs))
    prior_logit = logit(prior)
    target = sigmoid(prior_logit[None, :] + true_delta_logit)
    # Raw model undershoots the true dynamic by a known factor (like the
    # amplitude_ratio<1 collapse observed in training) -- alpha should recover
    # close to 1/shrink to restore the true amplitude.
    shrink = 0.5
    raw_prediction = sigmoid(prior_logit[None, :] + shrink * true_delta_logit)
    fit = fit_alpha(target, prior, raw_prediction, objective="mse", alpha_max=6.0, alpha_grid_points=241)
    assert abs(fit.alpha - 1.0 / shrink) < 0.1
    assert fit.fit_split == "validation"
    assert fit.observed_cells == n_samples * n_cpgs


def test_fit_alpha_median_skill_requires_eligibility_mask() -> None:
    target = np.full((4, 3), 0.5)
    prior = np.full(3, 0.5)
    raw = np.full((4, 3), 0.5)
    try:
        fit_alpha(target, prior, raw, objective="median_skill")
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_fit_alpha_median_skill_prefers_matching_amplitude() -> None:
    rng = np.random.default_rng(41)
    n_samples, n_cpgs = 150, 20
    prior = rng.uniform(0.3, 0.7, size=n_cpgs)
    true_delta_logit = rng.normal(scale=1.2, size=(n_samples, n_cpgs))
    prior_logit = logit(prior)
    target = sigmoid(prior_logit[None, :] + true_delta_logit)
    shrink = 0.4
    raw_prediction = sigmoid(prior_logit[None, :] + shrink * true_delta_logit)
    eligible = np.ones(n_cpgs, dtype=bool)
    fit = fit_alpha(
        target, prior, raw_prediction, objective="median_skill",
        eligible_cpg_mask=eligible, alpha_max=6.0, alpha_grid_points=241,
    )
    assert abs(fit.alpha - 1.0 / shrink) < 0.15
    assert fit.eligible_cpgs == n_cpgs
