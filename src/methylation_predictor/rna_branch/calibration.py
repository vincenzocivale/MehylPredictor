"""Post-hoc global calibration of a model's dynamic (prior-relative) component.

Frozen alpha protocol (must be fit on validation only, never on test):

    logit(beta_cal[s, i]) = logit(prior[i]) + alpha * (logit(raw[s, i]) - logit(prior[i]))

At alpha=1 this is the identity (beta_cal == raw). At alpha=0 it collapses to
the prior. This is architecture-agnostic: for our residual model,
raw == sigmoid(prior_logit + delta_logit) by construction (see models.py), so
logit(raw) - logit(prior) recovers delta_logit exactly; for MethylProphet (or
any other model exposing only its output probability) the same expression is
the natural analogue -- "how far this model's raw prediction moved from the
fixed prior, in logit space" -- with no dependence on that model's internals.

alpha is chosen by a dense grid search over validation data only (never test)
under one of two objectives:

- "mse" (default, recommended): minimizes global validation MSE. Conservative;
  cannot be gamed by median-of-per-CpG-ratio quirks.
- "median_skill": maximizes median per-CpG skill-vs-prior on eligible
  (variable) CpGs. More aggressive; kept only as a secondary/diagnostic
  criterion, not the frozen default.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def logit(probability: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    clipped = np.clip(probability, epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def apply_calibration(
    prior: np.ndarray,
    raw_prediction: np.ndarray,
    alpha: float,
    *,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """Return the calibrated prediction matrix for a frozen scalar alpha.

    ``prior`` is ``[cpgs]``; ``raw_prediction`` is ``[samples, cpgs]`` (or any
    shape broadcastable against a trailing ``prior`` axis). NaNs in
    ``raw_prediction`` propagate through untouched.
    """
    prior_logit = logit(np.asarray(prior, dtype=np.float64), epsilon)
    raw_logit = logit(np.asarray(raw_prediction, dtype=np.float64), epsilon)
    delta = raw_logit - prior_logit
    return sigmoid(prior_logit + alpha * delta)


@dataclass(frozen=True, slots=True)
class AlphaFit:
    alpha: float
    objective: str
    objective_value: float
    fit_split: str
    alpha_grid_min: float
    alpha_grid_max: float
    alpha_grid_points: int
    observed_cells: int
    eligible_cpgs: int | None
    grid: list[tuple[float, float]]


def fit_alpha(
    target: np.ndarray,
    prior: np.ndarray,
    raw_prediction: np.ndarray,
    *,
    objective: str = "mse",
    eligible_cpg_mask: np.ndarray | None = None,
    alpha_min: float = 0.0,
    alpha_max: float = 4.0,
    alpha_grid_points: int = 161,
    epsilon: float = 1e-4,
    fit_split: str = "validation",
) -> AlphaFit:
    """Grid-search the scalar alpha on (target, prior, raw_prediction).

    Must be called with validation-only arrays -- the caller is responsible
    for never passing test data here; ``fit_split`` is recorded in the
    returned object purely for provenance/audit, it does not gate anything.
    """
    if objective not in {"mse", "median_skill"}:
        raise ValueError(f"unknown calibration objective: {objective}")
    target = np.asarray(target, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    raw_prediction = np.asarray(raw_prediction, dtype=np.float64)
    if target.shape != raw_prediction.shape:
        raise ValueError("target and raw_prediction shapes differ")
    if target.shape[1] != len(prior):
        raise ValueError("prior length does not match CpG dimension")

    prior_logit = logit(prior, epsilon)
    raw_logit = logit(raw_prediction, epsilon)
    delta = raw_logit - prior_logit[None, :]

    valid = np.isfinite(target) & np.isfinite(raw_prediction) & np.isfinite(prior)[None, :]
    if objective == "median_skill":
        if eligible_cpg_mask is None:
            raise ValueError("median_skill objective requires eligible_cpg_mask")
        eligible_cpg_mask = np.asarray(eligible_cpg_mask, dtype=bool)
        if eligible_cpg_mask.shape != (target.shape[1],):
            raise ValueError("eligible_cpg_mask length does not match CpG dimension")
        prior_matrix = np.broadcast_to(prior[None, :], target.shape)
        prior_sse = np.where(valid, (target - prior_matrix) ** 2, 0.0).sum(axis=0)
        eligible_columns = eligible_cpg_mask & (prior_sse > 0) & (valid.sum(axis=0) >= 2)
    else:
        eligible_columns = None

    alpha_grid = np.linspace(alpha_min, alpha_max, alpha_grid_points)
    grid_values: list[tuple[float, float]] = []
    best_alpha = float(alpha_grid[0])
    best_score = float("inf") if objective == "mse" else float("-inf")

    for alpha in alpha_grid:
        calibrated = sigmoid(prior_logit[None, :] + alpha * delta)
        if objective == "mse":
            error = np.where(valid, (target - calibrated) ** 2, 0.0)
            score = float(error.sum() / max(valid.sum(), 1))
            better = score < best_score
        else:
            model_sse = np.where(valid, (target - calibrated) ** 2, 0.0).sum(axis=0)
            skill = 1.0 - model_sse[eligible_columns] / prior_sse[eligible_columns]
            score = float(np.median(skill)) if eligible_columns.any() else float("-inf")
            better = score > best_score
        grid_values.append((float(alpha), score))
        if better:
            best_score = score
            best_alpha = float(alpha)

    return AlphaFit(
        alpha=best_alpha,
        objective=objective,
        objective_value=best_score,
        fit_split=fit_split,
        alpha_grid_min=float(alpha_min),
        alpha_grid_max=float(alpha_max),
        alpha_grid_points=int(alpha_grid_points),
        observed_cells=int(valid.sum()),
        eligible_cpgs=int(eligible_columns.sum()) if eligible_columns is not None else None,
        grid=grid_values,
    )
