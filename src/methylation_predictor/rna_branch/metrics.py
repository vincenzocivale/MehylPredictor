"""Evaluation metrics separating total, locus-static, and patient-specific skill."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import spearmanr

from .biological_metrics import (
    BiologicalMetricConfig,
    per_locus_biological_metrics,
    per_sample_biological_metrics,
    summarize_biological_fidelity,
)


def _valid_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return np.asarray(x)[mask].astype(np.float64), np.asarray(y)[mask].astype(np.float64)


def _mse(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _valid_pair(x, y)
    return float(np.mean((x - y) ** 2)) if len(x) else float("nan")


def _mae(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _valid_pair(x, y)
    return float(np.mean(np.abs(x - y))) if len(x) else float("nan")


def _subsample_pair(x: np.ndarray, y: np.ndarray, max_n: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random subsample for large-N correlation estimates -- standard error of
    Pearson/Spearman r shrinks as 1/sqrt(n), so at max_n>=1e6 the subsampled
    estimate is statistically indistinguishable from the full-array one while
    being >>100x cheaper. max_n=None (default everywhere except explicit opt-in)
    preserves exact prior behaviour."""
    if max_n is None or len(x) <= max_n:
        return x, y
    idx = np.random.default_rng(seed).choice(len(x), size=max_n, replace=False)
    return x[idx], y[idx]


def _pearson(x: np.ndarray, y: np.ndarray, max_n: int | None = None, seed: int = 0) -> float:
    x, y = _valid_pair(x, y)
    x, y = _subsample_pair(x, y, max_n, seed)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray, max_n: int | None = None, seed: int = 0) -> float:
    x, y = _valid_pair(x, y)
    x, y = _subsample_pair(x, y, max_n, seed)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    statistic = spearmanr(x, y).statistic
    return float(statistic) if np.isfinite(statistic) else float("nan")


def _median_axis_correlation(
    prediction: np.ndarray,
    target: np.ndarray,
    axis: int,
    statistic: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """Median correlation across patients (axis=1) or CpGs (axis=0)."""
    if axis not in {0, 1}:
        raise ValueError("axis must be 0 or 1")
    values: list[float] = []
    count = prediction.shape[1 - axis]
    for index in range(count):
        if axis == 1:
            value = statistic(prediction[index, :], target[index, :])
        else:
            value = statistic(prediction[:, index], target[:, index])
        if np.isfinite(value):
            values.append(value)
    return float(np.median(values)) if values else float("nan")


def _safe_skill(error: float, baseline_error: float) -> float:
    if not np.isfinite(error) or not np.isfinite(baseline_error) or baseline_error <= 0:
        return float("nan")
    return float(1.0 - error / baseline_error)


def _centre_by_locus(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    counts = valid.sum(axis=0, keepdims=True)
    sums = np.where(valid, values, 0.0).sum(axis=0, keepdims=True)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    result = values - means
    result[~valid] = np.nan
    return result


def _within_cancer_centred(
    values: np.ndarray,
    valid: np.ndarray,
    cancer_types: np.ndarray,
) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=np.float64)
    for cancer_type in np.unique(cancer_types):
        rows = np.flatnonzero(cancer_types == cancer_type)
        group_valid = valid[rows]
        counts = group_valid.sum(axis=0, keepdims=True)
        sums = np.where(group_valid, values[rows], 0.0).sum(axis=0, keepdims=True)
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        centred = values[rows] - means
        centred[~group_valid] = np.nan
        result[rows] = centred
    return result


def _masked_axis_mean(values: np.ndarray, valid: np.ndarray, axis: int) -> np.ndarray:
    counts = valid.sum(axis=axis)
    sums = np.where(valid, values, 0.0).sum(axis=axis)
    return np.divide(
        sums,
        counts,
        out=np.full(np.shape(sums), np.nan, dtype=np.float64),
        where=counts > 0,
    )


def _tertile_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    prior_matrix: np.ndarray,
) -> dict[str, float | int]:
    true_dynamic = target - prior_matrix
    pred_dynamic = prediction - prior_matrix
    valid = np.isfinite(target) & np.isfinite(prediction) & np.isfinite(prior_matrix)
    true_centred = _centre_by_locus(true_dynamic, valid)
    pred_centred = _centre_by_locus(pred_dynamic, valid)
    model_mse = _mse(target, prediction)
    prior_mse = _mse(target, prior_matrix)
    return {
        "rows": int(valid.sum()),
        "cpgs": int(target.shape[1]),
        "mse": model_mse,
        "prior_mse": prior_mse,
        "skill_vs_prior": _safe_skill(model_mse, prior_mse),
        "dynamic_pearson": _pearson(pred_centred, true_centred),
        "dynamic_spearman": _spearman(pred_centred, true_centred),
        "patient_dynamic_pearson_median": _median_axis_correlation(
            pred_dynamic, true_dynamic, axis=1, statistic=_pearson
        ),
        "locus_dynamic_pearson_median": _median_axis_correlation(
            pred_dynamic, true_dynamic, axis=0, statistic=_pearson
        ),
    }


def evaluate_predictions(
    target: np.ndarray,
    prediction: np.ndarray,
    prior: np.ndarray,
    cancer_types: np.ndarray,
    cpg_tertiles: np.ndarray | None = None,
    include_median_correlations: bool = True,
    correlation_max_n: int | None = None,
    include_biological_fidelity: bool = False,
    biological_min_observed_samples: int = 20,
    biological_min_target_std: float = 0.05,
    biological_min_cancer_group_samples: int = 4,
    biological_eligibility_target: np.ndarray | None = None,
) -> dict[str, object]:
    """include_median_correlations=False skips the four *_dynamic_{pearson,spearman}_median
    diagnostics (a per-row/per-column Python loop over scipy calls -- O(n_samples + n_cpgs)
    calls, expensive at genome-wide scale with many samples). mse/skill_vs_prior and every
    other field are unaffected; use this for large batch comparisons where those four
    per-model diagnostics aren't needed (they're still available from any run that already
    computed them with the default True).

    correlation_max_n subsamples (fixed seed, reproducible) the full-array
    dynamic_pearson/dynamic_spearman/within_cancer_pearson/within_cancer_spearman
    calls above that size -- measured cost at genome-wide scale (~219M valid
    entries): dynamic_spearman alone ~108s (full-array O(n log n) rank).
    Standard error of r shrinks as 1/sqrt(n), so e.g. max_n=2_000_000 is
    statistically indistinguishable from the exact value while being >100x
    cheaper. Default None preserves the exact value for all existing callers."""
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    cancer_types = np.asarray(cancer_types)
    if target.shape != prediction.shape:
        raise ValueError("target and prediction shapes differ")
    if target.shape[1] != len(prior):
        raise ValueError("prior length does not match CpG dimension")
    if target.shape[0] != len(cancer_types):
        raise ValueError("cancer-type length does not match sample dimension")
    if cpg_tertiles is not None and target.shape[1] != len(cpg_tertiles):
        raise ValueError("CpG-tertile length does not match CpG dimension")

    prior_matrix = np.broadcast_to(prior[None, :], target.shape)
    valid = np.isfinite(target) & np.isfinite(prediction) & np.isfinite(prior_matrix)
    model_mse = _mse(target, prediction)
    prior_mse = _mse(target, prior_matrix)

    true_dynamic = target - prior_matrix
    pred_dynamic = prediction - prior_matrix
    true_centred = _centre_by_locus(true_dynamic, valid)
    pred_centred = _centre_by_locus(pred_dynamic, valid)
    dynamic_mse = _mse(true_centred, pred_centred)
    dynamic_baseline_mse = _mse(true_centred, np.zeros_like(true_centred))

    true_within = _within_cancer_centred(true_dynamic, valid, cancer_types)
    pred_within = _within_cancer_centred(pred_dynamic, valid, cancer_types)
    within_mse = _mse(true_within, pred_within)
    within_baseline_mse = _mse(true_within, np.zeros_like(true_within))

    px, ty = _valid_pair(pred_centred, true_centred)
    denominator = float(np.dot(px, px)) if len(px) else 0.0
    alpha = float(np.dot(px, ty) / denominator) if denominator > 0 else float("nan")
    amplitude_ratio = (
        float(np.std(px) / np.std(ty)) if len(px) > 1 and np.std(ty) > 0 else float("nan")
    )

    per_cancer: dict[str, dict[str, float | int]] = {}
    cancer_mses = []
    cancer_skills = []
    for cancer_type in np.unique(cancer_types):
        rows = cancer_types == cancer_type
        group_model = _mse(target[rows], prediction[rows])
        group_prior = _mse(target[rows], prior_matrix[rows])
        group_skill = _safe_skill(group_model, group_prior)
        per_cancer[str(cancer_type)] = {
            "samples": int(rows.sum()),
            "mse": group_model,
            "prior_mse": group_prior,
            "skill_vs_prior": group_skill,
        }
        if np.isfinite(group_model):
            cancer_mses.append(group_model)
        if np.isfinite(group_skill):
            cancer_skills.append(group_skill)

    sample_model_mse = _masked_axis_mean((target - prediction) ** 2, valid, axis=1)
    sample_prior_mse = _masked_axis_mean((target - prior_matrix) ** 2, valid, axis=1)
    cpg_model_mse = _masked_axis_mean((target - prediction) ** 2, valid, axis=0)
    cpg_prior_mse = _masked_axis_mean((target - prior_matrix) ** 2, valid, axis=0)

    mas_pcc = (
        _median_axis_correlation(pred_dynamic, true_dynamic, axis=0, statistic=_pearson)
        if include_median_correlations else None
    )

    biological_summary: dict[str, object] = {}
    if include_biological_fidelity:
        biological_config = BiologicalMetricConfig(
            min_observed_samples=biological_min_observed_samples,
            min_target_std=biological_min_target_std,
            min_cancer_group_samples=biological_min_cancer_group_samples,
        )
        per_locus_bio = per_locus_biological_metrics(
            target,
            prediction,
            prior,
            cancer_types,
            eligibility_target=biological_eligibility_target,
            config=biological_config,
        )
        per_sample_bio = per_sample_biological_metrics(
            target, prediction, prior, cancer_types=cancer_types
        )
        biological_summary = summarize_biological_fidelity(per_locus_bio, per_sample_bio)

    per_tertile: dict[str, dict[str, float | int]] = {}
    if cpg_tertiles is not None:
        cpg_tertiles = np.asarray(cpg_tertiles, dtype=np.int64)
        for label, name in enumerate(("low", "mid", "high")):
            columns = cpg_tertiles == label
            if columns.any():
                per_tertile[name] = _tertile_metrics(
                    target[:, columns], prediction[:, columns], prior_matrix[:, columns]
                )

    return {
        "rows": int(valid.sum()),
        "samples": int(target.shape[0]),
        "cpgs": int(target.shape[1]),
        "mse": model_mse,
        "mae": _mae(target, prediction),
        "prior_mse": prior_mse,
        "skill_vs_prior": _safe_skill(model_mse, prior_mse),
        "dynamic_skill": _safe_skill(dynamic_mse, dynamic_baseline_mse),
        "dynamic_pearson": _pearson(pred_centred, true_centred, max_n=correlation_max_n),
        "dynamic_spearman": _spearman(pred_centred, true_centred, max_n=correlation_max_n),
        "patient_dynamic_pearson_median": (
            _median_axis_correlation(pred_dynamic, true_dynamic, axis=1, statistic=_pearson)
            if include_median_correlations else None
        ),
        "patient_dynamic_spearman_median": (
            _median_axis_correlation(pred_dynamic, true_dynamic, axis=1, statistic=_spearman)
            if include_median_correlations else None
        ),
        "locus_dynamic_pearson_median": mas_pcc,
        "mas_pcc": mas_pcc,
        "locus_dynamic_spearman_median": (
            _median_axis_correlation(pred_dynamic, true_dynamic, axis=0, statistic=_spearman)
            if include_median_correlations else None
        ),
        "within_cancer_skill": _safe_skill(within_mse, within_baseline_mse),
        "within_cancer_pearson": _pearson(pred_within, true_within, max_n=correlation_max_n),
        "within_cancer_spearman": _spearman(pred_within, true_within, max_n=correlation_max_n),
        "dynamic_calibration_alpha": alpha,
        "dynamic_amplitude_ratio": amplitude_ratio,
        "sample_win_fraction": float(np.nanmean(sample_model_mse < sample_prior_mse)),
        "cpg_win_fraction": float(np.nanmean(cpg_model_mse < cpg_prior_mse)),
        "macro_cancer_mse": float(np.mean(cancer_mses)) if cancer_mses else float("nan"),
        "macro_cancer_skill_vs_prior": float(np.mean(cancer_skills)) if cancer_skills else float("nan"),
        **biological_summary,
        "per_cancer": per_cancer,
        "per_variability_tertile": per_tertile,
    }
