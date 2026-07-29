"""Evaluation metrics separating total, locus-static, and patient-specific skill."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def _valid_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return np.asarray(x)[mask].astype(np.float64), np.asarray(y)[mask].astype(np.float64)


def _mse(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _valid_pair(x, y)
    return float(np.mean((x - y) ** 2)) if len(x) else float("nan")


def _mae(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _valid_pair(x, y)
    return float(np.mean(np.abs(x - y))) if len(x) else float("nan")


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _valid_pair(x, y)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _valid_pair(x, y)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    statistic = spearmanr(x, y).statistic
    return float(statistic) if np.isfinite(statistic) else float("nan")


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


def evaluate_predictions(
    target: np.ndarray,
    prediction: np.ndarray,
    prior: np.ndarray,
    cancer_types: np.ndarray,
) -> dict[str, object]:
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

    return {
        "rows": int(valid.sum()),
        "samples": int(target.shape[0]),
        "cpgs": int(target.shape[1]),
        "mse": model_mse,
        "mae": _mae(target, prediction),
        "prior_mse": prior_mse,
        "skill_vs_prior": _safe_skill(model_mse, prior_mse),
        "dynamic_skill": _safe_skill(dynamic_mse, dynamic_baseline_mse),
        "dynamic_pearson": _pearson(pred_centred, true_centred),
        "dynamic_spearman": _spearman(pred_centred, true_centred),
        "within_cancer_skill": _safe_skill(within_mse, within_baseline_mse),
        "within_cancer_pearson": _pearson(pred_within, true_within),
        "within_cancer_spearman": _spearman(pred_within, true_within),
        "dynamic_calibration_alpha": alpha,
        "dynamic_amplitude_ratio": amplitude_ratio,
        "sample_win_fraction": float(np.nanmean(sample_model_mse < sample_prior_mse)),
        "cpg_win_fraction": float(np.nanmean(cpg_model_mse < cpg_prior_mse)),
        "macro_cancer_mse": float(np.mean(cancer_mses)) if cancer_mses else float("nan"),
        "macro_cancer_skill_vs_prior": float(np.mean(cancer_skills)) if cancer_skills else float("nan"),
        "per_cancer": per_cancer,
    }
