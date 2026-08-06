"""Biological-fidelity metrics for RNA-conditioned methylome prediction.

The module deliberately separates per-locus quantitative agreement from global
pointwise error.  All thresholds are explicit and must be fixed on development
splits before the held-out test is opened.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


@dataclass(frozen=True, slots=True)
class BiologicalMetricConfig:
    min_observed_samples: int = 20
    min_target_std: float = 0.05
    min_cancer_group_samples: int = 4
    dmr_delta_threshold: float = 0.10
    top_fraction: float = 0.05
    knn_k: int = 10
    max_structure_cpgs: int = 2048
    structure_seed: int = 17


def _finite_quantiles(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {key: float("nan") for key in ("q10", "q25", "median", "q75", "q90")}
    q = np.quantile(x, [0.10, 0.25, 0.50, 0.75, 0.90])
    return dict(zip(("q10", "q25", "median", "q75", "q90"), map(float, q), strict=True))


def lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=np.float64)[mask]
    y = np.asarray(y, dtype=np.float64)[mask]
    if len(x) < 2:
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = np.dot(xc, xc) + np.dot(yc, yc) + len(x) * (x.mean() - y.mean()) ** 2
    if denominator <= 0:
        return float("nan")
    return float(2.0 * np.dot(xc, yc) / denominator)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=np.float64)[mask]
    y = np.asarray(y, dtype=np.float64)[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _dynamic_r2(x: np.ndarray, y: np.ndarray) -> float:
    """R2 after separately centring target and prediction."""
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=np.float64)[mask]
    y = np.asarray(y, dtype=np.float64)[mask]
    if len(x) < 2:
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = np.dot(xc, xc)
    if denominator <= 0:
        return float("nan")
    return float(1.0 - np.dot(xc - yc, xc - yc) / denominator)


def _within_cancer_vectors(
    target: np.ndarray,
    prediction: np.ndarray,
    cancer_types: np.ndarray,
    *,
    min_group_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    target_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    for cancer in np.unique(cancer_types):
        rows = np.flatnonzero(cancer_types == cancer)
        valid = np.isfinite(target[rows]) & np.isfinite(prediction[rows])
        keep = valid.sum() >= min_group_samples
        if not keep:
            continue
        t = target[rows][valid]
        p = prediction[rows][valid]
        target_parts.append(t - t.mean())
        prediction_parts.append(p - p.mean())
    if not target_parts:
        return np.empty(0), np.empty(0)
    return np.concatenate(target_parts), np.concatenate(prediction_parts)


def per_locus_biological_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    prior: np.ndarray,
    cancer_types: np.ndarray,
    *,
    cpg_ids: np.ndarray | None = None,
    eligibility_target: np.ndarray | None = None,
    config: BiologicalMetricConfig = BiologicalMetricConfig(),
) -> pd.DataFrame:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    cancer_types = np.asarray(cancer_types)
    if target.shape != prediction.shape:
        raise ValueError("target and prediction shapes differ")
    if target.ndim != 2:
        raise ValueError("target and prediction must be [samples, cpgs]")
    if len(prior) != target.shape[1] or len(cancer_types) != target.shape[0]:
        raise ValueError("prior/cancer metadata dimensions do not match prediction matrix")
    ids = np.arange(target.shape[1]).astype(str) if cpg_ids is None else np.asarray(cpg_ids).astype(str)
    if len(ids) != target.shape[1]:
        raise ValueError("cpg_ids length does not match CpG dimension")
    eligibility = target if eligibility_target is None else np.asarray(eligibility_target, dtype=np.float64)
    if eligibility.ndim != 2 or eligibility.shape[1] != target.shape[1]:
        raise ValueError("eligibility_target must be [reference_samples, cpgs]")

    records: list[dict[str, Any]] = []
    for index, cpg_id in enumerate(ids):
        t = target[:, index]
        p = prediction[:, index]
        valid = np.isfinite(t) & np.isfinite(p) & np.isfinite(prior[index])
        tv = t[valid]
        pv = p[valid]
        cancers = cancer_types[valid]
        count = int(valid.sum())
        target_std = float(np.std(tv)) if count else float("nan")
        eligibility_values = eligibility[:, index]
        eligibility_values = eligibility_values[np.isfinite(eligibility_values)]
        eligibility_count = int(len(eligibility_values))
        eligibility_std = (
            float(np.std(eligibility_values)) if eligibility_count else float("nan")
        )
        eligible = (
            eligibility_count >= config.min_observed_samples
            and eligibility_std >= config.min_target_std
        )
        prior_sse = float(np.sum((tv - prior[index]) ** 2)) if count else float("nan")
        model_sse = float(np.sum((tv - pv) ** 2)) if count else float("nan")
        skill = 1.0 - model_sse / prior_sse if prior_sse > 0 else float("nan")
        wc_t, wc_p = _within_cancer_vectors(
            tv, pv, cancers, min_group_samples=config.min_cancer_group_samples
        )
        records.append(
            {
                "cpg_id": cpg_id,
                "observed_samples": count,
                "eligibility_observed_samples": eligibility_count,
                "eligibility_target_std": eligibility_std,
                "target_mean": float(np.mean(tv)) if count else float("nan"),
                "target_std": target_std,
                "prediction_mean": float(np.mean(pv)) if count else float("nan"),
                "prediction_std": float(np.std(pv)) if count else float("nan"),
                "amplitude_ratio": (
                    float(np.std(pv) / target_std) if count > 1 and target_std > 0 else float("nan")
                ),
                "mse": model_sse / count if count else float("nan"),
                "prior_mse": prior_sse / count if count else float("nan"),
                "skill_vs_prior": skill,
                "pcc": _pearson(tv, pv),
                "ccc": lin_ccc(tv, pv),
                "dynamic_r2": _dynamic_r2(tv, pv),
                "within_cancer_observations": int(len(wc_t)),
                "within_cancer_pcc": _pearson(wc_t, wc_p),
                "within_cancer_ccc": lin_ccc(wc_t, wc_p),
                "within_cancer_dynamic_r2": _dynamic_r2(wc_t, wc_p),
                "eligible_variable_cpg": bool(eligible),
            }
        )
    return pd.DataFrame.from_records(records)


def per_sample_biological_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    prior: np.ndarray,
    *,
    sample_ids: np.ndarray | None = None,
    cancer_types: np.ndarray | None = None,
) -> pd.DataFrame:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    ids = np.arange(target.shape[0]).astype(str) if sample_ids is None else np.asarray(sample_ids).astype(str)
    cancers = np.repeat("unknown", target.shape[0]) if cancer_types is None else np.asarray(cancer_types).astype(str)
    records = []
    for row, sample_id in enumerate(ids):
        valid = np.isfinite(target[row]) & np.isfinite(prediction[row]) & np.isfinite(prior)
        t, p, b = target[row, valid], prediction[row, valid], prior[valid]
        model_mse = float(np.mean((t - p) ** 2)) if len(t) else float("nan")
        prior_mse = float(np.mean((t - b) ** 2)) if len(t) else float("nan")
        records.append(
            {
                "sample_id": sample_id,
                "cancer_type": cancers[row],
                "observed_cpgs": int(valid.sum()),
                "mse": model_mse,
                "prior_mse": prior_mse,
                "skill_vs_prior": 1.0 - model_mse / prior_mse if prior_mse > 0 else float("nan"),
                "mac_pcc": _pearson(t, p),
                "mac_ccc": lin_ccc(t, p),
            }
        )
    return pd.DataFrame.from_records(records)


def summarize_biological_fidelity(per_locus: pd.DataFrame, per_sample: pd.DataFrame) -> dict[str, Any]:
    eligible = per_locus.loc[per_locus["eligible_variable_cpg"]].copy()
    def median(column: str) -> float:
        values = eligible[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if len(values) else float("nan")

    skill = eligible["skill_vs_prior"].to_numpy(dtype=float)
    finite_skill = skill[np.isfinite(skill)]
    return {
        "variable_cpgs_evaluated": int(len(eligible)),
        "mas_skill_vs_prior_variable": median("skill_vs_prior"),
        "mean_skill_vs_prior_variable": (
            float(np.mean(finite_skill)) if len(finite_skill) else float("nan")
        ),
        "positive_skill_fraction_variable": (
            float(np.mean(finite_skill > 0)) if len(finite_skill) else float("nan")
        ),
        "mas_pcc_variable": median("pcc"),
        "mas_ccc_variable": median("ccc"),
        "mas_dynamic_r2_variable": median("dynamic_r2"),
        "within_cancer_mas_pcc_variable": median("within_cancer_pcc"),
        "within_cancer_mas_ccc_variable": median("within_cancer_ccc"),
        "within_cancer_mas_dynamic_r2_variable": median("within_cancer_dynamic_r2"),
        "median_amplitude_ratio_variable": median("amplitude_ratio"),
        "mac_pcc_median": float(np.nanmedian(per_sample["mac_pcc"])),
        "mac_ccc_median": float(np.nanmedian(per_sample["mac_ccc"])),
        "locus_skill_quantiles": _finite_quantiles(skill),
        "locus_ccc_quantiles": _finite_quantiles(eligible["ccc"].to_numpy(dtype=float)),
        "locus_dynamic_r2_quantiles": _finite_quantiles(eligible["dynamic_r2"].to_numpy(dtype=float)),
    }


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / positives)


def differential_effect_recovery(
    target: np.ndarray,
    prediction: np.ndarray,
    cancer_types: np.ndarray,
    *,
    config: BiologicalMetricConfig = BiologicalMetricConfig(),
) -> dict[str, Any]:
    """One-vs-rest differential methylation effect recovery by cancer type."""
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    cancer_types = np.asarray(cancer_types)
    per_contrast: dict[str, dict[str, float | int]] = {}
    for cancer in np.unique(cancer_types):
        inside = cancer_types == cancer
        outside = ~inside
        if inside.sum() < config.min_cancer_group_samples or outside.sum() < config.min_cancer_group_samples:
            continue
        real_effect = np.nanmean(target[inside], axis=0) - np.nanmean(target[outside], axis=0)
        pred_effect = np.nanmean(prediction[inside], axis=0) - np.nanmean(prediction[outside], axis=0)
        valid = np.isfinite(real_effect) & np.isfinite(pred_effect)
        r, p = real_effect[valid], pred_effect[valid]
        positives = np.abs(r) >= config.dmr_delta_threshold
        k = max(1, int(np.ceil(config.top_fraction * len(r)))) if len(r) else 0
        real_top = set(np.argsort(-np.abs(r))[:k].tolist())
        pred_top = set(np.argsort(-np.abs(p))[:k].tolist())
        union = real_top | pred_top
        per_contrast[str(cancer)] = {
            "cpgs": int(len(r)),
            "real_dmp_positives": int(positives.sum()),
            "effect_pearson": _pearson(r, p),
            "effect_spearman": (
                float(spearmanr(r, p).statistic) if len(r) > 1 and np.std(r) > 0 and np.std(p) > 0 else float("nan")
            ),
            "sign_agreement": float(np.mean(np.sign(r) == np.sign(p))) if len(r) else float("nan"),
            "dmp_auprc": _average_precision(positives, np.abs(p)),
            "top_effect_jaccard": float(len(real_top & pred_top) / len(union)) if union else float("nan"),
        }
    aggregate: dict[str, float] = {}
    for key in ("effect_pearson", "effect_spearman", "sign_agreement", "dmp_auprc", "top_effect_jaccard"):
        values = np.array([row[key] for row in per_contrast.values()], dtype=float)
        values = values[np.isfinite(values)]
        aggregate[f"macro_{key}"] = float(np.mean(values)) if len(values) else float("nan")
    return {"aggregate": aggregate, "per_cancer_one_vs_rest": per_contrast}



def regional_effect_recovery(
    target: np.ndarray,
    prediction: np.ndarray,
    cancer_types: np.ndarray,
    region_ids: np.ndarray,
    *,
    min_cpgs_per_region: int = 2,
    config: BiologicalMetricConfig = BiologicalMetricConfig(),
) -> dict[str, Any]:
    """Differential-effect recovery after aggregating CpGs into frozen regions.

    ``region_ids`` must come from a pre-specified annotation or a region caller
    frozen before model comparison.  The function never learns regions from a
    model's predictions, which keeps the comparison matched.
    """
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    region_ids = np.asarray(region_ids).astype(str)
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("target and prediction must share [samples, cpgs] shape")
    if len(region_ids) != target.shape[1]:
        raise ValueError("region_ids length does not match CpG dimension")

    valid_labels = region_ids != ""
    unique_regions, counts = np.unique(region_ids[valid_labels], return_counts=True)
    keep_regions = unique_regions[counts >= min_cpgs_per_region]
    if not len(keep_regions):
        return {
            "regions_evaluated": 0,
            "min_cpgs_per_region": int(min_cpgs_per_region),
            "aggregate": {},
            "per_cancer_one_vs_rest": {},
        }

    real_regions = np.empty((target.shape[0], len(keep_regions)), dtype=np.float64)
    pred_regions = np.empty_like(real_regions)
    for column, region in enumerate(keep_regions):
        members = region_ids == region
        real_regions[:, column] = np.nanmean(target[:, members], axis=1)
        pred_regions[:, column] = np.nanmean(prediction[:, members], axis=1)
    result = differential_effect_recovery(
        real_regions,
        pred_regions,
        cancer_types,
        config=config,
    )
    result["regions_evaluated"] = int(len(keep_regions))
    result["min_cpgs_per_region"] = int(min_cpgs_per_region)
    return result

def structural_fidelity(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    eligible_cpg_mask: np.ndarray | None = None,
    config: BiologicalMetricConfig = BiologicalMetricConfig(),
) -> dict[str, float | int]:
    """Preservation of CpG co-methylation and patient geometry.

    To keep full-test evaluation tractable, CpGs are sampled once with a fixed
    seed. The same subset is used for every compared model.
    """
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if eligible_cpg_mask is None:
        eligible_cpg_mask = (
            (np.isfinite(target).sum(axis=0) >= config.min_observed_samples)
            & (np.nanstd(target, axis=0) >= config.min_target_std)
        )
    eligible_cpg_mask = np.asarray(eligible_cpg_mask, dtype=bool)
    if eligible_cpg_mask.shape != (target.shape[1],):
        raise ValueError("eligible_cpg_mask length does not match CpG dimension")
    valid_columns = np.flatnonzero(eligible_cpg_mask)
    if len(valid_columns) > config.max_structure_cpgs:
        valid_columns = np.sort(
            np.random.default_rng(config.structure_seed).choice(
                valid_columns, config.max_structure_cpgs, replace=False
            )
        )
    if len(valid_columns) < 2:
        return {"structure_cpgs": int(len(valid_columns))}
    real = target[:, valid_columns]
    pred = prediction[:, valid_columns]
    # Mean-impute only for the structural summaries; pointwise metrics remain
    # strictly observed-only.
    real = np.where(np.isfinite(real), real, np.nanmean(real, axis=0, keepdims=True))
    pred = np.where(np.isfinite(pred), pred, np.nanmean(pred, axis=0, keepdims=True))
    def standardized(values: np.ndarray) -> np.ndarray:
        centred = values - values.mean(axis=0, keepdims=True)
        scale = values.std(axis=0, keepdims=True)
        return np.divide(
            centred,
            scale,
            out=np.zeros_like(centred),
            where=scale > 1e-12,
        )

    real_z = standardized(real)
    pred_z = standardized(pred)
    denominator = max(len(real) - 1, 1)
    real_corr = (real_z.T @ real_z) / denominator
    pred_corr = (pred_z.T @ pred_z) / denominator
    tri = np.triu_indices_from(real_corr, k=1)
    real_edges, pred_edges = real_corr[tri], pred_corr[tri]
    # Patient geometry is measured after per-CpG standardization so highly
    # variable loci do not dominate and constant predictions remain finite.
    real_distance = pdist(real_z, metric="euclidean")
    pred_distance = pdist(pred_z, metric="euclidean")

    k = min(config.knn_k, max(1, len(real) - 1))
    real_square = squareform(real_distance)
    pred_square = squareform(pred_distance)
    real_neighbors = np.argsort(real_square, axis=1)[:, 1 : k + 1]
    pred_neighbors = np.argsort(pred_square, axis=1)[:, 1 : k + 1]
    overlap = [len(set(a) & set(b)) / k for a, b in zip(real_neighbors, pred_neighbors, strict=True)]
    return {
        "structure_cpgs": int(len(valid_columns)),
        "comethylation_edge_pearson": _pearson(real_edges, pred_edges),
        "comethylation_edge_mae": float(np.mean(np.abs(real_edges - pred_edges))),
        "patient_distance_pearson": _pearson(real_distance, pred_distance),
        "patient_distance_spearman": float(spearmanr(real_distance, pred_distance).statistic),
        "patient_knn_overlap": float(np.mean(overlap)),
        "patient_knn_k": int(k),
    }
