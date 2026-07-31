"""Fixed linear probes used to measure accessible RNA information.

These probes are deliberately closed-form Ridge regressions.  There is no
methylation target, task decoder, neural head, or representation fine-tuning.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge

_EPS = 1e-12


@dataclass(slots=True)
class ProbeResult:
    alpha: float
    validation_mse: float
    test_mse: float
    global_r2: float
    variance_weighted_r2: float
    median_gene_r2: float
    fraction_positive_gene_r2: float
    mean_gene_pearson: float


def standardize_train(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    train = np.asarray(train, dtype=np.float64)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    arrays = (train,) + others
    return tuple(((np.asarray(value, dtype=np.float64) - mean) / scale).astype(np.float32) for value in arrays)


def _mse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.square(y - pred)))


def _gene_r2(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    residual = np.square(y - pred).sum(axis=0)
    total = np.square(y - y.mean(axis=0, keepdims=True)).sum(axis=0)
    return 1.0 - residual / np.maximum(total, _EPS)


def _gene_pearson(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    yc = y - y.mean(axis=0, keepdims=True)
    pc = pred - pred.mean(axis=0, keepdims=True)
    numerator = np.sum(yc * pc, axis=0)
    denominator = np.sqrt(np.sum(yc**2, axis=0) * np.sum(pc**2, axis=0))
    values = numerator / np.maximum(denominator, _EPS)
    values[denominator < _EPS] = np.nan
    return values


def fit_ridge_probe(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
    alphas: tuple[float, ...],
) -> ProbeResult:
    x_train, x_validation, x_test = standardize_train(x_train, x_validation, x_test)
    y_train, y_validation, y_test = standardize_train(y_train, y_validation, y_test)
    best_alpha = None
    best_mse = float("inf")
    for alpha in alphas:
        model = Ridge(alpha=float(alpha), fit_intercept=True)
        model.fit(x_train, y_train)
        score = _mse(y_validation, model.predict(x_validation))
        if score < best_mse:
            best_mse, best_alpha = score, float(alpha)
    if best_alpha is None:
        raise RuntimeError("ridge alpha selection failed")
    combined_x = np.concatenate([x_train, x_validation], axis=0)
    combined_y = np.concatenate([y_train, y_validation], axis=0)
    model = Ridge(alpha=best_alpha, fit_intercept=True)
    model.fit(combined_x, combined_y)
    pred = model.predict(x_test)
    residual = np.square(y_test - pred).sum()
    total = np.square(y_test - y_test.mean(axis=0, keepdims=True)).sum()
    gene_r2 = _gene_r2(y_test, pred)
    gene_var = np.var(y_test, axis=0)
    pearson = _gene_pearson(y_test, pred)
    return ProbeResult(
        alpha=best_alpha,
        validation_mse=best_mse,
        test_mse=_mse(y_test, pred),
        global_r2=float(1.0 - residual / max(total, _EPS)),
        variance_weighted_r2=float(np.average(gene_r2, weights=np.maximum(gene_var, _EPS))),
        median_gene_r2=float(np.median(gene_r2)),
        fraction_positive_gene_r2=float(np.mean(gene_r2 > 0)),
        mean_gene_pearson=float(np.nanmean(pearson)),
    )


def training_group_means(y_train: np.ndarray, cancer_train: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    global_mean = np.asarray(y_train, dtype=np.float64).mean(axis=0)
    means = {
        str(label): np.asarray(y_train, dtype=np.float64)[np.asarray(cancer_train).astype(str) == str(label)].mean(axis=0)
        for label in np.unique(np.asarray(cancer_train).astype(str))
    }
    return global_mean, means


def residualize_using_training_means(
    y: np.ndarray,
    cancer: np.ndarray,
    global_mean: np.ndarray,
    group_means: dict[str, np.ndarray],
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    cancer = np.asarray(cancer).astype(str)
    result = np.empty_like(y)
    for index, label in enumerate(cancer):
        result[index] = y[index] - group_means.get(str(label), global_mean)
    return result.astype(np.float32)
