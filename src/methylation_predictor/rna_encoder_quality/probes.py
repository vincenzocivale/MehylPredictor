"""Fixed linear probes used to measure accessible RNA information.

These probes are deliberately closed-form Ridge regressions.  There is no
methylation target, task decoder, neural head, or representation fine-tuning.
Hyper-parameters are selected only inside the training split; validation is a
confirmation split and test metrics remain exploratory.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold

_EPS = 1e-12


@dataclass(slots=True)
class ProbeResult:
    alpha: float
    train_cv_mse: float
    train_cv_global_r2: float
    train_cv_mean_gene_pearson: float
    validation_mse: float
    validation_global_r2: float
    validation_mean_gene_pearson: float
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


def _global_r2(y: np.ndarray, pred: np.ndarray) -> float:
    residual = np.square(y - pred).sum()
    total = np.square(y - y.mean(axis=0, keepdims=True)).sum()
    return float(1.0 - residual / max(float(total), _EPS))


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


def _mean_gene_pearson(y: np.ndarray, pred: np.ndarray) -> float:
    values = _gene_pearson(y, pred)
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")


def _folds(
    n_samples: int,
    requested_folds: int,
    seed: int,
    groups: np.ndarray | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_samples < 2:
        raise ValueError("at least two training samples are required for Ridge CV")
    requested_folds = max(2, min(int(requested_folds), n_samples))
    if groups is not None:
        labels = np.asarray(groups).astype(str)
        if len(labels) != n_samples:
            raise ValueError("ridge CV group labels do not match training samples")
        _, counts = np.unique(labels, return_counts=True)
        if len(counts) > 1 and int(counts.min()) >= 2:
            n_splits = min(requested_folds, int(counts.min()))
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            return [(train, holdout) for train, holdout in splitter.split(np.zeros(n_samples), labels)]
    splitter = KFold(n_splits=requested_folds, shuffle=True, random_state=seed)
    return [(train, holdout) for train, holdout in splitter.split(np.arange(n_samples))]


def _select_alpha_train_cv(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alphas: tuple[float, ...],
    cv_folds: int,
    seed: int,
    groups: np.ndarray | None,
) -> tuple[float, float, np.ndarray]:
    splits = _folds(len(x_train), cv_folds, seed, groups)
    best_alpha: float | None = None
    best_mse = float("inf")
    best_predictions: np.ndarray | None = None
    for alpha in alphas:
        predictions = np.empty_like(y_train, dtype=np.float32)
        for fit_rows, holdout_rows in splits:
            model = Ridge(alpha=float(alpha), fit_intercept=True)
            model.fit(x_train[fit_rows], y_train[fit_rows])
            predictions[holdout_rows] = model.predict(x_train[holdout_rows]).astype(np.float32)
        score = _mse(y_train, predictions)
        # Stable tie-breaking prevents numerical noise from preferring a larger
        # regularizer when two grid points are effectively equivalent.
        if score < best_mse - 1e-12:
            best_mse = score
            best_alpha = float(alpha)
            best_predictions = predictions
    if best_alpha is None or best_predictions is None:
        raise RuntimeError("ridge alpha selection failed")
    return best_alpha, best_mse, best_predictions


def fit_ridge_probe(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
    alphas: tuple[float, ...],
    *,
    cv_folds: int = 5,
    seed: int = 17,
    cv_groups: np.ndarray | None = None,
) -> ProbeResult:
    """Fit a matched linear readout without validation-based tuning.

    Feature and target scaling are estimated on the complete training split.
    The Ridge penalty is selected by deterministic out-of-fold predictions on
    that split.  The fixed penalty is then evaluated on validation; test is
    evaluated after refitting on train+validation in the same train-defined
    coordinate system.
    """
    x_train, x_validation, x_test = standardize_train(x_train, x_validation, x_test)
    y_train, y_validation, y_test = standardize_train(y_train, y_validation, y_test)
    best_alpha, train_cv_mse, train_cv_pred = _select_alpha_train_cv(
        x_train,
        y_train,
        alphas,
        cv_folds,
        seed,
        cv_groups,
    )

    validation_model = Ridge(alpha=best_alpha, fit_intercept=True)
    validation_model.fit(x_train, y_train)
    validation_pred = validation_model.predict(x_validation)

    combined_x = np.concatenate([x_train, x_validation], axis=0)
    combined_y = np.concatenate([y_train, y_validation], axis=0)
    test_model = Ridge(alpha=best_alpha, fit_intercept=True)
    test_model.fit(combined_x, combined_y)
    test_pred = test_model.predict(x_test)

    gene_r2 = _gene_r2(y_test, test_pred)
    gene_var = np.var(y_test, axis=0)
    pearson = _gene_pearson(y_test, test_pred)
    return ProbeResult(
        alpha=best_alpha,
        train_cv_mse=train_cv_mse,
        train_cv_global_r2=_global_r2(y_train, train_cv_pred),
        train_cv_mean_gene_pearson=_mean_gene_pearson(y_train, train_cv_pred),
        validation_mse=_mse(y_validation, validation_pred),
        validation_global_r2=_global_r2(y_validation, validation_pred),
        validation_mean_gene_pearson=_mean_gene_pearson(y_validation, validation_pred),
        test_mse=_mse(y_test, test_pred),
        global_r2=_global_r2(y_test, test_pred),
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
