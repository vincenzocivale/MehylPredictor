"""Optional token-level audits for contextual gene embeddings."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from .io import TokenData
from .metrics import effective_rank_entropy, covariance_spectrum, linear_cka

_EPS = 1e-12


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y)
    pred = np.asarray(pred).reshape(y.shape)
    residual = float(np.square(y - pred).sum())
    total = float(np.square(y - y.mean(axis=0, keepdims=True)).sum())
    return 1.0 - residual / max(total, _EPS)


def _select_alpha(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, alphas: tuple[float, ...]) -> float:
    best = (float("inf"), float(alphas[0]))
    for alpha in alphas:
        model = Ridge(alpha=float(alpha)).fit(x_train, y_train)
        mse = float(np.mean(np.square(y_val - model.predict(x_val))))
        best = min(best, (mse, float(alpha)))
    return best[1]


def audit_tokens(
    token_data: TokenData,
    metadata_ids: np.ndarray,
    metadata_splits: np.ndarray,
    metadata_cancer: np.ndarray,
    alphas: tuple[float, ...],
    global_pc_count: int,
) -> pd.DataFrame:
    lookup = {str(value): index for index, value in enumerate(metadata_ids)}
    rows = [lookup[str(value)] for value in token_data.sample_ids]
    split = np.asarray(metadata_splits, dtype=object)[rows].astype(str)
    cancer = np.asarray(metadata_cancer, dtype=object)[rows].astype(str)
    train, val, test = split == "train", split == "validation", split == "test"
    if not train.any() or not val.any() or not test.any():
        raise ValueError("token audit requires train, validation and test samples")
    expression = np.asarray(token_data.expression, dtype=np.float32)
    token_ids = np.asarray(token_data.token_ids, dtype=np.float32)
    # A gene's bin/token id is the simple scalar baseline for its continuous expression.
    records: list[dict[str, object]] = []

    # Global within-cancer expression PCs are RNA-only patient-state targets.
    residual = expression.astype(np.float64).copy()
    train_global = expression[train].mean(axis=0)
    train_means = {
        label: expression[train & (cancer == label)].mean(axis=0)
        for label in np.unique(cancer[train])
        if np.any(train & (cancer == label))
    }
    for index, label in enumerate(cancer):
        residual[index] -= train_means.get(label, train_global)
    pc_count = min(global_pc_count, residual[train].shape[0] - 1, residual.shape[1])
    pca = PCA(n_components=max(pc_count, 1), random_state=0).fit(residual[train])
    global_pc = pca.transform(residual)

    for layer, tokens in token_data.layers.items():
        tokens = np.asarray(tokens, dtype=np.float32)
        pooled = tokens.mean(axis=1)
        records.append({
            "layer": layer,
            "scope": "pooled",
            "metric": "effective_rank_test",
            "value": effective_rank_entropy(covariance_spectrum(pooled[test])),
        })
        if layer in token_data.pooled_layers:
            records.append({
                "layer": layer,
                "scope": "pooled",
                "metric": "sampled_vs_full_pooling_cka",
                "value": linear_cka(pooled, token_data.pooled_layers[layer]),
            })

        own_r2: list[float] = []
        scalar_r2: list[float] = []
        global_r2: list[float] = []
        scalar_global_r2: list[float] = []
        for gene_index in range(tokens.shape[1]):
            x = tokens[:, gene_index, :]
            y = expression[:, gene_index:gene_index + 1]
            alpha = _select_alpha(x[train], y[train], x[val], y[val], alphas)
            model = Ridge(alpha=alpha).fit(np.concatenate([x[train], x[val]]), np.concatenate([y[train], y[val]]))
            own_r2.append(_r2(y[test], model.predict(x[test])))

            scalar = token_ids[:, gene_index:gene_index + 1]
            scalar_alpha = _select_alpha(scalar[train], y[train], scalar[val], y[val], alphas)
            scalar_model = Ridge(alpha=scalar_alpha).fit(
                np.concatenate([scalar[train], scalar[val]]), np.concatenate([y[train], y[val]])
            )
            scalar_r2.append(_r2(y[test], scalar_model.predict(scalar[test])))

            alpha_global = _select_alpha(x[train], global_pc[train], x[val], global_pc[val], alphas)
            global_model = Ridge(alpha=alpha_global).fit(
                np.concatenate([x[train], x[val]]), np.concatenate([global_pc[train], global_pc[val]])
            )
            global_r2.append(_r2(global_pc[test], global_model.predict(x[test])))

            scalar_global_alpha = _select_alpha(scalar[train], global_pc[train], scalar[val], global_pc[val], alphas)
            scalar_global_model = Ridge(alpha=scalar_global_alpha).fit(
                np.concatenate([scalar[train], scalar[val]]), np.concatenate([global_pc[train], global_pc[val]])
            )
            scalar_global_r2.append(_r2(global_pc[test], scalar_global_model.predict(scalar[test])))

        for metric, values in (
            ("median_own_expression_r2", own_r2),
            ("median_scalar_token_expression_r2", scalar_r2),
            ("median_context_gain_expression_r2", np.asarray(own_r2) - np.asarray(scalar_r2)),
            ("median_global_within_cancer_pc_r2", global_r2),
            ("median_scalar_global_pc_r2", scalar_global_r2),
            ("median_context_gain_global_pc_r2", np.asarray(global_r2) - np.asarray(scalar_global_r2)),
        ):
            records.append({"layer": layer, "scope": "gene_token", "metric": metric, "value": float(np.median(values))})
    return pd.DataFrame.from_records(records)
