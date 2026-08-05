from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from methylation_predictor.rna_encoder_quality.probes import fit_ridge_probe
from methylation_predictor.rna_encoder_quality.runner import _encoder_ranking


def test_ridge_alpha_is_selected_inside_training_split() -> None:
    rng = np.random.default_rng(17)
    n_train, n_val, n_test = 120, 40, 40
    latent_train = rng.normal(size=(n_train, 8))
    latent_val = rng.normal(size=(n_val, 8))
    latent_test = rng.normal(size=(n_test, 8))
    weights = rng.normal(size=(8, 24))

    def target(x: np.ndarray) -> np.ndarray:
        return x @ weights + 0.1 * rng.normal(size=(len(x), weights.shape[1]))

    groups = np.asarray([f"cancer_{index % 4}" for index in range(n_train)])
    result = fit_ridge_probe(
        latent_train,
        latent_val,
        latent_test,
        target(latent_train),
        target(latent_val),
        target(latent_test),
        (0.01, 0.1, 1.0, 10.0),
        cv_folds=5,
        seed=17,
        cv_groups=groups,
    )
    assert result.alpha in {0.01, 0.1, 1.0, 10.0}
    assert result.train_cv_global_r2 > 0.9
    assert result.validation_global_r2 > 0.9
    assert result.global_r2 > 0.9


def test_encoder_ranking_uses_train_cv_not_test() -> None:
    config = SimpleNamespace(analysis=SimpleNamespace(primary_encoder_dimension=256))
    reconstruction = pd.DataFrame(
        [
            {
                "representation": "encoder_a__pca256",
                "target": "within_cancer_rna",
                "alpha": 1.0,
                "train_cv_mse": 0.4,
                "train_cv_global_r2": 0.60,
                "train_cv_mean_gene_pearson": 0.50,
                "validation_mse": 0.45,
                "validation_global_r2": 0.55,
                "validation_mean_gene_pearson": 0.48,
                "test_mse": 0.9,
                "global_r2": 0.10,
                "variance_weighted_r2": 0.10,
                "median_gene_r2": 0.05,
                "fraction_positive_gene_r2": 0.55,
                "mean_gene_pearson": 0.20,
            },
            {
                "representation": "encoder_b__pca256",
                "target": "within_cancer_rna",
                "alpha": 1.0,
                "train_cv_mse": 0.5,
                "train_cv_global_r2": 0.50,
                "train_cv_mean_gene_pearson": 0.45,
                "validation_mse": 0.4,
                "validation_global_r2": 0.58,
                "validation_mean_gene_pearson": 0.50,
                "test_mse": 0.2,
                "global_r2": 0.80,
                "variance_weighted_r2": 0.75,
                "median_gene_r2": 0.70,
                "fraction_positive_gene_r2": 0.90,
                "mean_gene_pearson": 0.85,
            },
            *[
                {
                    "representation": name,
                    "target": "total_rna",
                    "alpha": 1.0,
                    "train_cv_mse": 0.2,
                    "train_cv_global_r2": score,
                    "train_cv_mean_gene_pearson": score,
                    "validation_mse": 0.2,
                    "validation_global_r2": score,
                    "validation_mean_gene_pearson": score,
                    "test_mse": 0.2,
                    "global_r2": score,
                    "variance_weighted_r2": score,
                    "median_gene_r2": score,
                    "fraction_positive_gene_r2": 1.0,
                    "mean_gene_pearson": score,
                }
                for name, score in (("encoder_a__pca256", 0.7), ("encoder_b__pca256", 0.8))
            ],
        ]
    )
    ranking = _encoder_ranking(config, reconstruction, pd.DataFrame(), pd.DataFrame())
    assert ranking.iloc[0]["representation"] == "encoder_a__pca256"
    assert not bool(ranking.iloc[0]["validation_confirms_cv_winner"])
