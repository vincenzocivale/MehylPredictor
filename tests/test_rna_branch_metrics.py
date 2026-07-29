from __future__ import annotations

import numpy as np

from methylation_predictor.rna_branch.metrics import evaluate_predictions


def test_perfect_prediction_metrics():
    prior = np.array([0.2, 0.8])
    target = np.array([[0.1, 0.9], [0.3, 0.7], [0.25, 0.75], [0.15, 0.85]])
    cancer = np.array(["A", "A", "B", "B"])
    metrics = evaluate_predictions(target, target.copy(), prior, cancer)
    assert metrics["mse"] == 0.0
    assert metrics["skill_vs_prior"] == 1.0
    assert np.isclose(metrics["dynamic_skill"], 1.0)
    assert np.isclose(metrics["within_cancer_skill"], 1.0)
