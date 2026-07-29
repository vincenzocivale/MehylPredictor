import numpy as np
import pandas as pd
import unittest

from methylation_predictor.diagnostics.methylprophet.locus_decomposition import GroupLayout, Moments, summarize
from methylation_predictor.diagnostics.methylprophet.gene_encoder_effect import _metrics
from methylation_predictor.diagnostics.methylprophet.derive_tcga_sample_metadata import derive
from methylation_predictor.diagnostics.methylprophet.empirical_hybrid import Stats


def state_from_rows(rows):
    frame = pd.DataFrame(rows, columns=["cpg_idx", "sample_idx", "pred_methyl", "gt_methyl"])
    layout = GroupLayout(
        cpg_ids=np.sort(frame.cpg_idx.unique()).astype(np.int64),
        sample_ids=np.sort(frame.sample_idx.unique()).astype(np.int64),
    )
    state = Moments.empty(layout)
    for cpg, part in frame.groupby("cpg_idx"):
        idx = np.searchsorted(layout.cpg_ids, cpg)
        y, p = part.gt_methyl.to_numpy(float), part.pred_methyl.to_numpy(float)
        state.n[idx] = len(part)
        state.sum_y[idx], state.sum_y2[idx] = y.sum(), np.dot(y, y)
        state.sum_p[idx], state.sum_p2[idx] = p.sum(), np.dot(p, p)
        state.sum_py[idx] = np.dot(p, y)
        state.sse += np.dot(p - y, p - y)
        state.sae += np.abs(p - y).sum()
        state.observed_rows += len(part)
    return state


class LocusDominanceTest(unittest.TestCase):
    def test_exact_mse_decomposition_on_unbalanced_matrix(self):
        state = state_from_rows([(1, 1, 0.1, 0.0), (1, 2, 0.3, 0.4), (2, 1, 0.7, 0.8)])
        metrics, _ = summarize(state)
        self.assertLess(abs(metrics["mse_decomposition_closure"]), 1e-12)
        self.assertLess(abs(metrics["f_locus"] + metrics["f_within"] - 1.0), 1e-12)

    def test_static_prediction_has_zero_dynamic_skill(self):
        state = state_from_rows([(1, 1, 0.2, 0.1), (1, 2, 0.2, 0.3), (2, 1, 0.8, 0.7), (2, 2, 0.8, 0.9)])
        metrics, _ = summarize(state)
        self.assertAlmostEqual(metrics["dynamic_skill"], 0.0)

    def test_correlated_overscaled_residuals_have_negative_skill(self):
        # Residual truth is [-1, 1]; predicted residual is [-3, 3].
        state = state_from_rows([(1, 1, -3.0, -1.0), (1, 2, 3.0, 1.0)])
        metrics, _ = summarize(state)
        self.assertGreater(metrics["residual_correlation_pooled"], 0.99)
        self.assertLess(metrics["dynamic_skill"], 0)

    def test_total_skill_requires_explicit_training_prior(self):
        state = state_from_rows([(1, 1, 0.1, 0.1), (1, 2, 0.3, 0.3)])
        metrics, _ = summarize(state)
        self.assertNotIn("total_skill", metrics)
        metrics, _ = summarize(state, pd.DataFrame({"cpg_idx": [1], "mean_train": [0.2]}))
        self.assertEqual(metrics["total_skill"], 1.0)

    def test_gene_intervention_separates_effect_from_fixed_baseline(self):
        frame = pd.DataFrame(
            {
                "cpg_idx": [1, 1, 1, 1],
                "sample_idx": [1, 1, 2, 2],
                "condition": ["factual", "fixed", "factual", "fixed"],
                "pred_methyl": [0.1, 0.2, 0.9, 0.8],
                "gt_methyl": [0.1, 0.1, 0.9, 0.9],
            }
        )
        metrics = _metrics(frame)
        self.assertAlmostEqual(metrics["delta_gene_rms"], 0.1)
        self.assertAlmostEqual(metrics["incremental_skill_vs_fixed"], 1.0)

    def test_empirical_prior_hybrid_calibration_corrects_dynamic_scale(self):
        # Both loci have a reliable zero prior.  MP gets the direction right but
        # is three times too large, so alpha=1/3 reconstructs the targets exactly.
        state = Stats()
        state.pred_mean = {1: 0.0, 2: 0.0}
        state.add(
            np.array([1, 2, 1, 2]), np.array([10, 10, 11, 11]),
            np.array([0.1, -0.1, -0.1, 0.1]), np.array([0.3, -0.3, -0.3, 0.3]),
            np.zeros(4), 1 / 3,
        )
        metrics = state.metrics()["models"]
        self.assertLess(metrics["hybrid_calibrated"]["mse"], 1e-15)
        self.assertGreater(metrics["hybrid_calibrated"]["total_skill_vs_prior"], 0.999999)
        self.assertGreater(metrics["hybrid_alpha_1"]["mse"], metrics["prior"]["mse"])
