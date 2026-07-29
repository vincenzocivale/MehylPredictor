import unittest
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from methylation_predictor.genomic_encoder.static_prior import _partition, _target, _inverse_target, main
from methylation_predictor.genomic_encoder.downstream_evaluation import _alpha, _sse


class StaticPriorTest(unittest.TestCase):
    def test_logit_round_trip(self):
        y = np.array([0.0, 0.01, 0.5, 0.99, 1.0])
        result = _inverse_target(_target(y, "logit"), "logit")
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))
        self.assertAlmostEqual(result[2], 0.5)

    def test_block_split_never_splits_a_block(self):
        frame = pd.DataFrame({"cpg_idx": [1, 2, 3, 4], "chromosome": ["chr1"] * 4,
                              "position": [100, 999_999, 1_000_001, 1_500_000]})
        split = _partition(frame, "block", "cpg_idx", "chromosome", "position", 1_000_000, 7)
        self.assertEqual(split.iloc[0], split.iloc[1])
        self.assertEqual(len(split), 4)

    def test_runner_writes_metrics_and_predictions(self):
        # A complete tiny run catches schema, preprocessing and output regressions.
        rng = np.random.default_rng(11)
        n = 400
        frame = pd.DataFrame({
            "cpg_idx": np.arange(n), "chromosome": np.where(np.arange(n) % 2, "chr1", "chr2"),
            "position": np.arange(n) * 1_000_000, "cgi_category": np.where(np.arange(n) % 3, "shore", "island"),
            "gc_content": rng.uniform(.2, .8, n),
        })
        frame["mean_train"] = np.clip(.1 + .7 * frame.gc_content + (frame.cgi_category == "island") * .05, 0, 1)
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "input.parquet", Path(tmp) / "out"
            frame.to_parquet(source, index=False)
            old = sys.argv
            try:
                sys.argv = ["runner", "--input", str(source), "--output-dir", str(output), "--split", "random",
                            "--numeric", "gc_content", "--categorical", "cgi_category,chromosome", "--mlp-max-iter", "15"]
                main()
            finally:
                sys.argv = old
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual(metrics["rows"]["test"], len(pd.read_parquet(output / "test_predictions.parquet")))
            self.assertIn("tabular_mlp", metrics["test"])

    def test_downstream_alpha_uses_only_validation_sufficient_statistics(self):
        # y residual equals half of the MP residual, so alpha must be 0.5.
        frame = pd.DataFrame({"n": [2, 2], "gt_mean": [0.0, 0.0], "pred_mean": [0.0, 0.0],
                              "pred_mean_train": [0.0, 0.0], "pred_within_ss": [2.0, 2.0],
                              "gt_within_ss": [0.5, 0.5], "residual_cross_product": [1.0, 1.0]})
        alpha = _alpha(frame, np.zeros(2))
        self.assertAlmostEqual(alpha, 0.5)
        self.assertAlmostEqual(_sse(frame, np.zeros(2), alpha).sum(), 0.0)
