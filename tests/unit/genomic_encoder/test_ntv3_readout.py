import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from methylation_predictor.genomic_encoder.ntv3_prior_common import base_to_output, centred_window, locus_vectors, reverse_complement
from methylation_predictor.genomic_encoder.ntv3_prior import _load, main


class NTv3PriorTest(unittest.TestCase):
    def test_centre_padding_and_reverse_complement(self):
        genome = "CGAAA"
        self.assertEqual(centred_window(genome, 1, 8), "NNNCGAAA")
        self.assertEqual(centred_window("AAACGAAA", 4, 8)[3:5], "CG")
        self.assertEqual(reverse_complement("ACGTN"), "NACGT")
        with self.assertRaises(ValueError): centred_window(genome, 2, 8)

    def test_output_mapping_and_local_pooling(self):
        values = np.arange(16, dtype=np.float32).reshape(8, 2)
        result = locus_vectors(values, 8)
        self.assertEqual(base_to_output(3, 8, 8), 3)
        self.assertTrue(np.array_equal(result["centre"], values[[3, 4]].mean(axis=0)))
        self.assertTrue(np.array_equal(result["pool_512"], values.mean(axis=0)))

    def test_probe_runner_with_npz_and_rc_average(self):
        rng = np.random.default_rng(4); n, d = 80, 6
        cpg = np.arange(n, dtype=np.int64); z = rng.normal(size=(n, d)).astype(np.float32)
        frame = pd.DataFrame({"cpg_idx": cpg, "chromosome": "chr1", "position": cpg * 5_000_000 + 1,
                              "mean_train": 1 / (1 + np.exp(-z[:, 0])), "cgi_category": "island"})
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); table, forward, reverse, out = tmp / "table.parquet", tmp / "f.npz", tmp / "r.npz", tmp / "out"
            frame.to_parquet(table, index=False)
            payload = {"cpg_idx": cpg, "centre": z, "transformer": z, "pool_32": z, "pool_128": z, "pool_512": z}
            np.savez_compressed(forward, **payload); np.savez_compressed(reverse, **payload)
            old = sys.argv
            try:
                sys.argv = ["runner", "--input", str(table), "--embeddings", str(forward), "--reverse-embeddings", str(reverse), "--output-dir", str(out), "--representation", "multiscale", "--split", "random", "--epochs", "2", "--device", "cpu"]
                main()
            finally: sys.argv = old
            report = json.loads((out / "metrics.json").read_text())
            self.assertIn("mlp_ensemble", report["validation"])
            self.assertEqual(len(pd.read_parquet(out / "locus_predictions.parquet")), report["rows"]["validation"] + report["rows"]["test"])
            _, projected = _load(forward, "multiscale", reverse)
            self.assertEqual(projected.shape, (n, d))
