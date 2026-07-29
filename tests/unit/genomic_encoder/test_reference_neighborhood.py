import unittest

import numpy as np
import pandas as pd

from methylation_predictor.genomic_encoder.reference_neighborhood import build_neighborhood_features


class ReferenceNeighborhoodTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"cpg_idx": [0, 1, 2, 3, 4], "chromosome": ["chr1"] * 5,
                                   "position": [100, 200, 300, 20_000, 30_000],
                                   "mean_train": [.1, .2, .9, .4, .5], "cgi_category": ["a", "a", "b", "b", "b"],
                                   "within_cpg_variance": [.01] * 5, "n_train": [10] * 5})

    def test_target_value_is_not_used_as_its_own_neighbor(self):
        features, baseline = build_neighborhood_features(self.frame.iloc[[1]], self.frame, target="mean_train",
            chromosome="chromosome", position="position", cgi="cgi_category", variance="within_cpg_variance", count="n_train",
            max_neighbors=1, windows=(1000,), exclusion_radius=0)
        # CpG at 200 sees values .1 and .9, never its own .2.
        self.assertAlmostEqual(features.iloc[0]["neighbor_mean"], .5)
        self.assertNotAlmostEqual(baseline.loc[1], .2)

    def test_block_ood_removes_local_anchors(self):
        features, baseline = build_neighborhood_features(self.frame.iloc[[1]], self.frame, target="mean_train",
            chromosome="chromosome", position="position", cgi="cgi_category", variance=None, count=None,
            max_neighbors=2, windows=(1000, 100000), exclusion_radius=10_000)
        self.assertTrue(np.isnan(baseline.loc[1]) or .4 <= baseline.loc[1] <= .5)
        self.assertTrue(np.isnan(features.iloc[0]["w1000_mean"]))

    def test_reference_values_change_feature_but_target_value_does_not(self):
        target = self.frame.iloc[[4]]
        _, first = build_neighborhood_features(target, self.frame.iloc[:4], target="mean_train", chromosome="chromosome",
            position="position", cgi=None, variance=None, count=None, max_neighbors=2, windows=(100000,), exclusion_radius=0)
        altered = self.frame.iloc[:4].copy(); altered.loc[3, "mean_train"] = .99
        _, second = build_neighborhood_features(target, altered, target="mean_train", chromosome="chromosome",
            position="position", cgi=None, variance=None, count=None, max_neighbors=2, windows=(100000,), exclusion_radius=0)
        self.assertNotEqual(first.loc[4], second.loc[4])


if __name__ == "__main__": unittest.main()
