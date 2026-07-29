import unittest
import pandas as pd

from methylation_predictor.genomic_encoder.build_ntv3_borzoi_fold_manifest import _assign


class BorzoiFoldTest(unittest.TestCase):
    def test_assigns_bed_half_open_intervals(self):
        loci = pd.DataFrame({"cpg_idx": [1, 2, 3], "chromosome": ["chr1"] * 3, "position": [1, 100, 101]})
        intervals = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 100], "end": [100, 200], "fold": ["fold3", "fold4"], "split": ["test", "val"]})
        result = _assign(loci, intervals)
        self.assertEqual(result.borzoi_folds.tolist()[:2], ["fold3", "fold3"])
        self.assertEqual(result.borzoi_folds.iloc[2], "fold4")
        self.assertEqual(result.posttraining_exposure.tolist(), ["held_out_from_train"] * 3)

    def test_overlapping_contigs_are_conservatively_train_exposed(self):
        loci = pd.DataFrame({"cpg_idx": [1], "chromosome": ["chr1"], "position": [20]})
        intervals = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 10], "end": [30, 40], "fold": ["fold2", "fold4"], "split": ["train", "val"]})
        result = _assign(loci, intervals)
        self.assertEqual(result.posttraining_exposure.iloc[0], "train_exposed")
