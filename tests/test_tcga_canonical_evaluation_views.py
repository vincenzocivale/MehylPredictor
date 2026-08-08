"""E: exact finite-target regression counts for the three official Array
evaluation views of the released tcga_mix_chr1 checkpoint's split.

These read real (non-trivial) slices of the 12GB Array beta matrix, so this
module is the slow part of the suite (a minute or so) -- that cost is
inherent to computing an exact count over real data, not something to
optimize away for the sake of a "fast" test.
"""
from __future__ import annotations

import pytest

from methylation_predictor.tcga_canonical import load_protocol

EXPECTED_FINITE_COUNTS = {
    "train_cpg_x_val_sample": 30_574_946,
    "val_cpg_x_train_sample": 55_155_121,
    "val_cpg_x_val_sample": 6_129_547,
}


@pytest.mark.slow
def test_evaluation_view_shapes(bundle):
    protocol = load_protocol("tcga_mix_chr1", bundle=bundle)
    views = protocol.evaluation_views()
    assert set(views) == set(EXPECTED_FINITE_COUNTS)

    v1 = views["train_cpg_x_val_sample"]
    assert len(v1.sample_idx) == 918 and len(v1.cpg_idx) == 33885
    v2 = views["val_cpg_x_train_sample"]
    assert len(v2.sample_idx) == 8260 and len(v2.cpg_idx) == 6742
    v3 = views["val_cpg_x_val_sample"]
    assert len(v3.sample_idx) == 918 and len(v3.cpg_idx) == 6742


@pytest.mark.slow
def test_evaluation_view_finite_counts_exact(bundle):
    protocol = load_protocol("tcga_mix_chr1", bundle=bundle)
    counts = protocol.evaluation_finite_counts()
    assert counts == EXPECTED_FINITE_COUNTS
