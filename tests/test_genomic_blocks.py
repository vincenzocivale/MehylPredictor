from __future__ import annotations

import numpy as np

from methylation_predictor.diagnostics.methylprophet.stage_d1 import genomic_blocks


def test_same_local_position_different_chromosome_do_not_collide():
    chromosome = np.array(["chr1", "chr2", "chr1", "chr2"])
    position = np.array([1_000_000, 1_000_000, 6_000_000, 6_000_000])
    blocks = genomic_blocks(chromosome, position, block_bp=5_000_000)
    # 4 CpGs, 4 distinct (chromosome, 5Mb-bin) combinations -> 4 singleton blocks
    assert len(blocks) == 4
    assert sorted(len(b) for b in blocks) == [1, 1, 1, 1]


def test_single_chromosome_matches_position_only_binning():
    chromosome = np.array(["chr1"] * 6)
    position = np.array([0, 1_000_000, 5_000_001, 6_000_000, 10_000_001, 10_500_000])
    blocks = genomic_blocks(chromosome, position, block_bp=5_000_000)
    naive_labels = position // 5_000_000
    naive_blocks = {label: np.flatnonzero(naive_labels == label).tolist() for label in np.unique(naive_labels)}
    recovered = {tuple(sorted(b.tolist())) for b in blocks}
    expected = {tuple(sorted(v)) for v in naive_blocks.values()}
    assert recovered == expected
