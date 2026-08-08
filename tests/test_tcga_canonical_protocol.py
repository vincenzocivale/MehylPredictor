"""B: chr1 protocol splits. C: chr123 Array splits. D: source views."""
from __future__ import annotations

import numpy as np
import pytest

from methylation_predictor.tcga_canonical import load_protocol


@pytest.fixture(scope="module")
def chr1_protocol(bundle):
    return load_protocol("tcga_mix_chr1", bundle=bundle)


@pytest.fixture(scope="module")
def chr123_protocol(bundle):
    return load_protocol("tcga_mix_chr123", bundle=bundle)


def test_chr1_array_sample_split_counts(chr1_protocol):
    assert len(chr1_protocol.array_train_sample_idx) == 8260
    assert len(chr1_protocol.array_val_sample_idx) == 918
    assert set(chr1_protocol.array_train_sample_idx.tolist()).isdisjoint(
        chr1_protocol.array_val_sample_idx.tolist()
    )


def test_chr1_array_cpg_split_counts(chr1_protocol):
    assert len(chr1_protocol.array_train_cpg_idx) == 33885
    assert len(chr1_protocol.array_val_cpg_idx) == 6742
    assert set(chr1_protocol.array_train_cpg_idx.tolist()).isdisjoint(chr1_protocol.array_val_cpg_idx.tolist())
    assert len(chr1_protocol.array_train_cpg_idx) + len(chr1_protocol.array_val_cpg_idx) == 40627


def test_chr1_source_views(bundle):
    root = bundle.root
    views = {
        "epic_chr1": 71748,
        "wgbs_chr1": 1999548,
        "epic_chr123": 172723,
        "wgbs_chr123": 5396437,
    }
    for filename, expected in views.items():
        path = root / "protocols" / "_source_views" / f"{filename}_cpg_idx.npy"
        assert np.load(path).shape == (expected,), filename


def test_chr123_array_total_train_heldout(chr123_protocol):
    assert len(chr123_protocol.array_train_cpg_idx) == 78211
    assert len(chr123_protocol.array_val_cpg_idx) == 14893
    assert len(chr123_protocol.array_train_cpg_idx) + len(chr123_protocol.array_val_cpg_idx) == 93104
    assert set(chr123_protocol.array_train_cpg_idx.tolist()).isdisjoint(
        chr123_protocol.array_val_cpg_idx.tolist()
    )


def test_chr123_sample_split_matches_chr1(chr1_protocol, chr123_protocol):
    """Sample-axis (patient) split is genome-wide, not chromosome-specific."""
    assert np.array_equal(
        np.sort(chr1_protocol.array_train_sample_idx), np.sort(chr123_protocol.array_train_sample_idx)
    )
    assert np.array_equal(
        np.sort(chr1_protocol.array_val_sample_idx), np.sort(chr123_protocol.array_val_sample_idx)
    )


def test_auxiliary_cpg_pools_present_for_mix_protocols(chr1_protocol, chr123_protocol):
    assert len(chr1_protocol.auxiliary_cpg_idx["epic"]) == 71748
    assert len(chr1_protocol.auxiliary_cpg_idx["wgbs"]) == 1999548
    assert len(chr123_protocol.auxiliary_cpg_idx["epic"]) == 172723
    assert len(chr123_protocol.auxiliary_cpg_idx["wgbs"]) == 5396437


def test_array_only_protocol_has_no_auxiliary_sources(bundle):
    protocol = load_protocol("tcga_array_chr1", bundle=bundle)
    assert protocol.sources == ("array",)
    assert protocol.auxiliary_cpg_idx == {}


@pytest.mark.parametrize("name", ["tcga_array_epic_chr1", "tcga_array_wgbs_chr1"])
def test_two_source_protocols_reuse_the_exact_chr1_array_split(bundle, chr1_protocol, name):
    protocol = load_protocol(name, bundle=bundle)
    assert np.array_equal(
        np.sort(protocol.array_train_sample_idx), np.sort(chr1_protocol.array_train_sample_idx)
    )
    assert np.array_equal(protocol.array_train_cpg_idx, chr1_protocol.array_train_cpg_idx)
