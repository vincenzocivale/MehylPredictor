import pytest

from methylation_predictor.locus_features.ids import (
    canonical_locus,
    decode_locus_key,
    locus_key,
    tcga_cpg_idx_to_locus_key,
)


def test_stable_round_trip_and_aliases():
    key = locus_key("chr1", 100034965)
    assert key == (1 << 32) | 100034965
    assert decode_locus_key(key) == ("chr1", 100034965)
    assert tcga_cpg_idx_to_locus_key(8942983, {8942983: ("chr1", 100034965)}) == key


@pytest.mark.parametrize("chrom,pos", [("1", 12), ("chrM", 12), ("chr1", 0), ("chr1", -1), ("chr1", 2**32)])
def test_invalid_coordinate(chrom, pos):
    with pytest.raises(ValueError):
        locus_key(chrom, pos)


def test_bounds_and_missing_alias():
    with pytest.raises(ValueError):
        canonical_locus("chr1", 100, lengths={"chr1": 100})
    with pytest.raises(KeyError):
        tcga_cpg_idx_to_locus_key(1, {})
