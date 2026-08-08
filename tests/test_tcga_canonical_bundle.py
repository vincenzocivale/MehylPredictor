"""A: canonical shapes. G: every measurement's sample_idx resolves to real RNA."""
from __future__ import annotations

import numpy as np

EXPECTED_SHAPES = {
    "rna": (10916, 25017),
    "array": (9178, 408399),
    "epic": (1706, 740296),
    "wgbs": (32, 23047052),
}


def test_canonical_shapes(bundle):
    assert bundle.rna.shape == EXPECTED_SHAPES["rna"]
    for name, expected in EXPECTED_SHAPES.items():
        if name == "rna":
            continue
        assert bundle.sources[name].shape == expected, name


def test_rna_uses_all_25017_genes_not_legacy_21792(bundle):
    assert bundle.rna.gene_ids.shape == (25017,)
    assert len(set(bundle.rna.gene_ids.tolist())) == 25017


def test_every_source_measurement_resolves_a_real_rna_row(bundle):
    """G: every measurement's sample_idx must find exactly one RNA row; a
    protocol using an RNA-less measurement must fail loudly, not silently."""
    for name, source in bundle.sources.items():
        sample_idx = np.unique(source.sample_idx)
        assert bool(bundle.rna.has_sample(sample_idx).all()), f"{name}: missing RNA for some sample_idx"
        rows = bundle.rna.rows(sample_idx[:8])
        assert rows.shape == (min(8, len(sample_idx)), 25017)
        assert np.isfinite(rows).all()


def test_rna_missing_sample_raises(bundle):
    bogus_sample_idx = int(bundle.rna.sample_index._sorted_ids.max()) + 999_999
    try:
        bundle.rna.rows([bogus_sample_idx])
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for a sample_idx with no RNA row")


def test_wgbs_32_measurements_31_unique_samples_duplicate_preserved(bundle):
    """F: WGBS duplicate sample_idx is intentional and must not be deduplicated."""
    wgbs = bundle.sources["wgbs"]
    assert wgbs.n_rows == 32
    assert len(np.unique(wgbs.sample_idx)) == 31
    assert len(np.unique(wgbs.measurement_idx)) == 32

    counts = {}
    for value in wgbs.sample_idx.tolist():
        counts[value] = counts.get(value, 0) + 1
    duplicated = [sample_id for sample_id, count in counts.items() if count == 2]
    assert len(duplicated) == 1
    rows = wgbs.rows_of_sample(duplicated[0])
    assert len(rows) == 2, "duplicate WGBS measurement must not be collapsed"
