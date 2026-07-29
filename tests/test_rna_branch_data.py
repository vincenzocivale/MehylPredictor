from __future__ import annotations

import numpy as np

from methylation_predictor.rna_branch.config import load_config
from methylation_predictor.rna_branch.data import load_bundle


def test_alignment_and_standardization(synthetic_project):
    config = load_config(synthetic_project)
    bundle = load_bundle(config.data, seed=config.training.seed)
    try:
        assert len(bundle.samples.ids) == 24
        assert len(bundle.loci.ids) == 12
        train = bundle.sample_indices("train")
        values = bundle.rna(train)
        assert np.allclose(values.mean(axis=0), 0.0, atol=1e-5)
        assert bundle.beta(train[:2], bundle.cpg_indices("train")[:3]).shape == (2, 3)
    finally:
        bundle.close()


def test_within_cancer_shuffle_preserves_split_and_cancer(synthetic_project):
    config = load_config(synthetic_project)
    config.data.rna_control = "shuffle_within_cancer"
    bundle = load_bundle(config.data, seed=config.training.seed)
    try:
        mapping = bundle.control_row_map
        assert np.all(bundle.samples.splits == bundle.samples.splits[mapping])
        assert np.all(bundle.samples.cancer_types == bundle.samples.cancer_types[mapping])
    finally:
        bundle.close()
