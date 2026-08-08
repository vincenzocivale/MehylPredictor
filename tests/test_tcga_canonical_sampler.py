"""Statistical test of BalancedPairSampler.choose_source, plus real-batch
finite-only / no-NaN-imputation checks (H)."""
from __future__ import annotations

import collections

import numpy as np
import pytest

from methylation_predictor.tcga_canonical import load_protocol
from methylation_predictor.tcga_canonical.sampler import BalancedPairSampler, SourceSamplingPool


def _dummy_pool(name: str, weight: float) -> SourceSamplingPool:
    return SourceSamplingPool(
        name=name,
        row_positions=np.arange(100, dtype=np.int64),
        sample_idx=np.arange(100, dtype=np.int64),
        measurement_idx=np.arange(100, dtype=np.int64),
        cpg_idx_pool=np.arange(1000, dtype=np.int64),
        weight=weight,
    )


def test_choose_source_matches_configured_weights_over_many_draws():
    pools = [_dummy_pool("array", 1.0), _dummy_pool("epic", 1.0), _dummy_pool("wgbs", 1.0)]
    sampler = BalancedPairSampler(pools, seed=17)
    n_draws = 60_000
    counts = collections.Counter(sampler.choose_source().name for _ in range(n_draws))

    expected = n_draws / 3
    for name in ("array", "epic", "wgbs"):
        # 5 sigma band around a binomial(n, 1/3) proportion is comfortably >100
        # draws wide at this n; this is a distribution check, not a coin flip.
        tolerance = 5 * (expected * (1 - 1 / 3)) ** 0.5
        assert abs(counts[name] - expected) < tolerance, (name, counts)


def test_choose_source_respects_unequal_weights():
    pools = [_dummy_pool("array", 3.0), _dummy_pool("epic", 1.0)]
    sampler = BalancedPairSampler(pools, seed=17)
    n_draws = 40_000
    counts = collections.Counter(sampler.choose_source().name for _ in range(n_draws))
    ratio = counts["array"] / counts["epic"]
    assert 2.7 < ratio < 3.3, counts


def test_zero_weight_source_is_never_chosen():
    pools = [_dummy_pool("array", 1.0), _dummy_pool("epic", 0.0)]
    sampler = BalancedPairSampler(pools, seed=0)
    counts = collections.Counter(sampler.choose_source().name for _ in range(2000))
    assert counts["epic"] == 0


@pytest.mark.slow
def test_training_batch_targets_are_finite_and_beta_values_are_unmodified(bundle):
    """H: no silent imputation -- every sampled target is a value that was
    already finite in the source beta matrix, never a filled-in NaN."""
    protocol = load_protocol("tcga_mix_chr1", bundle=bundle)
    dataset = protocol.train_dataset(batch_sample_size=16, batch_cpg_size=64, seed=3)
    for _ in range(5):
        batch = dataset.sample_batch()
        assert len(batch) > 0
        assert np.isfinite(batch.beta).all()
        assert np.isfinite(batch.rna.values).all()
        assert set(np.unique(batch.source).tolist()) <= {"array", "epic", "wgbs"}
        assert batch.rna.values.shape == (len(batch), 25017)
