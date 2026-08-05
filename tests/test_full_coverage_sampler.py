from __future__ import annotations

import numpy as np

from methylation_predictor.rna_branch.full_coverage_sampler import build_epoch_schedule


def test_full_coverage_guaranteed_realistic_sizes():
    cpg_pool = np.arange(326_906)
    sample_pool = np.arange(7_304)
    for epoch in (1, 2, 17, 47):
        schedule = build_epoch_schedule(cpg_pool, sample_pool, cpg_block_size=2048,
                                         sample_block_size=64, epoch=epoch, seed=17)
        report = schedule.coverage_report()
        assert report["cpg_coverage_fraction"] == 1.0
        assert report["sample_coverage_fraction"] == 1.0
        assert report["steps_per_epoch"] == max(report["n_cpg_blocks"], report["n_sample_blocks"])


def test_full_coverage_with_remainders_small_pools():
    cpg_pool = np.arange(101)
    sample_pool = np.arange(13)
    schedule = build_epoch_schedule(cpg_pool, sample_pool, cpg_block_size=8,
                                     sample_block_size=5, epoch=3, seed=17)
    report = schedule.coverage_report()
    assert report["cpg_coverage_fraction"] == 1.0
    assert report["sample_coverage_fraction"] == 1.0
    assert report["cpg_exposure_min"] >= 1
    assert report["sample_exposure_min"] >= 1

    seen_cpg = set()
    seen_samples = set()
    for step in range(len(schedule)):
        s, c = schedule[step]
        seen_samples.update(s.tolist())
        seen_cpg.update(c.tolist())
    assert seen_cpg == set(cpg_pool.tolist())
    assert seen_samples == set(sample_pool.tolist())


def test_schedule_is_a_pure_function_of_seed_and_epoch():
    cpg_pool = np.arange(500)
    sample_pool = np.arange(50)
    a = build_epoch_schedule(cpg_pool, sample_pool, 32, 8, epoch=5, seed=17)
    b = build_epoch_schedule(cpg_pool, sample_pool, 32, 8, epoch=5, seed=17)
    for step in range(len(a)):
        sa, ca = a[step]
        sb, cb = b[step]
        assert np.array_equal(sa, sb)
        assert np.array_equal(ca, cb)


def test_different_epochs_rotate_pairing():
    cpg_pool = np.arange(500)
    sample_pool = np.arange(50)
    a = build_epoch_schedule(cpg_pool, sample_pool, 32, 8, epoch=1, seed=17)
    b = build_epoch_schedule(cpg_pool, sample_pool, 32, 8, epoch=2, seed=17)
    assert a.rotation_offset != b.rotation_offset
