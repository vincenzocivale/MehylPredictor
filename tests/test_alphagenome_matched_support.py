from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT.parent / 'chr1_alphagenome_v1_2026-10-17/pilot.py').exists():
    pytest.skip('Research helpers require the frozen laboratory', allow_module_level=True)
sys.path.insert(0, str(ROOT / 'scripts'))
from validate_alphagenome_matched_support import (
    patient_split, shrink_mean, residual_prediction, tissue_prediction,
)
from validate_alphagenome_support_uncertainty import crossed_bootstrap


def test_support_and_query_are_disjoint_unique_patients_and_nested():
    ids = np.arange(31) + 100
    support, query = patient_split(ids, 10, 2)
    assert len(query) == 21
    assert not np.intersect1d(ids[support], ids[query]).size
    np.testing.assert_array_equal(np.sort(np.r_[support, query]), np.arange(31))
    np.testing.assert_array_equal(support, patient_split(ids, 10, 2)[0])
    with pytest.raises(ValueError):
        patient_split(np.array([1, 1, 2]), 1, 0)


def test_mean_controls_use_finite_support_count_and_keep_prior_without_labels():
    prior = torch.tensor([.2, .7, .3])
    y = torch.tensor([[.8, float('nan'), .6], [.6, float('nan'), float('nan')]])
    torch.testing.assert_close(shrink_mean(y, prior, 0), torch.tensor([.7, .7, .6]))
    torch.testing.assert_close(shrink_mean(y, prior, 2), torch.tensor([.45, .7, .4]))
    torch.testing.assert_close(shrink_mean(y[:0], prior, 0), prior)


def test_residual_control_preserves_response_and_clips_beta():
    sy = torch.tensor([[.8, float('nan')], [.6, float('nan')]])
    sp = torch.tensor([[.4, .5], [.4, .5]])
    qp = torch.tensor([[.2, .1], [.9, .7]])
    expected = torch.tensor([[.5, .1], [1., .7]])
    torch.testing.assert_close(residual_prediction(sy, sp, qp, 0), expected)
    torch.testing.assert_close(residual_prediction(sy[:0], sp[:0], qp, 5), qp)


def test_tissue_control_uses_global_fallback_for_unknown_or_absent_tissue():
    sy = torch.tensor([[.1], [.9]])
    p = tissue_prediction(sy, torch.tensor([.3]), np.array(['A', 'B']),
                          np.array(['A', 'B', 'C', 'NA']), 0)
    torch.testing.assert_close(p[:, 0], torch.tensor([.1, .9, .5, .5]))


def test_crossed_patient_block_bootstrap_is_paired_and_reproducible():
    counts = np.array([[10., 20.], [15., 10.], [12., 8.]])
    baseline = .2 * counts
    candidate = .1 * counts
    result = crossed_bootstrap(candidate, baseline, counts, draws=100)
    assert result['mse_delta'] == pytest.approx(-.1)
    assert result['mse_delta_percent'] == pytest.approx(-50.)
    assert result['ci95_crossed_low'] == pytest.approx(-.1)
    assert result['ci95_crossed_high'] == pytest.approx(-.1)
    assert result == crossed_bootstrap(candidate, baseline, counts, draws=100)
