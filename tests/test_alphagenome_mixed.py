from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT.parent / 'chr1_alphagenome_v1_2026-10-17/pilot.py').exists():
    pytest.skip('Mixed-source research runner requires frozen laboratory helpers', allow_module_level=True)
sys.path.insert(0, str(ROOT / 'scripts'))
from train_alphagenome_mixed import column_subset, positions
from compare_alphagenome_methylprophet import paired_crossed,paper_correlations
from validate_alphagenome_hypotheses import evaluate
from methylation_predictor.storage import SortedIndex


def test_scattered_source_columns_preserve_order_missing_values_and_duplicates(tmp_path):
    rng = np.random.default_rng(9)
    y = rng.random((4, 20001), dtype=np.float32)
    y[2, 18000] = np.nan
    columns = np.array([18000, 3, 8193, 18000, 20000])
    with h5py.File(tmp_path/'source.h5', 'w') as f:
        ds = f.create_dataset('beta', data=y, chunks=(4, 1024))
        np.testing.assert_allclose(column_subset(ds, columns), y[:, columns], equal_nan=True)


def test_paired_crossed_bootstrap_uses_identical_pair_weights():
    n = np.array([[10., 0., 20.], [15., 10., 5.], [20., 10., 7.]])
    result = paired_crossed(.1*n, .2*n, n, draws=100)
    assert result['mse_delta'] == pytest.approx(-.1)
    assert result['mse_delta_percent'] == pytest.approx(-50.)
    assert result['ci95_percent_low'] == pytest.approx(-50.)
    assert result['ci95_percent_high'] == pytest.approx(-50.)


def test_cached_lookup_is_exactly_equivalent_for_unsorted_global_ids():
    rng = np.random.default_rng(73)
    ids = rng.permutation(10000)*13+7
    query = rng.choice(ids, size=800, replace=True)
    cached = SortedIndex(ids)
    np.testing.assert_array_equal(cached.positions_of(query), positions(ids, query))
    with pytest.raises(KeyError):
        cached.positions_of(np.array([-1]))


def test_original_paper_correlations_match_pandas_on_both_axes_with_nan():
    y=torch.tensor([[.1,.8,.4,.5],[.3,.6,.2,float('nan')],[.7,.2,.9,.8],[.9,.3,.6,.7]])
    p=torch.tensor([[.2,.7,.3,.8],[.4,.5,.1,.9],[.6,.4,.8,.6],[.8,.1,.7,.5]])
    _,per=evaluate(y,p,np.array(['NA']*4))
    actual=paper_correlations(y,p,per)
    a,b=pd.DataFrame(y.numpy().astype(np.float64)),pd.DataFrame(p.numpy().astype(np.float64))
    cpg=a.corrwith(b);sample=a.corrwith(b,axis=1)
    assert actual['mas_pcc']==pytest.approx(float(cpg.median()),abs=1e-12)
    assert actual['cpg_pcc_mean']==pytest.approx(float(cpg.mean()),abs=1e-12)
    assert actual['sample_pcc_median']==pytest.approx(float(sample.median()),abs=1e-12)
    assert actual['sample_pcc_mean']==pytest.approx(float(sample.mean()),abs=1e-12)
