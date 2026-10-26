from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
if not (ROOT.parent/'chr1_alphagenome_v1_2026-10-17/pilot.py').exists():
    pytest.skip('Research helpers require the frozen laboratory',allow_module_level=True)
sys.path.insert(0,str(ROOT/'scripts'))
from validate_alphagenome_hypotheses import evaluate,solve_bias,permutations
from validate_alphagenome_transfer import collapse_measurements


def test_metrics_match_finite_pair_reference_and_decompose_error():
    y=torch.tensor([[0.,.3],[.2,float('nan')],[.8,.7],[1.,.9]])
    p=torch.tensor([[.1,.2],[.3,.4],[.5,.6],[.6,.8]])
    metric,per=evaluate(y,p,np.array(['NA']*4))
    mask=torch.isfinite(y)
    assert metric['pairs']==7
    assert metric['mse']==pytest.approx(float((p-y)[mask].square().double().mean()))
    assert metric['mae']==pytest.approx(float((p-y)[mask].abs().double().mean()))
    np.testing.assert_allclose(per['mse'],per['bias2']+per['centered_mse'],atol=1e-12)
    np.testing.assert_array_equal(per['n'],[4,3])


def test_few_shot_missing_support_keeps_prior_and_shrinkage_limits_update():
    b=torch.tensor([0.,.3]);response=torch.zeros(2,2)
    y=torch.tensor([[.9,float('nan')],[.9,float('nan')]])
    direct=solve_bias(b,response,y,0.)
    shrunk=solve_bias(b,response,y,5.)
    assert direct[1]==b[1] and shrunk[1]==b[1]
    assert direct[0].sigmoid()==pytest.approx(.9,abs=1e-5)
    assert shrunk[0].sigmoid()==pytest.approx(.5+2/7*(.9-.5),abs=1e-5)


def test_within_tissue_permutations_preserve_group_and_are_reproducible():
    tissue=np.array(['A','B','A','B','A','C'])
    ix=permutations(tissue,17)
    np.testing.assert_array_equal(tissue[ix],tissue)
    np.testing.assert_array_equal(np.sort(ix),np.arange(len(ix)))
    np.testing.assert_array_equal(ix,permutations(tissue,17))


def test_repeated_assays_have_one_patient_weight_and_ignore_missing_values():
    y=np.array([[.2,np.nan],[.4,.9],[.8,.3]],np.float32)
    actual=collapse_measurements(y,np.array([7,7,9]),np.array([9,7]))
    np.testing.assert_allclose(actual,[[.8,.3],[.3,.9]],rtol=1e-6)
