"""Check scientific invariants of the isolated capacity experiment."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch


@pytest.fixture(scope='module')
def pilot():
    path=Path(__file__).resolve().parents[1]/'scripts/verify_alphagenome_capacity.py'
    if not (path.parents[1].parent/'chr1_alphagenome_v1_2026-10-17/pilot.py').exists():
        pytest.skip('External frozen AlphaGenome laboratory is not available')
    spec=importlib.util.spec_from_file_location('capacity_pilot',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hdf5_selection_preserves_both_unsorted_axes(pilot,tmp_path):
    source=np.arange(7*4101,dtype=np.float32).reshape(7,4101)
    with h5py.File(tmp_path/'test.h5','w') as h:
        d=h.create_dataset('y',data=source)
        s=np.array([6,1,3]); c=np.array([4099,1,2050,0,2047])
        assert np.array_equal(pilot.selected_matrix(d,s,c),source[s][:,c])


def test_embargo_excludes_entire_block_and_both_flanks(pilot):
    pos=np.array([4699999,4700000,4999999,5000000,9999999,10299999,10300000])
    d=SimpleNamespace(cpg_pos_bp=torch.tensor(pos))
    assert np.array_equal(pilot.embargo(d,np.arange(len(pos)),[1]),[0,6])


def functional_inputs():
    return {'track_indices':torch.tensor([1,3,4,7,9]),
            'offsets':torch.tensor([0,2,3,5]),
            'dense':torch.randn(3,23+1536)}


def test_common_initialization_and_depth8_prediction_parity(pilot):
    torch.manual_seed(17); small=pilot.CapacityModel(7,'d0').eval()
    torch.manual_seed(17); deep=pilot.CapacityModel(7,'d8').eval()
    torch.manual_seed(17); reference=pilot.Candidate(7,'hybrid',False).eval()
    for key,value in small.state_dict().items():
        assert torch.equal(value,deep.state_dict()[key]),key
    x=torch.randn(4,7); func=functional_inputs()
    with torch.no_grad():
        assert torch.allclose(deep(x,**func)['beta'],reference(x,**func)['beta'],atol=1e-7)


@pytest.mark.parametrize('arm',['d0','d2','d8','shrink','gated','prototype'])
def test_cached_factorization_matches_forward_and_has_finite_gradients(pilot,arm):
    model=pilot.CapacityModel(7,arm).eval()
    x=torch.randn(4,7); func=functional_inputs()
    result=model(x,**func)
    b,a,_,_,_=model.encode_cpg(**func)
    u=model.rna_factor(model.rna_encoder.shared(x))
    expected=(b[None]+u@a.T/model.rank**.5).sigmoid()
    assert torch.allclose(result['beta'],expected,atol=1e-7)
    (result['beta'].square().mean()+result['penalty']).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_error_decomposition_and_masked_tissue_correlation(pilot):
    rng=np.random.default_rng(4)
    y=rng.random((45,3)).astype(np.float32); y[0,0]=np.nan
    p=np.nan_to_num(y)*.5+.1
    summary,per=pilot.score(y,p,np.array(['A']*22+['B']*23))
    assert np.allclose(per['mse'],per['bias2']+per['centered_mse'])
    assert np.allclose(per['within_corr'],1)
    assert summary['within_tissue']==pytest.approx(1)
