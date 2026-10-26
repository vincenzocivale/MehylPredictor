from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch

if not (Path(__file__).resolve().parents[2]/'chr1_alphagenome_v1_2026-10-17/pilot.py').exists():
    pytest.skip('External frozen AlphaGenome laboratory is not available',allow_module_level=True)

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from search_alphagenome_hierarchical import fit_known_offsets
from search_alphagenome_architecture import neighbor_table


def test_empirical_intercept_uses_only_observed_training_pairs():
    torch.manual_seed(2)
    u=torch.randn(200,4); a=torch.randn(3,4); b=torch.tensor([-1.,0.,1.])
    y=(b[None]+torch.tensor([.7,-.4,.9])[None]+u@a.T/2).sigmoid()
    y[:40,1]=torch.nan
    fitted=fit_known_offsets(b,a,u,y,[0.,.5,1.])
    mask=torch.isfinite(y)
    original=(b[None]+u@a.T/2).sigmoid()
    old=torch.where(mask,original,0).sum(0)/mask.sum(0)
    target=torch.nanmean(y,0)
    assert torch.equal(fitted[0.],torch.zeros_like(b))
    for w in [.5,1.]:
        prediction=(b[None]+fitted[w][None]+u@a.T/2).sigmoid()
        observed=torch.where(mask,prediction,0).sum(0)/mask.sum(0)
        assert torch.allclose(observed,(1-w)*old+w*target,atol=2e-6)


def test_neighbor_preprocessing_excludes_query_loci_and_targets(tmp_path):
    rng=np.random.default_rng(7)
    dense=np.zeros((8,23),dtype=np.float32)
    d=SimpleNamespace(device=torch.device('cpu'),cpg_idx=np.arange(100,108),
                      cpg_pos_bp=torch.arange(8)*2_000_000,raw_dense=dense)
    x=rng.normal(size=(8,8)).astype(np.float32); train=np.arange(6)
    first=neighbor_table(d,train,x,tmp_path/'first.npz')
    x[6]=1000
    second=neighbor_table(d,train,x,tmp_path/'second.npz')
    assert torch.equal(first['nearest'][:6],second['nearest'][:6])
    assert torch.equal(first['nearest'][7],second['nearest'][7])
    assert torch.equal(first['random'],second['random'])
    assert np.isin(first['nearest'].numpy(),train).all()
    assert ((d.cpg_pos_bp[:,None]-d.cpg_pos_bp[first['nearest']]).abs()>=1_000_000).all()
