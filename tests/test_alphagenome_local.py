from __future__ import annotations

import numpy as np
from pathlib import Path
import sys
import pytest
import torch

from methylation_predictor.modeling.alphagenome import (
    AlphaGenomeLocalPredictor, AlphaGenomePredictorFromCheckpoint, local_response_penalty,
)
from methylation_predictor.modeling.factory import build_alphagenome_predictor


def model(blend=0., shallow=True):
    return AlphaGenomeLocalPredictor(7,width=16,rank=4,n_tracks=12,ag_dim=8,depth=2,
                                     dropout=0.,shallow_mean=shallow,mean_blend=blend,reference_count=20)


def inputs():
    return dict(track_indices=torch.tensor([1,3,4,7]),offsets=torch.tensor([0,2,3,4]),
                dense=torch.randn(3,31))


def ready(m):
    with torch.no_grad():
        m.reference_u.copy_(torch.randn_like(m.reference_u))
        m.prior_intercept.fill_(.3)
        m.calibration_ready.fill_(True)


@pytest.mark.parametrize('blend',[.5,1.])
def test_calibration_matches_population_mean_and_retains_response(blend):
    torch.manual_seed(9)
    m=model(blend).eval(); ready(m); f=inputs()
    b,a=m.encode_cpg(**f,calibrate=False)
    raw=(b[None]+m.reference_u@a.T/2).sigmoid()
    calibrated,_=m.encode_cpg(**f)
    result=(calibrated[None]+m.reference_u@a.T/2).sigmoid()
    assert torch.allclose(result.mean(0),(1-blend)*raw.mean(0)+blend*.3,atol=1e-6)
    assert result.std(0).min()>0


def test_calibrated_prediction_is_independent_of_patient_and_cpg_batching():
    m=model(.5).eval(); ready(m); f=inputs(); rna=torch.randn(5,7)
    expected=m(rna,**f)['beta']
    assert torch.allclose(expected[:1],m(rna[:1],**f)['beta'],atol=1e-6)
    one=dict(track_indices=f['track_indices'][:2],offsets=torch.tensor([0,2]),dense=f['dense'][:1])
    assert torch.allclose(expected[:,:1],m(rna,**one)['beta'],atol=1e-6)


def test_shallow_mean_gradient_is_separate_from_deep_response():
    m=model().train(); result=m(torch.randn(5,7),**inputs())
    result['mu_hat'].sum().backward()
    assert m.mean_head[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in m.functional_ffn.parameters())
    assert all(p.grad is None for p in m.rna_encoder.parameters())


def test_graph_penalty_is_centered_and_invariant_to_factor_basis():
    torch.manual_seed(3)
    u=torch.randn(10,4,requires_grad=True); a=torch.randn(3,4,requires_grad=True); b=torch.randn(3,4)
    penalty=local_response_penalty(u,a,b)
    change=torch.randn(4,4)+4*torch.eye(4); inv=torch.linalg.inv(change).T
    equivalent=local_response_penalty(u@change,a@inv,b@inv)
    assert torch.allclose(penalty,equivalent,atol=1e-5)
    assert torch.allclose(penalty,local_response_penalty(u+5,a,b),atol=1e-5)
    assert local_response_penalty(torch.ones_like(u),a,b)==0
    penalty.backward()
    assert torch.isfinite(u.grad).all() and torch.isfinite(a.grad).all()


def test_calibration_rejects_missing_fit_and_factory_rejects_unknown_variant():
    with pytest.raises(RuntimeError,match='fitted prior'):
        model(1.).eval()(torch.randn(2,7),**inputs())
    with pytest.raises(ValueError,match='unsupported'):
        build_alphagenome_predictor(variant='unknown',input_dim=7)


def test_portable_checkpoint_preprocessing_and_gene_order(tmp_path):
    m=model(.5).eval(); ready(m); f=inputs(); rna=torch.randn(5,7)
    genes=[f'g{i}' for i in range(7)]
    norm=dict(rna_mean=torch.ones(7),rna_std=torch.ones(7)*2,dense17_mean=torch.tensor(2.),dense17_std=torch.tensor(3.))
    checkpoint=tmp_path/'model.pt'
    torch.save(dict(schema='alphagenome_local_chr1_v2',model_spec=m.model_spec,
                    model_state=m.state_dict(),normalization=norm,gene_ids=genes),checkpoint)
    adapter=AlphaGenomePredictorFromCheckpoint(checkpoint)
    raw=f['dense'][:,:23].clone(); ag=f['dense'][:,23:]
    processed=raw.clone(); processed[:,17]=(processed[:,17]-2)/3
    expected=m((rna-1)/2,track_indices=f['track_indices'],offsets=f['offsets'],dense=torch.cat([processed,ag],1))['beta']
    predicted=adapter(rna,genes,f['track_indices'],f['offsets'],raw,ag)
    assert torch.allclose(expected,predicted,atol=1e-7)
    with pytest.raises(ValueError,match='order'):
        adapter(rna,genes[::-1],f['track_indices'],f['offsets'],raw,ag)
    with pytest.raises(ValueError,match='offsets'):
        adapter(rna,genes,f['track_indices'],torch.tensor([0,1]),raw,ag)


def test_unmodified_control_matches_frozen_v1_architecture():
    lab=Path(__file__).resolve().parents[2]/'chr1_alphagenome_v1_2026-10-17'
    if not (lab/'pilot.py').exists():
        pytest.skip('Frozen v1 laboratory unavailable')
    sys.path.insert(0,str(lab))
    from pilot import Candidate
    reference=Candidate(7,'hybrid',False).eval()
    candidate=AlphaGenomeLocalPredictor(7,shallow_mean=False).eval()
    missing=candidate.load_state_dict(reference.state_dict(),strict=False)
    assert not missing.unexpected_keys
    f=dict(track_indices=torch.tensor([1,3,4,7]),offsets=torch.tensor([0,2,3,4]),
           dense=torch.randn(3,23+1536))
    rna=torch.randn(5,7)
    with torch.inference_mode():
        assert torch.allclose(candidate(rna,**f)['beta'],reference(rna,**f)['beta'],atol=1e-7)
