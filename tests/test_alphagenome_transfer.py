from __future__ import annotations

import pytest
import torch

from methylation_predictor.modeling.factory import build_alphagenome_predictor
from methylation_predictor.modeling.alphagenome import AlphaGenomePredictorFromCheckpoint
from methylation_predictor.modeling.alphagenome_transfer import transfer_response


def model():
    return build_alphagenome_predictor(variant='alphagenome_two_stage_chr1_v3',
        input_dim=7,width=16,rank=4,n_tracks=12,ag_dim=8,depth=2,dropout=.5,reference_count=20)


def inputs():
    return dict(track_indices=torch.tensor([1,3,4,7]),offsets=torch.tensor([0,2,3,4]),dense=torch.randn(3,31))


def test_complete_mean_function_is_frozen_while_responses_can_learn():
    torch.manual_seed(3)
    m=model().eval(); f=inputs()
    before,a0=m.encode_cpg(**f)
    m.freeze_mean(); m.train()
    assert not m.mean_snapshot.training
    b,a=m.encode_cpg(**f)
    assert torch.allclose(b,before,atol=1e-7)
    loss=a.square().sum()
    opt=torch.optim.SGD([p for p in m.parameters() if p.requires_grad],lr=.1)
    opt.zero_grad(); loss.backward(); opt.step()
    b1,a1=m.eval().encode_cpg(**f)
    assert torch.equal(b,b1)
    assert not torch.allclose(a0,a1)
    assert all(p.grad is None for p in m.mean_snapshot.parameters())
    assert m.ag_encoder[1].weight.grad.abs().sum()>0


def test_snapshot_checkpoint_is_portable(tmp_path):
    m=model().eval(); m.freeze_mean(); f=inputs(); rna=torch.randn(5,7)
    genes=[f'g{i}' for i in range(7)]
    path=tmp_path/'checkpoint.pt'
    norm=dict(rna_mean=torch.zeros(7),rna_std=torch.ones(7),dense17_mean=0.,dense17_std=1.)
    torch.save(dict(schema='alphagenome_two_stage_chr1_v3',model_spec=m.model_spec,
                    model_state=m.state_dict(),gene_ids=genes,normalization=norm),path)
    adapter=AlphaGenomePredictorFromCheckpoint(path)
    expected=m(rna,**f)['beta']
    actual=adapter(rna,genes,f['track_indices'],f['offsets'],f['dense'][:,:23],f['dense'][:,23:])
    assert torch.allclose(expected,actual,atol=1e-7)


@pytest.mark.parametrize('blend',[.25,.5,1.])
def test_transfer_preserves_reference_mean_and_is_basis_invariant(blend):
    torch.manual_seed(7)
    b=torch.randn(3); a=torch.randn(3,4); near=torch.randn(3,5,4); u=torch.randn(20,4)
    original=(b[None]+u@a.T/2).sigmoid()
    bias,factors=transfer_response(b,a,near,u,blend)
    prediction=(bias[None]+u@factors.T/2).sigmoid()
    assert torch.allclose(original.mean(0),prediction.mean(0),atol=2e-6)
    change=torch.randn(4,4)+5*torch.eye(4); inv=torch.linalg.inv(change).T
    bb,aa=transfer_response(b,a@inv,near@inv,u@change,blend)
    equivalent=(bb[None]+(u@change)@aa.T/2).sigmoid()
    assert torch.allclose(prediction,equivalent,atol=2e-6)
    # A query locus processed alone has exactly the same reference correction.
    b1,a1=transfer_response(b[:1],a[:1],near[:1],u,blend)
    assert torch.allclose(bias[:1],b1,atol=2e-6)
    assert torch.equal(factors[:1],a1)


def test_transfer_rejects_invalid_inputs_and_zero_blend_is_identity():
    b=torch.zeros(3); a=torch.ones(3,4); n=torch.ones(3,5,4); u=torch.ones(20,4)
    bb,aa=transfer_response(b,a,n,u,0.)
    assert bb is b and aa is a
    with pytest.raises(ValueError,match='blend'):
        transfer_response(b,a,n,u,2.)
    with pytest.raises(ValueError,match='shape'):
        transfer_response(b,a,n[0],u,.5)
    with pytest.raises(ValueError,match='reference'):
        transfer_response(b,a,n,u[:0],.5)
