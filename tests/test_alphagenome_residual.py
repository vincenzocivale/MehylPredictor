import io
import pytest
import torch

from methylation_predictor.modeling.factory import build_alphagenome_residual_head


@pytest.mark.parametrize('mode',['ag','sequence','hybrid'])
def test_residual_starts_as_exact_identity_and_roundtrips(mode):
    model=build_alphagenome_residual_head(mode=mode,rank=3,ag_dim=7).eval()
    ag=torch.randn(2,7);seq=torch.randint(0,5,(2,34),dtype=torch.uint8)
    b,a=model(ag,seq)
    assert torch.equal(b,torch.zeros_like(b))
    assert torch.equal(a,torch.zeros_like(a))
    with torch.no_grad():model.output.weight.normal_(std=.1)
    stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
    loaded=build_alphagenome_residual_head(**model.model_spec).eval()
    loaded.load_state_dict(torch.load(stream,weights_only=True))
    for actual,expected in zip(loaded(ag,seq),model(ag,seq)):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_sequence_correction_is_reverse_complement_invariant_and_learns():
    torch.manual_seed(18)
    model=build_alphagenome_residual_head(mode='hybrid',rank=3,ag_dim=7)
    with torch.no_grad():model.output.weight.normal_(std=.1)
    ag=torch.randn(2,7);seq=torch.randint(0,5,(2,34))
    rc=torch.tensor([3,2,1,0,4])[seq.flip(1)]
    first=model(ag,seq);second=model(ag,rc)
    for a,b in zip(first,second):torch.testing.assert_close(a,b)
    sum(v.square().sum() for v in first).backward()
    assert model.sequence[0].weight.grad.abs().sum()>0
    assert model.ag[1].weight.grad.abs().sum()>0


def test_sequence_head_rejects_missing_sequence():
    model=build_alphagenome_residual_head(mode='sequence',ag_dim=7)
    with pytest.raises(ValueError,match='sequence'):model(torch.randn(2,7))

