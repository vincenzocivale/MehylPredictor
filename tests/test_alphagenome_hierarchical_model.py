from __future__ import annotations
import pytest
import torch
from methylation_predictor.modeling.alphagenome_hierarchical import AlphaGenomeHierarchicalPredictor


def model():
    m=AlphaGenomeHierarchicalPredictor(7,n_known=2,width=16,rank=4,n_tracks=12,ag_dim=8,
                                      depth=2,dropout=0.,mean_blend=.5,reference_count=20).eval()
    m.reference_u.copy_(torch.randn_like(m.reference_u))
    m.prior_intercept.fill_(.3); m.calibration_ready.fill_(True)
    return m


def inputs():
    return dict(track_indices=torch.tensor([1,3,4,7]),offsets=torch.tensor([0,2,3,4]),
                dense=torch.randn(3,31),cpg_ids=torch.tensor([101,203,999]))


def test_known_offsets_are_id_aligned_and_leave_unknown_predictions_unchanged():
    m=model(); f=inputs(); raw,a=m.encode_cpg(**f,calibrate=False)
    expected_novel=m.calibrated_intercept(raw,a,f['dense'][:,23:])
    m.set_known_offsets(torch.tensor([203,101]),torch.tensor([-.4,.7]))
    b,aa=m.encode_cpg(**f)
    assert torch.equal(a,aa)
    assert torch.allclose(b[:2],raw[:2]+torch.tensor([.7,-.4]))
    assert torch.equal(b[2:],expected_novel[2:])
    one=dict(track_indices=f['track_indices'][:2],offsets=torch.tensor([0,2]),
             dense=f['dense'][:1],cpg_ids=f['cpg_ids'][:1])
    b1,a1=m.encode_cpg(**one)
    assert torch.allclose(b1,b[:1],atol=1e-6)
    assert torch.allclose(a1,aa[:1],atol=1e-6)


def test_hierarchical_fit_and_inference_validate_registry_identifiers():
    m=model(); f=inputs()
    with pytest.raises(RuntimeError,match='Fit'):
        m.encode_cpg(**f)
    with pytest.raises(ValueError,match='unique'):
        m.set_known_offsets(torch.tensor([101,101]),torch.zeros(2))
    with pytest.raises(ValueError,match='integer'):
        m.set_known_offsets(torch.tensor([101.,203.]),torch.zeros(2))
    m.set_known_offsets(torch.tensor([101,203]),torch.zeros(2))
    f.pop('cpg_ids')
    with pytest.raises(ValueError,match='Global registry'):
        m.encode_cpg(**f)
    with pytest.raises(ValueError,match='integer'):
        m.encode_cpg(**f,cpg_ids=torch.tensor([101.,203.,999.]))


@pytest.mark.parametrize('frozen_mean',[False,True])
def test_hierarchical_checkpoint_preserves_known_and_novel_predictions(tmp_path,frozen_mean):
    from methylation_predictor.modeling.alphagenome import AlphaGenomePredictorFromCheckpoint
    m=model()
    if frozen_mean: m.freeze_mean()
    m.set_known_offsets(torch.tensor([203,101]),torch.tensor([-.4,.7]))
    f=inputs(); rna=torch.randn(5,7); genes=[f'g{i}' for i in range(7)]
    path=tmp_path/'model.pt'
    norm=dict(rna_mean=torch.zeros(7),rna_std=torch.ones(7),dense17_mean=0.,dense17_std=1.)
    torch.save(dict(schema='alphagenome_hierarchical_chr1_v3',model_spec=m.model_spec,
                    model_state=m.state_dict(),normalization=norm,gene_ids=genes),path)
    adapter=AlphaGenomePredictorFromCheckpoint(path)
    expected=m(rna,**f)['beta']
    actual=adapter(rna,genes,f['track_indices'],f['offsets'],f['dense'][:,:23],f['dense'][:,23:],cpg_ids=f['cpg_ids'])
    assert torch.equal(expected,actual)
    with pytest.raises(ValueError,match='Global registry'):
        adapter(rna,genes,f['track_indices'],f['offsets'],f['dense'][:,:23],f['dense'][:,23:])
    with pytest.raises(ValueError,match='integer'):
        adapter(rna,genes,f['track_indices'],f['offsets'],f['dense'][:,:23],f['dense'][:,23:],cpg_ids=f['cpg_ids'].float())
