import copy

import pytest
import torch

from methylation_predictor.modeling.alphagenome import AlphaGenomeLocalPredictor
from methylation_predictor.modeling.alphagenome_deep import AlphaGenomeDeepDecoderPredictor, VARIANT
from methylation_predictor.modeling.factory import build_alphagenome_predictor

SPEC = dict(width=32, rank=8, n_tracks=50, ag_dim=16, depth=2, reference_count=4)
DECODER = dict(width=32, n_rounds=2, ffn_per_round=1, n_head_ffn=1)
N_LOCI, N_PATIENTS = 7, 5


def _inputs():
    g = torch.Generator().manual_seed(1)
    return dict(
        track_indices=torch.randint(0, 50, (2 * N_LOCI,), generator=g),
        offsets=torch.arange(0, 2 * N_LOCI + 1, 2),
        dense=torch.randn(N_LOCI, 23 + 16, generator=g),
    ), torch.randn(N_PATIENTS, 100, generator=g)


def _randomise_output(model):
    for p in model.decoder.head_out[-1].parameters():
        torch.nn.init.normal_(p, std=0.5)


def _deep(**kw):
    torch.manual_seed(0)
    return AlphaGenomeDeepDecoderPredictor(100, decoder=DECODER, **SPEC, **kw)


def test_starts_exactly_at_the_factorised_model():
    torch.manual_seed(0)
    base = AlphaGenomeLocalPredictor(100, **SPEC).eval()
    deep = _deep().eval()
    cpg, rna = _inputs()
    torch.testing.assert_close(deep(rna, **cpg)["beta"], base(rna, **cpg)["beta"])


def test_residual_index_touches_only_selected_cpgs_and_chunking_is_exact():
    cpg, rna = _inputs()
    model = _deep().eval()
    _randomise_output(model)
    base = AlphaGenomeLocalPredictor(100, **SPEC).eval()
    base.load_state_dict(model.state_dict(), strict=False)
    full = model(rna, **cpg)["beta"]
    idx = torch.tensor([1, 4, 5])
    part = model(rna, **cpg, residual_index=idx)["beta"]
    torch.testing.assert_close(part[:, idx], full[:, idx])
    rest = torch.tensor([0, 2, 3, 6])
    torch.testing.assert_close(part[:, rest], base(rna, **cpg)["beta"][:, rest])
    assert not torch.allclose(full[:, idx], base(rna, **cpg)["beta"][:, idx])
    model.patient_chunk, model.cpg_chunk = 2, 3
    torch.testing.assert_close(model(rna, **cpg)["beta"], full, atol=1e-5, rtol=0)


def test_training_gradients_reach_the_decoder_through_recomputation():
    cpg, rna = _inputs()
    model = _deep(cpg_chunk=3).train()
    out = model(rna, **cpg, return_internals=True, residual_index=torch.tensor([0, 2, 5]))
    out["beta"].sum().backward()
    assert out["beta"].shape == (N_PATIENTS, N_LOCI) and out["cpg_factors"].shape == (N_LOCI, 8)
    assert model.decoder.head_out[-1].weight.grad.abs().sum() > 0
    assert model.token_proj.weight.grad is not None


def test_factory_roundtrip_through_model_spec():
    model = _deep().eval()
    _randomise_output(model)
    spec = copy.deepcopy(model.model_spec)
    assert spec["variant"] == VARIANT
    clone = build_alphagenome_predictor(**spec).eval()
    clone.load_state_dict(model.state_dict(), strict=True)
    cpg, rna = _inputs()
    torch.testing.assert_close(clone(rna, **cpg)["beta"], model(rna, **cpg)["beta"])
    with pytest.raises(ValueError):
        build_alphagenome_predictor(variant="nope")
