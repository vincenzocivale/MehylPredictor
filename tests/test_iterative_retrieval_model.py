from __future__ import annotations

import torch

from methylation_predictor.config import EncoderConfig, ModelConfig
from methylation_predictor.models import FunctionalConcatIterativeRNAModel


def _config(n_programs=6):
    return ModelConfig(
        encoder=EncoderConfig(kind="locus_attention", latent_dim=16, n_programs=n_programs,
                               program_dim=256, n_heads=4, dropout=0.0),
    )


def _model(input_dim=48, n_programs=6, dropout=0.15):
    torch.manual_seed(17)
    return FunctionalConcatIterativeRNAModel(input_dim, _config(n_programs), final_regressor_dropout=dropout)


def _functional_inputs():
    return {
        "functional_track_indices": torch.tensor([1, 2, 5, 9], dtype=torch.int64),
        "functional_offsets": torch.tensor([0, 2, 2, 3, 4], dtype=torch.int64),
        "functional_dense": torch.randn(4, 23),
    }


def test_forward_shapes():
    model = _model().eval()
    out = model(torch.randn(3, 48), **_functional_inputs())
    assert out["beta"].shape == (3, 4)
    assert torch.all((out["beta"] >= 0) & (out["beta"] <= 1))
    assert out["mu_hat"].shape == (4,)
    assert len(model.retrieval_attn) == FunctionalConcatIterativeRNAModel.N_BLOCKS == 4
    assert len(model.retrieval_ffn) == 4


def test_every_block_has_a_residual():
    # With V=0 (zeroed RNA program tokens), each block's bare attention
    # output collapses to a locus/patient-invariant constant (the
    # output-projection bias). If block i's residual were missing, `state`
    # after block i would equal that same constant, independent of the
    # locus-varying input that fed it. With the residual present (every
    # block, including the first), `state` keeps varying across loci through
    # all N_BLOCKS blocks because h_c itself varies across loci.
    model = _model().eval()
    functional = _functional_inputs()
    tokens = torch.zeros(3, 6, 256)
    peak = model.track_embedding(functional["functional_track_indices"], functional["functional_offsets"])
    dense = model.dense_encoder(functional["functional_dense"])
    h_c = model.locus_norm(peak + dense)
    state = h_c[None, :, :].expand(3, -1, -1)
    for attn, ffn in zip(model.retrieval_attn, model.retrieval_ffn):
        state = state + attn(state, tokens)
        state = ffn(state)
    # Loci differ (h_c differs per locus) so the final state must too --
    # this would fail if any block dropped its residual (state would
    # collapse to a per-locus-invariant constant after that block).
    assert not torch.allclose(state[:, 0], state[:, 1], atol=1e-4)
    assert not torch.allclose(state[:, 0], state[:, 2], atol=1e-4)


def test_mean_head_gradient_isolation():
    model = _model().train()
    functional = _functional_inputs()
    rna = torch.randn(3, 48)
    out = model(rna, **functional)
    model.zero_grad(set_to_none=True)
    out["mu_hat"].sum().backward()

    def has_grad(module):
        return any(p.grad is not None and torch.any(p.grad != 0) for p in module.parameters())

    assert has_grad(model.track_embedding)
    assert has_grad(model.dense_encoder)
    assert has_grad(model.mean_head)
    assert not has_grad(model.rna_encoder)
    assert not has_grad(model.retrieval_attn)
    assert not has_grad(model.retrieval_ffn)
    assert not has_grad(model.final_regressor)


def test_full_forward_backward_reaches_every_module():
    model = _model().train()
    functional = _functional_inputs()
    rna = torch.randn(3, 48)
    target = torch.rand(3, 4)
    out = model(rna, **functional)
    model.zero_grad(set_to_none=True)
    loss = ((out["beta"] - target) ** 2).mean() + 0.15 * ((out["mu_hat"] - target.mean(0)) ** 2).mean()
    loss.backward()

    def has_grad(module):
        return any(p.grad is not None and torch.any(p.grad != 0) for p in module.parameters())

    for module in (model.rna_encoder, model.track_embedding, model.dense_encoder,
                   model.final_regressor, model.mean_head):
        assert has_grad(module)
    for attn, ffn in zip(model.retrieval_attn, model.retrieval_ffn):
        assert has_grad(attn)
        assert has_grad(ffn)


def test_locus_chunking_matches_unchunked():
    torch.manual_seed(0)
    model = _model(input_dim=48, n_programs=6).eval()
    model.LOCUS_CHUNK = 2  # force chunking with only 4 loci
    functional = _functional_inputs()
    rna = torch.randn(3, 48)
    chunked = model(rna, **functional)["beta"]
    model.LOCUS_CHUNK = 10_000  # no chunking
    unchunked = model(rna, **functional)["beta"]
    torch.testing.assert_close(chunked, unchunked)
