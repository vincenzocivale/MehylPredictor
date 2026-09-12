from __future__ import annotations

import torch

from methylation_predictor.config import EncoderConfig, ModelConfig
from methylation_predictor.modeling import EfficientSingleAttentionPredictor
from methylation_predictor.modeling.retrieval import BatchedLocusToRNAAttention


def _config(n_programs=6):
    return ModelConfig(
        encoder=EncoderConfig(kind="locus_attention", latent_dim=16, n_programs=n_programs,
                               program_dim=256, n_heads=4, dropout=0.0),
    )


def _model(n_ffn_blocks=4, input_dim=48, n_programs=6, dropout=0.15):
    torch.manual_seed(17)
    return EfficientSingleAttentionPredictor(
        input_dim, _config(n_programs), n_ffn_blocks=n_ffn_blocks,
        final_regressor_dropout=dropout,
    )


def _functional_inputs():
    return {
        "functional_track_indices": torch.tensor([1, 2, 5, 9], dtype=torch.int64),
        "functional_offsets": torch.tensor([0, 2, 2, 3, 4], dtype=torch.int64),
        "functional_dense": torch.randn(4, 23),
    }


def test_attention_runs_exactly_once():
    model = _model(n_ffn_blocks=4)
    assert isinstance(model.retrieval_attn, BatchedLocusToRNAAttention)
    assert len(model.retrieval_ffn) == 4


def test_forward_shapes():
    model = _model(n_ffn_blocks=3).eval()
    out = model(torch.randn(3, 48), **_functional_inputs())
    assert out["beta"].shape == (3, 4)
    assert torch.all((out["beta"] >= 0) & (out["beta"] <= 1))
    assert out["mu_hat"].shape == (4,)


def test_ffn_stack_keeps_locus_variation():
    # Single attention + residual add, then FFN-only refinement -- state
    # should still vary across loci (from h_c via the residual), the same
    # property ablation_depth1_residual relies on.
    model = _model(n_ffn_blocks=4).eval()
    functional = _functional_inputs()
    tokens = torch.zeros(3, 6, 256)
    peak = model.track_embedding(functional["functional_track_indices"], functional["functional_offsets"])
    dense = model.dense_encoder(functional["functional_dense"])
    h_c = model.locus_norm(peak + dense)
    state = h_c[None, :, :].expand(3, -1, -1)
    state = state + model.retrieval_attn(state, tokens)
    for ffn in model.retrieval_ffn:
        state = ffn(state)
    assert not torch.allclose(state[:, 0], state[:, 1], atol=1e-4)


def test_mean_head_gradient_isolation():
    model = _model(n_ffn_blocks=4).train()
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
    model = _model(n_ffn_blocks=4).train()
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
                   model.final_regressor, model.mean_head, model.retrieval_attn):
        assert has_grad(module)
    for ffn in model.retrieval_ffn:
        assert has_grad(ffn)


def test_locus_chunking_matches_unchunked():
    torch.manual_seed(0)
    model = _model(n_ffn_blocks=4, input_dim=48, n_programs=6).eval()
    functional = _functional_inputs()
    rna = torch.randn(3, 48)
    model.LOCUS_CHUNK = 2
    chunked = model(rna, **functional)["beta"]
    model.LOCUS_CHUNK = 10_000
    unchunked = model(rna, **functional)["beta"]
    torch.testing.assert_close(chunked, unchunked)


def test_rejects_invalid_n_ffn_blocks():
    import pytest
    with pytest.raises(ValueError):
        EfficientSingleAttentionPredictor(48, _config(), n_ffn_blocks=0)
