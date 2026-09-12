from __future__ import annotations

import torch

from methylation_predictor.config import EncoderConfig, ModelConfig
from methylation_predictor.modeling import GatedResidualPredictor


def _config(n_programs=6):
    return ModelConfig(
        encoder=EncoderConfig(kind="locus_attention", latent_dim=16, n_programs=n_programs,
                               program_dim=256, n_heads=4, dropout=0.0),
    )


def _model(n_blocks=1, input_dim=48, n_programs=6, dropout=0.15):
    torch.manual_seed(17)
    return GatedResidualPredictor(
        input_dim, _config(n_programs), n_blocks=n_blocks, final_regressor_dropout=dropout,
    )


def _functional_inputs():
    return {
        "functional_track_indices": torch.tensor([1, 2, 5, 9], dtype=torch.int64),
        "functional_offsets": torch.tensor([0, 2, 2, 3, 4], dtype=torch.int64),
        "functional_dense": torch.randn(4, 23),
    }


def test_gate_inits_to_zero():
    model = _model(n_blocks=3)
    assert model.residual_gate.shape == (3,)
    assert torch.allclose(model.residual_gate, torch.zeros(3))
    assert torch.allclose(torch.tanh(model.residual_gate), torch.zeros(3))


def test_forward_shapes():
    model = _model(n_blocks=1).eval()
    out = model(torch.randn(3, 48), **_functional_inputs())
    assert out["beta"].shape == (3, 4)
    assert torch.all((out["beta"] >= 0) & (out["beta"] <= 1))
    assert out["mu_hat"].shape == (4,)


def test_zero_gate_makes_block_ignore_attention_output():
    # At init (gate=0), attn's output is fully suppressed -- state after the
    # block should equal ffn(h_c) exactly (state + 0*a = state), independent
    # of the (here zeroed) RNA tokens, unlike the unconditional residual cell.
    model = _model(n_blocks=1).eval()
    functional = _functional_inputs()
    tokens_zero = torch.zeros(3, 6, 256)
    tokens_random = torch.randn(3, 6, 256)
    peak = model.track_embedding(functional["functional_track_indices"], functional["functional_offsets"])
    dense = model.dense_encoder(functional["functional_dense"])
    h_c = model.locus_norm(peak + dense)
    state0 = h_c[None, :, :].expand(3, -1, -1)

    def run(tokens):
        state = state0
        for i, (attn, ffn) in enumerate(zip(model.retrieval_attn, model.retrieval_ffn)):
            a = attn(state, tokens)
            gate = torch.tanh(model.residual_gate[i])
            state = state + gate * a
            state = ffn(state)
        return state

    out_zero = run(tokens_zero)
    out_random = run(tokens_random)
    torch.testing.assert_close(out_zero, out_random)
    torch.testing.assert_close(out_zero, model.retrieval_ffn[0](state0))


def test_nonzero_gate_lets_attention_output_through():
    model = _model(n_blocks=1).eval()
    with torch.no_grad():
        model.residual_gate.fill_(2.0)  # tanh(2) ~= 0.964, far from 0
    functional = _functional_inputs()
    tokens = torch.randn(3, 6, 256)
    peak = model.track_embedding(functional["functional_track_indices"], functional["functional_offsets"])
    dense = model.dense_encoder(functional["functional_dense"])
    h_c = model.locus_norm(peak + dense)
    state = h_c[None, :, :].expand(3, -1, -1)
    a = model.retrieval_attn[0](state, tokens)
    gated = state + torch.tanh(model.residual_gate[0]) * a
    ungated_identity = model.retrieval_ffn[0](state)
    assert not torch.allclose(model.retrieval_ffn[0](gated), ungated_identity, atol=1e-4)


def test_mean_head_gradient_isolation():
    model = _model(n_blocks=1).train()
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
    assert model.residual_gate.grad is None or torch.all(model.residual_gate.grad == 0)


def test_full_forward_backward_reaches_every_module_and_gate():
    # At the zero-gate init, attn/rna_encoder correctly get NO gradient
    # (gate=tanh(0)=0 zeroes d(loss)/d(a) exactly -- see
    # test_zero_gate_makes_block_ignore_attention_output and the module
    # docstring); only the gate parameter itself has a nonzero local
    # derivative there. Open the gates first so this test can verify the
    # rest of the wiring (attn/rna_encoder reachable once the gate opens).
    model = _model(n_blocks=3).train()
    with torch.no_grad():
        model.residual_gate.fill_(0.5)
    functional = _functional_inputs()
    rna = torch.randn(3, 48)
    target = torch.rand(3, 4)
    out = model(rna, **functional)
    model.zero_grad(set_to_none=True)
    loss = ((out["beta"] - target) ** 2).mean() + 0.15 * ((out["mu_hat"] - target.mean(0)) ** 2).mean()
    loss.backward()

    def has_grad(t):
        return t.grad is not None and torch.any(t.grad != 0)

    for module in (model.rna_encoder, model.track_embedding, model.dense_encoder,
                   model.final_regressor, model.mean_head):
        assert any(has_grad(p) for p in module.parameters())
    for attn, ffn in zip(model.retrieval_attn, model.retrieval_ffn):
        assert any(has_grad(p) for p in attn.parameters())
        assert any(has_grad(p) for p in ffn.parameters())
    # Gate itself must receive gradient too -- it's on the path from attn's
    # output to the loss, at gate=0 (init) the local derivative w.r.t. alpha
    # is attn_output * (1 - tanh(0)^2) = attn_output, nonzero in general.
    assert has_grad(model.residual_gate)


def test_locus_chunking_matches_unchunked():
    torch.manual_seed(0)
    model = _model(n_blocks=2, input_dim=48, n_programs=6).eval()
    functional = _functional_inputs()
    rna = torch.randn(3, 48)
    model.LOCUS_CHUNK = 2
    chunked = model(rna, **functional)["beta"]
    model.LOCUS_CHUNK = 10_000
    unchunked = model(rna, **functional)["beta"]
    torch.testing.assert_close(chunked, unchunked)


def test_rejects_invalid_n_blocks():
    import pytest
    with pytest.raises(ValueError):
        GatedResidualPredictor(48, _config(), n_blocks=0)
