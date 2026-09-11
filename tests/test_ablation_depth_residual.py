from __future__ import annotations

import torch

from methylation_predictor.config import EncoderConfig, ModelConfig
from methylation_predictor.modeling import DepthResidualAblationPredictor


def _config(n_programs=6):
    return ModelConfig(
        encoder=EncoderConfig(kind="locus_attention", latent_dim=16, n_programs=n_programs,
                               program_dim=256, n_heads=4, dropout=0.0),
    )


def _model(n_blocks, attn_residual, input_dim=48, n_programs=6, dropout=0.15):
    torch.manual_seed(17)
    return DepthResidualAblationPredictor(
        input_dim, _config(n_programs), n_blocks=n_blocks, attn_residual=attn_residual,
        final_regressor_dropout=dropout,
    )


def _functional_inputs():
    return {
        "functional_track_indices": torch.tensor([1, 2, 5, 9], dtype=torch.int64),
        "functional_offsets": torch.tensor([0, 2, 2, 3, 4], dtype=torch.int64),
        "functional_dense": torch.randn(4, 23),
    }


def test_forward_shapes_both_cells():
    for n_blocks, attn_residual in [(1, True), (4, False)]:
        model = _model(n_blocks, attn_residual).eval()
        out = model(torch.randn(3, 48), **_functional_inputs())
        assert out["beta"].shape == (3, 4)
        assert torch.all((out["beta"] >= 0) & (out["beta"] <= 1))
        assert out["mu_hat"].shape == (4,)
        assert len(model.retrieval_attn) == n_blocks


def test_no_residual_cell_replaces_state_each_block():
    # attn_residual=False: with V=0 (zeroed RNA tokens), attention collapses
    # to a locus/patient-invariant constant, and with NO residual that
    # constant fully overwrites `state` each block -- so after N_BLOCKS
    # blocks, `state` is the same constant for every locus (the final
    # concat's variation comes only from h_c, not from `state`).
    model = _model(n_blocks=4, attn_residual=False).eval()
    functional = _functional_inputs()
    tokens = torch.zeros(3, 6, 256)
    peak = model.track_embedding(functional["functional_track_indices"], functional["functional_offsets"])
    dense = model.dense_encoder(functional["functional_dense"])
    h_c = model.locus_norm(peak + dense)
    state = h_c[None, :, :].expand(3, -1, -1)
    for attn, ffn in zip(model.retrieval_attn, model.retrieval_ffn):
        a = attn(state, tokens)
        state = a  # no residual
        state = ffn(state)
    for i in range(1, state.shape[1]):
        torch.testing.assert_close(state[:, i], state[:, 0])


def test_residual_cell_keeps_locus_variation():
    # attn_residual=True (depth=1): mirrors J1's per-block residual -- with
    # V=0, `state` still varies across loci because h_c itself does and the
    # residual preserves it.
    model = _model(n_blocks=1, attn_residual=True).eval()
    functional = _functional_inputs()
    tokens = torch.zeros(3, 6, 256)
    peak = model.track_embedding(functional["functional_track_indices"], functional["functional_offsets"])
    dense = model.dense_encoder(functional["functional_dense"])
    h_c = model.locus_norm(peak + dense)
    state = h_c[None, :, :].expand(3, -1, -1)
    for attn, ffn in zip(model.retrieval_attn, model.retrieval_ffn):
        state = state + attn(state, tokens)
        state = ffn(state)
    assert not torch.allclose(state[:, 0], state[:, 1], atol=1e-4)


def test_mean_head_gradient_isolation_both_cells():
    for n_blocks, attn_residual in [(1, True), (4, False)]:
        model = _model(n_blocks, attn_residual).train()
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


def test_full_forward_backward_reaches_every_module_both_cells():
    for n_blocks, attn_residual in [(1, True), (4, False)]:
        model = _model(n_blocks, attn_residual).train()
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


def test_locus_chunking_matches_unchunked_both_cells():
    torch.manual_seed(0)
    for n_blocks, attn_residual in [(1, True), (4, False)]:
        model = _model(n_blocks, attn_residual, input_dim=48, n_programs=6).eval()
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
        DepthResidualAblationPredictor(48, _config(), n_blocks=0, attn_residual=True)
