from __future__ import annotations

import pytest
import torch

from methylation_predictor.config import EncoderConfig, ModelConfig
from methylation_predictor.modeling import FunctionalGeneFFNFusionPredictor


def _config(n_programs=6):
    return ModelConfig(
        encoder=EncoderConfig(kind="locus_attention", latent_dim=16, n_programs=n_programs,
                               program_dim=256, n_heads=4, dropout=0.0),
    )


def _model(fusion_mode, input_dim=48, n_programs=6, dropout=0.15, **kwargs):
    torch.manual_seed(17)
    return FunctionalGeneFFNFusionPredictor(
        input_dim, _config(n_programs), fusion_mode=fusion_mode,
        final_regressor_dropout=dropout, **kwargs,
    )


def _functional_inputs():
    return {
        "functional_track_indices": torch.tensor([1, 2, 5, 9], dtype=torch.int64),
        "functional_offsets": torch.tensor([0, 2, 2, 3, 4], dtype=torch.int64),
        "functional_dense": torch.randn(4, 23),
    }


@pytest.mark.parametrize("fusion_mode", ["concat", "film", "two_stream_residual"])
def test_forward_shapes(fusion_mode):
    model = _model(fusion_mode).eval()
    out = model(torch.randn(3, 48), **_functional_inputs())
    assert out["beta"].shape == (3, 4)
    assert torch.all((out["beta"] >= 0) & (out["beta"] <= 1))
    assert out["mu_hat"].shape == (4,)


@pytest.mark.parametrize("fusion_mode", ["concat", "film", "two_stream_residual"])
def test_branch_ffn_block_counts(fusion_mode):
    model = _model(fusion_mode)
    assert len(model.functional_ffn) == 2
    assert len(model.gene_expr_ffn) == 2


@pytest.mark.parametrize("fusion_mode", ["concat", "film", "two_stream_residual"])
def test_locus_chunking_matches_unchunked(fusion_mode):
    torch.manual_seed(0)
    model = _model(fusion_mode, input_dim=48, n_programs=6).eval()
    functional = _functional_inputs()
    rna = torch.randn(3, 48)
    model.LOCUS_CHUNK = 2
    chunked = model(rna, **functional)["beta"]
    model.LOCUS_CHUNK = 10_000
    unchunked = model(rna, **functional)["beta"]
    torch.testing.assert_close(chunked, unchunked)


@pytest.mark.parametrize("fusion_mode", ["concat", "film", "two_stream_residual"])
def test_full_forward_backward_reaches_every_module(fusion_mode):
    model = _model(fusion_mode, n_programs=6).train()
    functional = _functional_inputs()
    rna = torch.randn(3, 48)
    target = torch.rand(3, 4)
    out = model(rna, **functional)
    model.zero_grad(set_to_none=True)
    loss = ((out["beta"] - target) ** 2).mean() + 0.15 * ((out["mu_hat"] - target.mean(0)) ** 2).mean()
    loss.backward()

    def has_grad(module):
        return any(p.grad is not None and torch.any(p.grad != 0) for p in module.parameters())

    for module in (
        model.rna_encoder, model.track_embedding, model.dense_encoder,
        model.final_regressor, model.mean_head, model.retrieval_attn,
        model.functional_ffn, model.gene_expr_ffn, model.head_ffn,
    ):
        assert has_grad(module)
    if fusion_mode == "concat":
        assert has_grad(model.fusion_proj)


@pytest.mark.parametrize("fusion_mode", ["concat", "film", "two_stream_residual"])
def test_head_is_identical_shape_across_fusion_modes(fusion_mode):
    # The head (head_ffn + final_regressor) must be byte-for-byte identical
    # across fusion_mode so the three cells stay a controlled comparison of
    # the fusion mechanism alone -- only fusion_proj (concat only) differs.
    model = _model(fusion_mode)
    assert len(model.head_ffn) == 2
    assert model.final_regressor[0].in_features == model.WIDTH


def test_concat_has_fusion_proj_others_dont():
    assert hasattr(_model("concat"), "fusion_proj")
    assert _model("concat").fusion_proj.in_features == 2 * 256
    assert not hasattr(_model("film"), "fusion_proj")
    assert not hasattr(_model("two_stream_residual"), "fusion_proj")


def test_film_generator_only_present_for_film_mode():
    assert hasattr(_model("film"), "film_generator")
    assert not hasattr(_model("concat"), "film_generator")
    assert not hasattr(_model("two_stream_residual"), "film_generator")


def test_two_stream_projections_only_present_for_that_mode():
    two_stream = _model("two_stream_residual")
    assert hasattr(two_stream, "gene_to_functional")
    assert hasattr(two_stream, "functional_to_gene")
    assert not hasattr(_model("concat"), "gene_to_functional")
    assert not hasattr(_model("film"), "gene_to_functional")


def test_rejects_invalid_fusion_mode():
    with pytest.raises(ValueError):
        FunctionalGeneFFNFusionPredictor(48, _config(), fusion_mode="bogus")
