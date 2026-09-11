from __future__ import annotations

import torch

from methylation_predictor.config import EncoderConfig, LossConfig, ModelConfig
from methylation_predictor.losses import (
    sample_correlation_loss,
    within_locus_centered_mse_loss,
)
from methylation_predictor.modeling import SingleRetrievalPredictor
from methylation_predictor.modeling.retrieval import LocusToRNAAttention


def _model(*, dropout: float = 0.15):
    cfg = ModelConfig(
        encoder=EncoderConfig(
            kind="locus_attention",
            latent_dim=8,
            n_programs=6,
            program_dim=256,
            n_heads=4,
            dropout=0.0,
        ),
    )
    return SingleRetrievalPredictor(
        48,
        cfg,
        final_regressor_dropout=dropout,
    )


def _inputs():
    return {
        "functional_track_indices": torch.tensor(
            [1, 2, 5, 9], dtype=torch.int64
        ),
        "functional_offsets": torch.tensor(
            [0, 2, 2, 3, 4], dtype=torch.int64
        ),
        "functional_dense": torch.randn(4, 23),
    }


def _has_grad(module):
    return any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in module.parameters()
    )


def test_forward_shapes_and_no_genomic_dependency():
    model = _model().eval()
    inputs = _inputs()
    rna = torch.randn(3, 48)

    with torch.no_grad():
        first = model(rna, torch.randn(4, 1536), **inputs)
        second = model(rna, torch.randn(4, 7), **inputs)

    assert first["beta"].shape == (3, 4)
    assert first["mu_hat"].shape == (4,)
    assert torch.equal(first["beta"], second["beta"])
    assert torch.equal(first["mu_hat"], second["mu_hat"])


def test_mu_hat_never_feeds_beta_hat_at_inference():
    model = _model().eval()
    inputs = _inputs()
    rna = torch.randn(3, 48)

    with torch.no_grad():
        out1 = model(rna, None, **inputs)
        for parameter in model.mean_head.parameters():
            parameter.add_(1000.0)
        out2 = model(rna, None, **inputs)

    assert torch.equal(out1["beta"], out2["beta"])
    assert not torch.equal(out1["mu_hat"], out2["mu_hat"])


def test_mean_proxy_gradient_isolated_from_patient_specific_path():
    model = _model().train()
    out = model(torch.randn(3, 48), None, **_inputs())
    out["mu_hat"].sum().backward()

    assert _has_grad(model.track_embedding)
    assert _has_grad(model.dense_encoder)
    assert _has_grad(model.locus_norm)
    assert _has_grad(model.mean_head)

    assert not _has_grad(model.rna_encoder)
    assert not _has_grad(model.cross_attention)
    assert not _has_grad(model.final_regressor)


def test_beta_gradient_reaches_both_locus_and_rna_branches_but_not_mean_head():
    model = _model().train()
    out = model(torch.randn(3, 48), None, **_inputs())
    out["beta"].sum().backward()

    assert _has_grad(model.track_embedding)
    assert _has_grad(model.dense_encoder)
    assert _has_grad(model.rna_encoder)
    assert _has_grad(model.cross_attention)
    assert _has_grad(model.final_regressor)
    assert not _has_grad(model.mean_head)


def test_retrieval_attention_has_no_implicit_locus_residual():
    torch.manual_seed(0)
    attn = LocusToRNAAttention(width=8, n_heads=2, dropout=0.0)
    h_c = torch.randn(3, 8)
    tokens = torch.randn(2, 5, 8)

    with torch.no_grad():
        for parameter in attn.parameters():
            parameter.zero_()

    out = attn(h_c, tokens)
    assert torch.equal(out, torch.zeros_like(out))


def test_separate_concat_norm_is_fixed_in_paper_candidate():
    model = _model()
    assert isinstance(model.h_c_concat_norm, torch.nn.LayerNorm)
    assert isinstance(model.r_pc_concat_norm, torch.nn.LayerNorm)

    out = model(torch.randn(3, 48), None, **_inputs())
    assert torch.isfinite(out["beta"]).all()

    out["beta"].sum().backward()
    assert _has_grad(model.h_c_concat_norm)
    assert _has_grad(model.r_pc_concat_norm)


def test_final_regressor_dropout_train_vs_eval():
    model = _model(dropout=0.5)
    inputs = _inputs()
    rna = torch.randn(3, 48)

    model.train()
    with torch.no_grad():
        train_a = model(rna, None, **inputs)["beta"]
        train_b = model(rna, None, **inputs)["beta"]
    assert not torch.equal(train_a, train_b)

    model.eval()
    with torch.no_grad():
        eval_a = model(rna, None, **inputs)["beta"]
        eval_b = model(rna, None, **inputs)["beta"]
    assert torch.equal(eval_a, eval_b)


# The two helpers below remain tested while they still exist in losses.py.
# They are historical objective variants, not part of the J0 final recipe.


def test_sample_correlation_loss_is_per_patient_across_cpgs():
    pred = torch.tensor(
        [[0.1, 0.5, 0.9, 0.3], [0.9, 0.1, 0.2, 0.8]]
    )
    target = torch.tensor(
        [[0.2, 1.0, 1.8, 0.6], [5.0, 5.0, 5.0, 5.0]]
    )
    mask = torch.ones_like(target, dtype=torch.bool)
    cfg = LossConfig(sample_pearson_min_observed_cpgs=2)

    loss, mean_rho, n_valid = sample_correlation_loss(
        pred, target, mask, cfg
    )

    assert n_valid == 1
    assert torch.isclose(mean_rho, torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4)


def test_within_locus_centered_mse_ignores_locus_constant_offset():
    target = torch.rand(6, 4)
    offset = torch.tensor([0.3, -0.1, 0.2, 0.05])
    pred = target + offset[None, :]
    mask = torch.ones_like(target, dtype=torch.bool)
    cfg = LossConfig(locus_min_observed_samples=2)

    loss, n_valid = within_locus_centered_mse_loss(
        pred, target, mask, cfg
    )

    assert n_valid == 4
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)
