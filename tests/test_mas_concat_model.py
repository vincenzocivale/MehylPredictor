from __future__ import annotations

import torch

from methylation_predictor.config import EncoderConfig, LossConfig, ModelConfig
from methylation_predictor.losses import sample_correlation_loss, within_locus_centered_mse_loss
from methylation_predictor.models import FunctionalConcatMASModel


def _model():
    cfg = ModelConfig(
        encoder=EncoderConfig(kind="locus_attention", latent_dim=8, n_programs=6,
                              program_dim=256, n_heads=4, dropout=0.0),
    )
    return FunctionalConcatMASModel(48, cfg)


def _inputs():
    return {
        "functional_track_indices": torch.tensor([1, 2, 5, 9], dtype=torch.int64),
        "functional_offsets": torch.tensor([0, 2, 2, 3, 4], dtype=torch.int64),
        "functional_dense": torch.randn(4, 23),
    }


def test_forward_shapes_and_no_genomic_dependency():
    model = _model()
    inputs = _inputs()
    rna = torch.randn(3, 48)
    first = model(rna, torch.randn(4, 1536), **inputs)
    second = model(rna, torch.randn(4, 7), **inputs)
    assert first["beta"].shape == (3, 4)
    assert first["mu_hat"].shape == (4,)  # locus-only, no batch axis
    assert torch.equal(first["beta"], second["beta"])  # genomic embedding input is ignored
    assert torch.equal(first["mu_hat"], second["mu_hat"])


def test_mu_hat_never_feeds_beta_hat_at_inference():
    model = _model().eval()
    inputs = _inputs()
    rna = torch.randn(3, 48)
    with torch.no_grad():
        out1 = model(rna, None, **inputs)
        # Corrupt mean_head's weights after the forward pass has already run --
        # if beta depended on mu_hat this would be a no-op either way, so
        # instead corrupt mean_head BEFORE a second forward and confirm beta
        # is unaffected while mu_hat does change.
        for p in model.mean_head.parameters():
            p.add_(1000.0)
        out2 = model(rna, None, **inputs)
    assert torch.equal(out1["beta"], out2["beta"])
    assert not torch.equal(out1["mu_hat"], out2["mu_hat"])


def test_gradient_contract_mean_loss_isolated_to_functional_encoder_and_mean_head():
    model = _model()
    inputs = _inputs()
    rna = torch.randn(3, 48)
    out = model(rna, None, **inputs)
    L_mean = out["mu_hat"].sum()
    L_mean.backward()

    def has_grad(module):
        return any(p.grad is not None and torch.any(p.grad != 0) for p in module.parameters())

    assert has_grad(model.track_embedding)
    assert has_grad(model.dense_encoder)
    assert has_grad(model.locus_norm)
    assert has_grad(model.mean_head)
    assert not has_grad(model.rna_encoder)
    assert not has_grad(model.cross_attention)
    assert not has_grad(model.final_regressor)


def test_gradient_contract_beta_loss_reaches_every_branch():
    model = _model()
    inputs = _inputs()
    rna = torch.randn(3, 48)
    out = model(rna, None, **inputs)
    L_beta = out["beta"].sum()
    L_beta.backward()

    def has_grad(module):
        return any(p.grad is not None and torch.any(p.grad != 0) for p in module.parameters())

    assert has_grad(model.track_embedding)
    assert has_grad(model.dense_encoder)
    assert has_grad(model.rna_encoder)
    assert has_grad(model.cross_attention)
    assert has_grad(model.final_regressor)
    # mean_head must NOT receive gradient from the beta path (mu_hat is never
    # concatenated into z_pc / read by final_regressor).
    assert not has_grad(model.mean_head)


def test_gradient_contract_sample_pcc_loss_matches_beta_loss_routing():
    model = _model()
    inputs = _inputs()
    rna = torch.randn(4, 48)
    out = model(rna, None, **inputs)
    target = torch.rand(4, 4)
    mask = torch.ones_like(target, dtype=torch.bool)
    loss_cfg = LossConfig(sample_pearson_min_observed_cpgs=2)
    L_pcc, mean_rho, n_valid = sample_correlation_loss(out["beta"], target, mask, loss_cfg)
    L_pcc.backward()

    def has_grad(module):
        return any(p.grad is not None and torch.any(p.grad != 0) for p in module.parameters())

    assert has_grad(model.track_embedding)
    assert has_grad(model.rna_encoder)
    assert has_grad(model.cross_attention)
    assert has_grad(model.final_regressor)
    assert not has_grad(model.mean_head)
    assert n_valid > 0
    assert torch.isfinite(mean_rho)


def _detached_model():
    cfg = ModelConfig(
        encoder=EncoderConfig(kind="locus_attention", latent_dim=8, n_programs=6,
                              program_dim=256, n_heads=4, dropout=0.0),
    )
    return FunctionalConcatMASModel(48, cfg, detach_h_c_main_path=True)


def test_detached_variant_beta_loss_no_longer_reaches_functional_encoder():
    # mas_concat_v2_detached: L_beta/L_sample_PCC must reach rna_encoder,
    # cross_attention, final_regressor as before, but NOT track_embedding/
    # dense_encoder/locus_norm any more -- only L_mean does.
    model = _detached_model()
    inputs = _inputs()
    rna = torch.randn(3, 48)
    out = model(rna, None, **inputs)
    out["beta"].sum().backward()

    def has_grad(module):
        return any(p.grad is not None and torch.any(p.grad != 0) for p in module.parameters())

    assert not has_grad(model.track_embedding)
    assert not has_grad(model.dense_encoder)
    assert not has_grad(model.locus_norm)
    assert has_grad(model.rna_encoder)
    assert has_grad(model.cross_attention)
    assert has_grad(model.final_regressor)
    assert not has_grad(model.mean_head)


def test_detached_variant_mean_loss_still_reaches_functional_encoder_only():
    model = _detached_model()
    inputs = _inputs()
    rna = torch.randn(3, 48)
    out = model(rna, None, **inputs)
    out["mu_hat"].sum().backward()

    def has_grad(module):
        return any(p.grad is not None and torch.any(p.grad != 0) for p in module.parameters())

    assert has_grad(model.track_embedding)
    assert has_grad(model.dense_encoder)
    assert has_grad(model.mean_head)
    assert not has_grad(model.rna_encoder)
    assert not has_grad(model.cross_attention)
    assert not has_grad(model.final_regressor)


def test_detached_variant_sample_pcc_loss_matches_beta_loss_routing():
    model = _detached_model()
    inputs = _inputs()
    rna = torch.randn(4, 48)
    out = model(rna, None, **inputs)
    target = torch.rand(4, 4)
    mask = torch.ones_like(target, dtype=torch.bool)
    loss_cfg = LossConfig(sample_pearson_min_observed_cpgs=2)
    L_pcc, _, n_valid = sample_correlation_loss(out["beta"], target, mask, loss_cfg)
    L_pcc.backward()

    def has_grad(module):
        return any(p.grad is not None and torch.any(p.grad != 0) for p in module.parameters())

    assert not has_grad(model.track_embedding)
    assert has_grad(model.rna_encoder)
    assert has_grad(model.cross_attention)
    assert has_grad(model.final_regressor)
    assert n_valid >= 0


def test_cross_attention_has_no_h_c_residual():
    # Hypothesis 1's core requirement: r_pc = MHA(Q=h_c, K=R_p, V=R_p) must be
    # a pure weighted sum of RNA values, with NO "+h_c" residual mixed in --
    # otherwise the final regressor could shortcut through h_c twice.
    from methylation_predictor.models import SimpleCrossAttention
    torch.manual_seed(0)
    attn = SimpleCrossAttention(width=8, n_heads=2, dropout=0.0)
    h_c = torch.randn(3, 8)
    tokens = torch.randn(2, 5, 8)
    with torch.no_grad():
        # Zero every projection weight/bias: a pure weighted-sum-of-values
        # attention with all-zero out-projection must return exactly zero --
        # if h_c leaked in via a residual, the output would equal h_c instead.
        for p in attn.parameters():
            p.zero_()
    out = attn(h_c, tokens)
    assert torch.equal(out, torch.zeros_like(out))


def test_separate_concat_norm_normalizes_both_halves_independently():
    torch.manual_seed(0)
    cfg = ModelConfig(
        encoder=EncoderConfig(kind="locus_attention", latent_dim=8, n_programs=6,
                              program_dim=256, n_heads=4, dropout=0.0),
    )
    model = FunctionalConcatMASModel(48, cfg, separate_concat_norm=True)
    assert model.h_c_concat_norm is not None and model.r_pc_concat_norm is not None
    inputs = _inputs()
    rna = torch.randn(3, 48)
    out = model(rna, None, **inputs)
    assert out["beta"].shape == (3, 4)
    assert torch.isfinite(out["beta"]).all()
    # Gradient contract must be unchanged by this flag (still reaches every
    # main-path module, still skips mean_head).
    out["beta"].sum().backward()

    def has_grad(module):
        return any(p.grad is not None and torch.any(p.grad != 0) for p in module.parameters())

    assert has_grad(model.h_c_concat_norm) and has_grad(model.r_pc_concat_norm)
    assert has_grad(model.track_embedding) and has_grad(model.cross_attention) and has_grad(model.final_regressor)
    assert not has_grad(model.mean_head)


def test_final_regressor_dropout_is_off_by_default():
    model = _model()
    assert model.final_regressor_dropout == 0.0
    dropouts = [m for m in model.final_regressor if isinstance(m, torch.nn.Dropout)]
    assert len(dropouts) == 2
    assert all(d.p == 0.0 for d in dropouts)


def test_final_regressor_dropout_differs_train_vs_eval():
    torch.manual_seed(0)
    cfg = ModelConfig(
        encoder=EncoderConfig(kind="locus_attention", latent_dim=8, n_programs=6,
                              program_dim=256, n_heads=4, dropout=0.0),
    )
    model = FunctionalConcatMASModel(48, cfg, final_regressor_dropout=0.5)
    inputs = _inputs()
    rna = torch.randn(3, 48)

    model.train()
    with torch.no_grad():
        out_a = model(rna, None, **inputs)["beta"]
        out_b = model(rna, None, **inputs)["beta"]
    assert not torch.equal(out_a, out_b)  # dropout makes two train-mode passes differ

    model.eval()
    with torch.no_grad():
        out_c = model(rna, None, **inputs)["beta"]
        out_d = model(rna, None, **inputs)["beta"]
    assert torch.equal(out_c, out_d)  # deterministic in eval mode


def test_final_regressor_dropout_does_not_change_gradient_routing():
    cfg = ModelConfig(
        encoder=EncoderConfig(kind="locus_attention", latent_dim=8, n_programs=6,
                              program_dim=256, n_heads=4, dropout=0.0),
    )
    model = FunctionalConcatMASModel(48, cfg, final_regressor_dropout=0.3)
    inputs = _inputs()
    rna = torch.randn(3, 48)
    out = model(rna, None, **inputs)
    out["beta"].sum().backward()

    def has_grad(module):
        return any(p.grad is not None and torch.any(p.grad != 0) for p in module.parameters())

    assert has_grad(model.final_regressor) and has_grad(model.cross_attention) and has_grad(model.track_embedding)
    assert not has_grad(model.mean_head)


def test_default_variant_is_unaffected_by_the_new_flag():
    # detach_h_c_main_path defaults to False -- the original H0/H1/H2 gradient
    # contract (already covered above) must be untouched.
    model = _model()
    assert model.detach_h_c_main_path is False


def test_sample_correlation_loss_is_the_mac_direction_not_locus_direction():
    # rho_p = corr_c(beta_hat[p, :], beta[p, :]) -- correlation across CpGs
    # for a fixed patient/row. A row that's a perfect affine function of the
    # target must get rho_p == 1 (loss == 0) regardless of what other rows do.
    pred = torch.tensor([[0.1, 0.5, 0.9, 0.3], [0.9, 0.1, 0.2, 0.8]])
    target = torch.tensor([[0.2, 1.0, 1.8, 0.6], [5.0, 5.0, 5.0, 5.0]])  # row0 = 2*pred; row1 constant (undefined rho)
    mask = torch.ones_like(target, dtype=torch.bool)
    loss_cfg = LossConfig(sample_pearson_min_observed_cpgs=2)
    loss, mean_rho, n_valid = sample_correlation_loss(pred, target, mask, loss_cfg)
    assert n_valid == 1  # only row 0 has nonzero target variance
    assert torch.isclose(mean_rho, torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4)


def test_within_locus_centered_mse_ignores_a_pure_locus_level_offset():
    # If prediction == target + a per-locus (column-wise) constant offset,
    # the between-locus error is nonzero but the WITHIN-locus centered error
    # must be exactly zero: centering removes any purely per-CpG-constant
    # discrepancy.
    target = torch.rand(6, 4)
    offset = torch.tensor([0.3, -0.1, 0.2, 0.05])  # per-CpG constant
    pred = target + offset[None, :]
    mask = torch.ones_like(target, dtype=torch.bool)
    cfg = LossConfig(locus_min_observed_samples=2)
    loss, n_valid = within_locus_centered_mse_loss(pred, target, mask, cfg)
    assert n_valid == 4
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_within_locus_centered_mse_penalizes_within_locus_error():
    # A per-locus-constant prediction (== locus mean, zero within-locus
    # signal at all) must incur nonzero centered-MSE against a target that
    # actually varies within each locus.
    torch.manual_seed(0)
    target = torch.rand(8, 5)
    pred = target.mean(dim=0, keepdim=True).expand_as(target).clone()  # locus-constant
    mask = torch.ones_like(target, dtype=torch.bool)
    cfg = LossConfig(locus_min_observed_samples=2)
    loss, n_valid = within_locus_centered_mse_loss(pred, target, mask, cfg)
    assert n_valid == 5
    assert loss.item() > 0.0
    # Should equal the mean within-locus variance of the target itself
    # (centred pred is exactly zero everywhere).
    expected = target.var(dim=0, unbiased=False).mean()
    assert torch.isclose(loss, expected, atol=1e-4)


def test_within_locus_centered_mse_respects_min_observed_samples():
    target = torch.rand(3, 2)
    pred = torch.rand(3, 2)
    mask = torch.ones_like(target, dtype=torch.bool)
    cfg = LossConfig(locus_min_observed_samples=8)  # more than the 3 rows available
    loss, n_valid = within_locus_centered_mse_loss(pred, target, mask, cfg)
    assert n_valid == 0
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_within_locus_centered_mse_gradient_reaches_prediction():
    pred = torch.rand(5, 4, requires_grad=True)
    target = torch.rand(5, 4)
    mask = torch.ones_like(target, dtype=torch.bool)
    cfg = LossConfig(locus_min_observed_samples=2)
    loss, _ = within_locus_centered_mse_loss(pred, target, mask, cfg)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_sample_correlation_loss_epsilon_stability_on_degenerate_rows():
    # All rows constant (zero target/pred variance): must return a finite,
    # zero-gradient no-op rather than NaN/inf.
    pred = torch.rand(3, 5, requires_grad=True)
    target = torch.full((3, 5), 0.5)
    mask = torch.ones_like(target, dtype=torch.bool)
    loss_cfg = LossConfig(sample_pearson_min_observed_cpgs=2)
    loss, mean_rho, n_valid = sample_correlation_loss(pred, target, mask, loss_cfg)
    assert n_valid == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
