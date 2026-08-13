import torch

from methylation_predictor.config import ModelConfig
from methylation_predictor.models import RNA2DNAmModel


def _canonical_config() -> ModelConfig:
    cfg = ModelConfig()
    cfg.encoder.kind = "linear"
    cfg.encoder.latent_dim = 64
    cfg.encoder.layer_norm = True

    cfg.interaction.kind = "concat"
    cfg.interaction.hidden_dim = 128
    cfg.interaction.dropout = 0.1
    cfg.interaction.include_product = True

    cfg.gate.kind = "variability"
    cfg.gate.hidden_dim = 64
    cfg.gate.dropout = 0.0

    cfg.anchor_to_mean_rna = True
    cfg.zero_init_residual = True
    cfg.prediction_mode = "residual_prior"
    return cfg


def test_zero_init_starts_exactly_from_prior():
    torch.manual_seed(1)
    model = RNA2DNAmModel(
        input_dim=17,
        locus_dim=11,
        config=_canonical_config(),
    )
    rna = torch.randn(3, 17)
    loci = torch.randn(5, 11)
    prior = torch.rand(5).clamp(0.05, 0.95)
    variability = torch.randn(5, 2)

    out = model(rna, loci, prior, variability)

    assert out["beta"].shape == (3, 5)
    assert torch.allclose(out["beta"], prior.unsqueeze(0).expand(3, -1), atol=1e-6)
    assert torch.allclose(out["delta_logit"], torch.zeros_like(out["delta_logit"]))


def test_mean_rna_anchor_zeroes_reference_profile_after_nonzero_init():
    torch.manual_seed(2)
    cfg = _canonical_config()
    cfg.zero_init_residual = False
    model = RNA2DNAmModel(input_dim=13, locus_dim=7, config=cfg)
    model.eval()  # dropout is stochastic in train mode and breaks exact cancellation

    reference = torch.randn(1, 13)
    loci = torch.randn(4, 7)
    prior = torch.rand(4).clamp(0.05, 0.95)
    variability = torch.randn(4, 2)

    out = model(reference, loci, prior, variability, reference_rna=reference)

    assert torch.allclose(out["delta_logit"], torch.zeros_like(out["delta_logit"]), atol=1e-6)
    assert torch.allclose(out["beta"], prior.unsqueeze(0), atol=1e-6)


def test_only_canonical_architecture_is_accepted():
    cfg = _canonical_config()
    cfg.interaction.kind = "bilinear"

    try:
        RNA2DNAmModel(input_dim=5, locus_dim=3, config=cfg)
    except ValueError as exc:
        assert "only model.interaction.kind='concat'" in str(exc)
    else:
        raise AssertionError("non-canonical interaction should be rejected")


def test_no_anchor_no_gate_ablation_is_direct_residual():
    torch.manual_seed(3)
    cfg = _canonical_config()
    cfg.anchor_to_mean_rna = False
    cfg.gate.kind = "none"
    cfg.zero_init_residual = False

    model = RNA2DNAmModel(input_dim=13, locus_dim=7, config=cfg)
    model.eval()

    rna = torch.randn(3, 13)
    loci = torch.randn(4, 7)
    prior = torch.rand(4).clamp(0.05, 0.95)
    variability = torch.randn(4, 2)
    reference_a = torch.randn(1, 13)
    reference_b = torch.randn(1, 13)

    out_a = model(rna, loci, prior, variability, reference_rna=reference_a)
    out_b = model(rna, loci, prior, variability, reference_rna=reference_b)

    assert sum(p.numel() for p in model.gate.parameters()) == 0
    assert torch.equal(out_a["delta_logit"], out_a["raw_delta_logit"])
    assert torch.equal(out_a["delta_logit"], out_b["delta_logit"])
    assert torch.equal(out_a["beta"], out_b["beta"])
    assert torch.equal(out_a["gate"], torch.ones_like(out_a["gate"]))


def test_no_anchor_no_gate_zero_init_still_starts_from_prior():
    torch.manual_seed(4)
    cfg = _canonical_config()
    cfg.anchor_to_mean_rna = False
    cfg.gate.kind = "none"

    model = RNA2DNAmModel(input_dim=9, locus_dim=5, config=cfg)
    rna = torch.randn(2, 9)
    loci = torch.randn(6, 5)
    prior = torch.rand(6).clamp(0.05, 0.95)
    variability = torch.randn(6, 2)

    out = model(rna, loci, prior, variability)

    assert torch.allclose(out["beta"], prior.unsqueeze(0).expand(2, -1), atol=1e-6)
    assert torch.allclose(out["delta_logit"], torch.zeros_like(out["delta_logit"]))


def test_unsupported_gate_kind_still_fails_closed():
    cfg = _canonical_config()
    cfg.gate.kind = "global"

    try:
        RNA2DNAmModel(input_dim=5, locus_dim=3, config=cfg)
    except ValueError as exc:
        assert "model.gate.kind must be 'variability'" in str(exc)
    else:
        raise AssertionError("unsupported gate kind should be rejected")


def test_no_product_ablation_removes_product_projection_parameters():
    cfg = _canonical_config()
    cfg.interaction.include_product = False
    cfg.zero_init_residual = False
    model = RNA2DNAmModel(input_dim=13, locus_dim=7, config=cfg)

    assert model.interaction.rna_product is None
    assert model.interaction.locus_product is None

    out = model(
        torch.randn(3, 13),
        torch.randn(4, 7),
        torch.rand(4).clamp(0.05, 0.95),
        torch.randn(4, 2),
    )
    assert out["beta"].shape == (3, 4)


def test_direct_prediction_does_not_add_prior_to_prediction_logit():
    torch.manual_seed(5)
    cfg = _canonical_config()
    cfg.prediction_mode = "direct"
    cfg.anchor_to_mean_rna = False
    cfg.zero_init_residual = False
    model = RNA2DNAmModel(input_dim=13, locus_dim=7, config=cfg)
    model.eval()

    rna = torch.randn(3, 13)
    loci = torch.randn(4, 7)
    variability = torch.randn(4, 2)
    prior_a = torch.full((4,), 0.1)
    prior_b = torch.full((4,), 0.9)

    out_a = model(rna, loci, prior_a, variability)
    out_b = model(rna, loci, prior_b, variability)

    assert torch.allclose(out_a["beta"], out_b["beta"], atol=1e-6)
    assert torch.allclose(out_a["prediction_logit"], out_b["prediction_logit"], atol=1e-6)
    assert not torch.allclose(out_a["delta_logit"], out_b["delta_logit"])


def test_direct_prediction_rejects_mean_rna_anchor():
    cfg = _canonical_config()
    cfg.prediction_mode = "direct"
    cfg.anchor_to_mean_rna = True
    try:
        RNA2DNAmModel(input_dim=5, locus_dim=3, config=cfg)
    except ValueError as exc:
        assert "direct prediction cannot use mean-RNA anchoring" in str(exc)
    else:
        raise AssertionError("direct prediction with mean-RNA anchoring should fail closed")


def test_larger_linear_rna_latent_is_supported():
    cfg = _canonical_config()
    cfg.encoder.latent_dim = 256
    model = RNA2DNAmModel(input_dim=17, locus_dim=11, config=cfg)
    assert model.rna_encoder.projection.out_features == 256
