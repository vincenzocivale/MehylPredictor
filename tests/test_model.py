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

    cfg.gate.kind = "variability"
    cfg.gate.hidden_dim = 64
    cfg.gate.dropout = 0.0

    cfg.anchor_to_mean_rna = True
    cfg.zero_init_residual = True
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
