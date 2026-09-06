"""Tests for the architecture-novelty suite (architecture_novelty_2026_09).

Covers the building blocks shared across the suite's arms and the
``FeatureFusionArchitectureVariantModel`` shared-backbone family:

1. the windowed axial-attention arm degrades gracefully to identity; and
2. the all-default configuration of ``FeatureFusionArchitectureVariantModel``
   is the reference architecture, making it a valid same-code control.

A Hyper-Connections/Manifold-Constrained-HC multi-stream trunk kind was tried
and removed (see ``docs/RNA_METHYLATION.md`` and CLAUDE.md's "Model
compatibility note") -- this file no longer tests it.
"""
from __future__ import annotations

import pytest
import torch

from methylation_predictor.config import (
    AxialConfig,
    EncoderConfig,
    InteractionConfig,
    LossConfig,
    ModelConfig,
    TrunkConfig,
)
from methylation_predictor.models import (
    AxialCpGAttention,
    BilinearInteraction,
    FeatureFusionArchitectureVariantModel,
    FeatureFusionLocusCLSModel,
    ProductInteraction,
    RNARepresentation,
    feature_fusion_variant_label,
    is_architecture_variant,
)


# --------------------------------------------------------------------------
# Axial attention.
# --------------------------------------------------------------------------

def test_axial_attention_is_identity_at_initialization():
    axial = AxialCpGAttention(width=8, n_heads=2, window=4, n_distance_buckets=8, dropout=0.0).eval()
    hidden = torch.randn(3, 10, 8)
    with torch.no_grad():
        assert torch.allclose(axial(hidden), hidden, atol=1e-6)


@pytest.mark.parametrize("n_loci", [3, 4, 10, 16])
def test_axial_attention_handles_windows_that_do_not_divide_the_axis(n_loci):
    axial = AxialCpGAttention(width=8, n_heads=2, window=4, n_distance_buckets=8, dropout=0.0).eval()
    torch.nn.init.normal_(axial.out.weight, std=0.1)
    hidden = torch.randn(2, n_loci, 8)
    positions = torch.arange(n_loci) * 137
    with torch.no_grad():
        out = axial(hidden, positions=positions)
        out_no_positions = axial(hidden)
    assert out.shape == hidden.shape
    assert torch.isfinite(out).all()
    assert torch.isfinite(out_no_positions).all()


def test_axial_attention_does_not_mix_across_the_sample_axis():
    """Cross-sample attention would be transductive; the arm must stay per-sample."""
    axial = AxialCpGAttention(width=8, n_heads=2, window=4, n_distance_buckets=8, dropout=0.0).eval()
    torch.nn.init.normal_(axial.out.weight, std=0.5)
    torch.nn.init.normal_(axial.qkv.weight, std=0.5)
    hidden = torch.randn(3, 9, 8)
    with torch.no_grad():
        together = axial(hidden)
        alone = axial(hidden[1:2])
    assert torch.allclose(together[1:2], alone, atol=1e-5)


# --------------------------------------------------------------------------
# Recorded algebraic finding about the 2026-08 fusion suite.
# --------------------------------------------------------------------------

def test_bilinear_at_canonical_rank_is_the_product_term():
    """``BilinearInteraction`` at rank == min(rna_dim, locus_dim) IS the
    reference ``ProductInteraction`` -- the shared building block both the
    retired two-stage fusion-mechanism ablation and the current
    ``interaction.kind`` baseline-reproduction axis (Bilinear RNA-CpG,
    docs/PAPER_EXPERIMENTS.md) rely on.

    Both compute ``Linear(rna) * Linear(loci)`` concatenated with the raw pieces, so
    with weights transplanted across they agree to the bit.
    """
    rna_dim, locus_dim = 16, 24
    product = ProductInteraction(rna_dim, locus_dim, 8, 0.0).eval()
    bilinear = BilinearInteraction(rna_dim, locus_dim, 8, 0.0, rank=min(rna_dim, locus_dim)).eval()
    bilinear.rna_bilinear.load_state_dict(product.rna_product.state_dict())
    bilinear.locus_bilinear.load_state_dict(product.locus_product.state_dict())
    bilinear.network.load_state_dict(product.network.state_dict())

    assert sum(p.numel() for p in product.parameters()) == sum(p.numel() for p in bilinear.parameters())
    generator = torch.Generator().manual_seed(11)
    rna = RNARepresentation(torch.randn(4, rna_dim, generator=generator))
    loci = torch.randn(6, locus_dim, generator=generator)
    with torch.no_grad():
        assert torch.equal(product(rna, loci), bilinear(rna, loci))


# ==========================================================================
# FeatureFusionArchitectureVariantModel -- the shared-backbone suite
# (architecture_novelty_2026_09, retargeted 2026-09-04 after
# FeatureFusionLocusCLSModel became the primary/reference architecture).
#
# This model family has no prior/sigma anchor, so there is no "starts exactly
# at the prior" invariant to test. Its own established convention
# (FeatureFusionLocusCLSModel's fusion_init_std) is a small NONZERO init; the
# exact, testable invariant is fusion_init_std=0.0 -> every prediction is
# exactly 0.5, whatever the encoder/trunk/axial configuration.
# ==========================================================================

FF_INPUT_DIM = 48
FF_CPG_DIM = 32
FF_SAMPLES = 4
FF_LOCI = 6


def _ff_config(**overrides) -> ModelConfig:
    config = ModelConfig(
        encoder=EncoderConfig(latent_dim=16, hidden_dim=24, n_programs=6, program_dim=8, n_heads=2),
        interaction=InteractionConfig(hidden_dim=12),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _ff_batch(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    rna = torch.randn(FF_SAMPLES, FF_INPUT_DIM, generator=generator)
    loci = torch.randn(FF_LOCI, FF_CPG_DIM, generator=generator)
    return rna, loci


def _ff_build(config: ModelConfig, **kwargs) -> FeatureFusionArchitectureVariantModel:
    kwargs.setdefault("trunk_hidden_dim", 20)
    kwargs.setdefault("bottleneck_dim", 10)
    return FeatureFusionArchitectureVariantModel(FF_INPUT_DIM, FF_CPG_DIM, config, **kwargs)


FF_ARM_CONFIGS = {
    "default_equivalent": {},
    "encoder_mlp": {"encoder": EncoderConfig(kind="mlp", latent_dim=16, hidden_dim=24)},
    "encoder_locus_attention": {
        "encoder": EncoderConfig(kind="locus_attention", latent_dim=16, n_programs=6, program_dim=8, n_heads=2)
    },
    "encoder_bottleneck_mlp": {
        "encoder": EncoderConfig(kind="bottleneck_mlp", latent_dim=16, hidden_dim=20, n_blocks=2, mlp_ratio=2)
    },
    "encoder_frozen_embedding": {
        "encoder": EncoderConfig(kind="frozen_embedding", latent_dim=16, frozen_embedding_source="test_fm")
    },
    "trunk_plain": {"trunk": TrunkConfig(kind="plain", depth=2, width=10)},
    "axial": {"trunk": TrunkConfig(kind="plain", depth=1, width=10), "axial": AxialConfig(enabled=True, n_heads=2, window=3)},
    "beta_head": {"beta_likelihood_head": True},
    "no_mean_branch": {},  # exercised via use_mean_branch=False below
}


@pytest.mark.parametrize("name", sorted(FF_ARM_CONFIGS))
def test_ff_arm_with_zero_fusion_init_predicts_exactly_half(name):
    """fusion_init_std=0.0 must zero the final head's weight AND bias, so every
    prediction is exactly sigmoid(0) = 0.5 regardless of encoder/trunk/axial."""
    kwargs = {"fusion_init_std": 0.0}
    if name == "no_mean_branch":
        kwargs["use_mean_branch"] = False
    model = _ff_build(_ff_config(**FF_ARM_CONFIGS[name]), **kwargs).eval()
    rna, loci = _ff_batch()
    with torch.no_grad():
        beta = model(rna, loci)["beta"]
    assert beta.shape == (FF_SAMPLES, FF_LOCI)
    assert torch.allclose(beta, torch.full_like(beta, 0.5), atol=1e-6)


@pytest.mark.parametrize("name", sorted(FF_ARM_CONFIGS))
def test_ff_arm_produces_finite_gradients_everywhere(name):
    kwargs = {}
    if name == "no_mean_branch":
        kwargs["use_mean_branch"] = False
    model = _ff_build(_ff_config(**FF_ARM_CONFIGS[name]), **kwargs).train()
    rna, loci = _ff_batch(seed=1)
    target = torch.rand(FF_SAMPLES, FF_LOCI)
    target[0, 0] = float("nan")
    loss_cfg = LossConfig(locus_pearson_weight=0.15, locus_min_observed_samples=2)
    if FF_ARM_CONFIGS[name].get("beta_likelihood_head") or name == "beta_head":
        loss_cfg.beta_nll_weight = 0.1
    # The real trainer always passes CpG ordinals when the axial arm is active
    # (LocusCLSJointTrainer._position_kwargs); without them axial.distance_scale
    # is a genuinely inert parameter (the decay term is skip-added only when
    # positions are given), so this only matters for the "axial" arm here.
    position_kwargs = {"cpg_positions": torch.arange(FF_LOCI) * 91} if model.requires_cpg_positions else {}
    outputs = model(rna, loci, **position_kwargs)
    from methylation_predictor.rna_training.locus_cls_trainer import _direct_beta_loss
    loss, pieces = _direct_beta_loss(outputs, target, loss_cfg)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [(n, p.grad) for n, p in model.named_parameters() if p.requires_grad]
    missing = [n for n, g in grads if g is None]
    # residual_head/mean_head only receive gradient through the auxiliary
    # residual/mean losses, which locus_cls_trainer applies separately
    # (residual_aux_weight/aux_weight) -- neither is exercised by
    # _direct_beta_loss alone, so both are expected to be gradient-free here.
    missing = [n for n in missing if not (n.startswith("residual_head") or n.startswith("mean_head"))]
    assert not missing, f"no gradient reached: {missing}"
    non_finite = [n for n, g in grads if g is not None and not torch.isfinite(g).all()]
    assert not non_finite, f"non-finite gradient: {non_finite}"
    assert pieces["observed"] == FF_SAMPLES * FF_LOCI - 1


def test_ff_default_matches_reference_parameter_shapes():
    config = _ff_config()
    assert not is_architecture_variant(config)
    variant = _ff_build(config)
    reference = FeatureFusionLocusCLSModel(
        FF_INPUT_DIM, FF_CPG_DIM, config, trunk_hidden_dim=20, bottleneck_dim=10,
    )
    assert sorted(tuple(p.shape) for p in variant.parameters()) == sorted(tuple(p.shape) for p in reference.parameters())
    assert sum(p.numel() for p in variant.parameters()) == sum(p.numel() for p in reference.parameters())


def test_ff_default_matches_reference_outputs_under_copied_weights():
    """At every default, FeatureFusionArchitectureVariantModel must reproduce
    FeatureFusionLocusCLSModel's forward computation exactly -- the same
    same-code-control role this class plays for the architecture-novelty
    suite."""
    config = _ff_config()
    variant = _ff_build(config, fusion_init_std=0.05).eval()
    reference = FeatureFusionLocusCLSModel(
        FF_INPUT_DIM, FF_CPG_DIM, config, trunk_hidden_dim=20, bottleneck_dim=10, fusion_init_std=0.05,
    ).eval()

    variant.rna_encoder.load_state_dict(reference.rna_encoder.state_dict())
    variant.rna_product.load_state_dict(reference.rna_product.state_dict())
    variant.locus_product.load_state_dict(reference.locus_product.state_dict())
    variant.raw_branch.load_state_dict(reference.raw_branch.state_dict())
    variant.residual_head.load_state_dict(reference.residual_head.state_dict())
    variant.trunk_cpg.load_state_dict(reference.trunk.state_dict())
    variant.mean_head.load_state_dict(reference.mean_head.state_dict())
    variant.fusion.load_state_dict(reference.fusion.state_dict())

    rna, loci = _ff_batch(seed=2)
    with torch.no_grad():
        assert torch.allclose(variant(rna, loci)["beta"], reference(rna, loci)["beta"], atol=1e-6)


# --------------------------------------------------------------------------
# include_raw_rna/include_raw_cpg -- the axis that reproduces the
# paper-required simplified baselines under this engine
# (docs/PAPER_EXPERIMENTS.md).
# --------------------------------------------------------------------------

def test_ff_raw_branch_requires_at_least_one_piece():
    with pytest.raises(ValueError, match="include_raw_rna/include_raw_cpg/use_raw_product"):
        _ff_build(_ff_config(), include_raw_rna=False, include_raw_cpg=False, use_raw_product=False)


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("global_rna_shift", {"use_mean_branch": False, "include_raw_cpg": False, "use_raw_product": False}),
        ("bilinear_rna_cpg", {"use_mean_branch": False, "include_raw_rna": False, "include_raw_cpg": False}),
        ("mlp_rna_cpg", {"use_mean_branch": False, "use_raw_product": False}),
    ],
)
def test_ff_baseline_equivalent_configs_produce_finite_gradients(name, kwargs):
    model = _ff_build(_ff_config(), **kwargs).train()
    rna, loci = _ff_batch(seed=3)
    target = torch.rand(FF_SAMPLES, FF_LOCI)
    outputs = model(rna, loci)
    assert outputs["beta"].shape == (FF_SAMPLES, FF_LOCI)
    loss = torch.nn.functional.mse_loss(outputs["beta"], target)
    loss.backward()
    # residual_head is an auxiliary probe on h_raw, only trained by the real
    # trainer's separate residual_aux_weight loss -- a plain MSE on "beta"
    # never touches it, so it is expected to be gradient-free here (same
    # exclusion as test_ff_arm_produces_finite_gradients_everywhere).
    grads = [(n, p.grad) for n, p in model.named_parameters() if p.requires_grad and not n.startswith("residual_head")]
    assert grads and all(g is not None and torch.isfinite(g).all() for _, g in grads)


def test_ff_global_rna_shift_equivalent_ignores_cpg_embedding():
    """Global RNA Shift baseline: raw_delta must be constant across loci for a
    given sample -- the model never sees the CpG embedding."""
    model = _ff_build(
        _ff_config(), use_mean_branch=False, include_raw_cpg=False, use_raw_product=False,
    ).eval()
    rna, loci = _ff_batch(seed=4)
    other_loci = torch.randn_like(loci)
    with torch.no_grad():
        beta_a = model(rna, loci)["beta"]
        beta_b = model(rna, other_loci)["beta"]
    assert torch.allclose(beta_a, beta_b, atol=1e-6)


def test_ff_variant_labels_are_distinct():
    plain_d2 = feature_fusion_variant_label(_ff_config(trunk=TrunkConfig(kind="plain", depth=2, width=64)))
    plain_d3 = feature_fusion_variant_label(_ff_config(trunk=TrunkConfig(kind="plain", depth=3, width=64)))
    assert plain_d2 != plain_d3

    # A resumed run must not silently mix incompatible bottleneck_mlp depths, or
    # incompatible frozen-embedding sources -- both differ in trainable
    # parameter shapes (n_blocks) or in what input_dim actually means.
    shallow = feature_fusion_variant_label(_ff_config(encoder=EncoderConfig(kind="bottleneck_mlp", n_blocks=2)))
    deep = feature_fusion_variant_label(_ff_config(encoder=EncoderConfig(kind="bottleneck_mlp", n_blocks=6)))
    assert shallow != deep
    bulkrnabert = feature_fusion_variant_label(
        _ff_config(encoder=EncoderConfig(kind="frozen_embedding", frozen_embedding_source="bulkrnabert_tcga"))
    )
    other_source = feature_fusion_variant_label(
        _ff_config(encoder=EncoderConfig(kind="frozen_embedding", frozen_embedding_source="other_fm"))
    )
    assert bulkrnabert != other_source
