from __future__ import annotations

import torch
import pytest

from methylation_predictor.config import EncoderConfig, InteractionConfig, ModelConfig
from methylation_predictor.models import FeatureFusionArchitectureVariantModel, LocusConditionedRNAEncoder


def _config() -> ModelConfig:
    return ModelConfig(
        encoder=EncoderConfig(
            kind="locus_attention", latent_dim=8, n_programs=4,
            program_dim=8, n_heads=2, dropout=0.0, layer_norm=True,
        ),
        interaction=InteractionConfig(kind="concat", hidden_dim=6, dropout=0.0),
        variance_normalized_residual=True,
    )


def _model(query_source: str) -> FeatureFusionArchitectureVariantModel:
    return FeatureFusionArchitectureVariantModel(
        input_dim=12, cpg_input_dim=10, config=_config(),
        trunk_hidden_dim=9, bottleneck_dim=5, trunk_dropout=0.0,
        use_mean_branch=True, use_raw_product=False, fusion_init_std=0.01,
        query_source=query_source,
    ).eval()


def test_hybrid_query_is_exact_ntv3_query_at_initialization_after_common_weights_are_loaded():
    torch.manual_seed(7)
    q0 = _model("ntv3")
    torch.manual_seed(13)
    hybrid = _model("hybrid_joint")
    missing, unexpected = hybrid.load_state_dict(q0.state_dict(), strict=False)
    assert unexpected == []
    assert missing == ["rna_encoder.mean_query.weight"]
    assert torch.count_nonzero(hybrid.rna_encoder.mean_query.weight) == 0
    rna = torch.randn(3, 12)
    cpg = torch.randn(4, 10)
    with torch.no_grad():
        assert torch.allclose(q0(rna, cpg)["beta"], hybrid(rna, cpg)["beta"], atol=1e-7)


def _encoder(query_source: str) -> LocusConditionedRNAEncoder:
    enc = LocusConditionedRNAEncoder(
        input_dim=7, locus_dim=6, n_programs=4, program_dim=8,
        n_heads=2, dropout=0.0, layer_norm=True, bottleneck_dim=5,
        mean_query_dim=3, query_source=query_source,
    )
    if enc.mean_query is not None:
        torch.nn.init.normal_(enc.mean_query.weight, std=0.2)
    return enc


def test_hybrid_detached_blocks_attention_gradient_into_mean_features():
    enc = _encoder("hybrid_detached")
    tokens = enc(torch.randn(2, 7)).program_tokens
    mean = torch.randn(5, 3, requires_grad=True)
    out = enc.attend(tokens, torch.randn(5, 6), mean_features=mean)
    out.sum().backward()
    assert mean.grad is None


def test_hybrid_joint_allows_attention_gradient_into_mean_features():
    enc = _encoder("hybrid_joint")
    tokens = enc(torch.randn(2, 7)).program_tokens
    mean = torch.randn(5, 3, requires_grad=True)
    out = enc.attend(tokens, torch.randn(5, 6), mean_features=mean)
    out.sum().backward()
    assert mean.grad is not None
    assert torch.count_nonzero(mean.grad) > 0


def test_mean_aware_query_requires_mean_branch():
    with pytest.raises(ValueError, match="requires use_mean_branch"):
        FeatureFusionArchitectureVariantModel(
            input_dim=12, cpg_input_dim=10, config=_config(),
            use_mean_branch=False, use_raw_product=False,
            query_source="mean_only",
        )
