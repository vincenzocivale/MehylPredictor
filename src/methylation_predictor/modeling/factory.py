"""Construction of the RNA methylation model (placeholder until the next model is chosen)."""
from __future__ import annotations

from torch import nn

from ..config import ModelConfig
from .final import EfficientSingleAttentionPredictor

FINAL_VARIANT = "efficient_single_attn_8ffn_residual_functional8_head2"
FINAL_MODEL_KWARGS = {
    "n_ffn_blocks": 8,
    "n_functional_ffn_blocks": 8,
    "deep_query": False,
    "n_head_ffn_blocks": 2,
}
SUPPORTED_FUNCTIONAL_VARIANTS = frozenset({FINAL_VARIANT})


def build_functional_predictor(
    *,
    variant: str,
    input_dim: int,
    config: ModelConfig,
    final_regressor_dropout: float,
    use_mean_proxy: bool,
) -> tuple[nn.Module, str]:
    if variant not in SUPPORTED_FUNCTIONAL_VARIANTS:
        raise ValueError(f"unsupported variant {variant!r}; expected {sorted(SUPPORTED_FUNCTIONAL_VARIANTS)}")
    model = EfficientSingleAttentionPredictor(
        input_dim, config,
        final_regressor_dropout=final_regressor_dropout,
        use_mean_proxy=use_mean_proxy,
        **FINAL_MODEL_KWARGS,
    )
    return model, "methylpredictor_final"
