"""Paper-facing model API."""

from .baselines import FunctionalBaselinePredictor
from .factory import (
    FINAL_MODEL_KWARGS,
    FINAL_VARIANT,
    RNA_COMPARISON_VARIANT,
    SUPPORTED_FUNCTIONAL_VARIANTS,
    build_functional_predictor,
)
from .final import EfficientSingleAttentionPredictor
from .rna_comparison import RNAEncoderComparisonPredictor

__all__ = [
    "EfficientSingleAttentionPredictor",
    "FunctionalBaselinePredictor",
    "RNAEncoderComparisonPredictor",
    "FINAL_MODEL_KWARGS",
    "FINAL_VARIANT",
    "RNA_COMPARISON_VARIANT",
    "SUPPORTED_FUNCTIONAL_VARIANTS",
    "build_functional_predictor",
]
