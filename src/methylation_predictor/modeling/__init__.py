"""Paper-facing model API."""

from .ablation import (
    DepthResidualAblationPredictor,
    EfficientSingleAttentionPredictor,
    FunctionalGeneFFNFusionPredictor,
    GatedResidualPredictor,
)
from .baselines import FunctionalBaselinePredictor
from .factory import (
    SUPPORTED_FUNCTIONAL_VARIANTS,
    build_functional_predictor,
)
from .reference import IterativeRetrievalPredictor, SingleRetrievalPredictor
from .rna_comparison import RNAEncoderComparisonPredictor

__all__ = [
    "FunctionalBaselinePredictor",
    "SingleRetrievalPredictor",
    "IterativeRetrievalPredictor",
    "RNAEncoderComparisonPredictor",
    "DepthResidualAblationPredictor",
    "EfficientSingleAttentionPredictor",
    "GatedResidualPredictor",
    "FunctionalGeneFFNFusionPredictor",
    "SUPPORTED_FUNCTIONAL_VARIANTS",
    "build_functional_predictor",
]
