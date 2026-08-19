"""MethylPredictor public model API.

Models remain lazy imports so data/protocol code can import pyarrow before torch
on cluster environments where libstdc++ load order matters.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import RNAMethylationPredictor, RNA2DNAmModel, VarianceNormalizedResidualModel

__all__ = ["RNAMethylationPredictor", "VarianceNormalizedResidualModel", "RNA2DNAmModel"]


def __getattr__(name: str):
    if name in __all__:
        from . import models
        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
