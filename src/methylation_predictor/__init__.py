"""Public API for MethylPredictor.

The paper-facing model API is intentionally small.  Historical shared-backbone
classes remain lazily reachable during the repository migration, but they are
not exported through ``__all__`` and are not the reference method.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .modeling import IterativeRetrievalPredictor, SingleRetrievalPredictor

__all__ = [
    "SingleRetrievalPredictor",
    "IterativeRetrievalPredictor",
]

_LEGACY_MODEL_NAMES = {
    "FeatureFusionArchitectureVariantModel",
    "FeatureFusionLocusCLSModel",
}


def __getattr__(name: str):
    if name in __all__:
        from . import modeling

        return getattr(modeling, name)

    # Transitional compatibility only.  Phase 3b will migrate the surviving
    # paper baselines/comparators away from this model family before deletion.
    if name in _LEGACY_MODEL_NAMES:
        from . import models

        return getattr(models, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
