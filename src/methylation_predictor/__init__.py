# Public API for the paper-facing MethylPredictor models.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .modeling import IterativeRetrievalPredictor, SingleRetrievalPredictor

__all__ = [
    "SingleRetrievalPredictor",
    "IterativeRetrievalPredictor",
]


def __getattr__(name: str):
    if name in __all__:
        from . import modeling

        return getattr(modeling, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
