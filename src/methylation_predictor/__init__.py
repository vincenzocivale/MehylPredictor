"""Public API for the final paper-facing MethylPredictor model."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .modeling import EfficientSingleAttentionPredictor

__all__ = [
    "EfficientSingleAttentionPredictor",
]


def __getattr__(name: str):
    if name in __all__:
        from . import modeling
        return getattr(modeling, name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
