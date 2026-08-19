"""Scope-general training/evaluation for the canonical RNA methylation model."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .trainer import ScopedRNATrainer

__all__ = ["ScopedRNATrainer"]


def __getattr__(name: str):
    if name == "ScopedRNATrainer":
        from .trainer import ScopedRNATrainer
        return ScopedRNATrainer
    raise AttributeError(name)
