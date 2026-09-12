"""Scope-general training/evaluation for the RNA-methylation model."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .locus_cls_trainer import RNAMethylationTrainer

__all__ = ["RNAMethylationTrainer"]


def __getattr__(name: str):
    if name in {"RNAMethylationTrainer", "LocusCLSJointTrainer"}:
        from .locus_cls_trainer import RNAMethylationTrainer
        return RNAMethylationTrainer
    raise AttributeError(name)
