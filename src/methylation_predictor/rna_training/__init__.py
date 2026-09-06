"""Scope-general training/evaluation for the reference RNA-methylation model."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .locus_cls_trainer import LocusCLSJointTrainer

__all__ = ["LocusCLSJointTrainer"]


def __getattr__(name: str):
    if name == "LocusCLSJointTrainer":
        from .locus_cls_trainer import LocusCLSJointTrainer
        return LocusCLSJointTrainer
    raise AttributeError(name)
