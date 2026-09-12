"""Training and evaluation for RNA-to-DNAm prediction."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .rna_methylation_trainer import (
        RNAMethylationTrainer,
        evaluate_rna_checkpoint,
    )

__all__ = [
    "RNAMethylationTrainer",
    "evaluate_rna_checkpoint",
]


def __getattr__(name: str):
    if name in {
        "RNAMethylationTrainer",
        "evaluate_rna_checkpoint",
        "LocusCLSJointTrainer",
        "evaluate_official_split",
    }:
        from .rna_methylation_trainer import (
            RNAMethylationTrainer,
            evaluate_rna_checkpoint,
        )
        mapping = {
            "RNAMethylationTrainer": RNAMethylationTrainer,
            "evaluate_rna_checkpoint": evaluate_rna_checkpoint,
            "LocusCLSJointTrainer": RNAMethylationTrainer,
            "evaluate_official_split": evaluate_rna_checkpoint,
        }
        return mapping[name]
    raise AttributeError(name)
