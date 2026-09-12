"""Deprecated import shim for the pre-4d RNA trainer module."""

from .rna_methylation_trainer import (
    RNAMethylationTrainer,
    _retired_locus_compat,
    evaluate_rna_checkpoint,
)

LocusCLSJointTrainer = RNAMethylationTrainer
evaluate_official_split = evaluate_rna_checkpoint

__all__ = [
    "RNAMethylationTrainer",
    "LocusCLSJointTrainer",
    "evaluate_rna_checkpoint",
    "evaluate_official_split",
]
