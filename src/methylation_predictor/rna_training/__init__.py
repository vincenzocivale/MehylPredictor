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
    "infer_tcga_genomewide",
]


def __getattr__(name: str):
    if name in __all__:
        if name == "infer_tcga_genomewide":
            from .inference import infer_tcga_genomewide
            return infer_tcga_genomewide
        from .rna_methylation_trainer import (
            RNAMethylationTrainer,
            evaluate_rna_checkpoint,
        )
        mapping = {
            "RNAMethylationTrainer": RNAMethylationTrainer,
            "evaluate_rna_checkpoint": evaluate_rna_checkpoint,
        }
        return mapping[name]
    raise AttributeError(name)
