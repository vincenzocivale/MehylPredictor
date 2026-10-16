"""Training for RNA-to-DNAm prediction."""

__all__ = ["RNAMethylationTrainer", "evaluate_rna_checkpoint"]


def __getattr__(name: str):
    if name in __all__:
        from . import rna_methylation_trainer
        return getattr(rna_methylation_trainer, name)
    raise AttributeError(name)
