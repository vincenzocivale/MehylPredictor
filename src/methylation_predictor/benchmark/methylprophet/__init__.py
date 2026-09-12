"""TCGA chr1 MethylProphet Table-5-compatible data preparation.

The exact two-stage ``MethylProphetTrainer`` reproduction that used to live in
this package has been retired, along with the shared-backbone engine and its
``--engine`` CLI flag; ``prepare.py``/``cache.py``/``feature_store.py``/
``probe.py``/``protocol.py`` remain live -- they build the chr1 matched cache
consumed by ``RNAMethylationTrainer`` (``scripts/train.py --model
rna_methylation``) and by ``benchmark.foundation_models``.
"""
from .protocol import Table5Protocol

__all__ = ["Table5Protocol"]
