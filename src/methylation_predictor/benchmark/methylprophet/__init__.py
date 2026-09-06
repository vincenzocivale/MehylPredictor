"""TCGA chr1 MethylProphet Table-5-compatible data preparation.

The exact two-stage ``MethylProphetTrainer`` reproduction that used to live in
this package has been retired; ``prepare.py``/``cache.py``/``feature_store.py``/
``probe.py``/``protocol.py`` remain live -- they build the chr1 matched cache
consumed by the current shared-backbone engine
(``scripts/train.py --engine matched_chr1_shared_backbone``) and by
``benchmark.foundation_models``.
"""
from .protocol import Table5Protocol

__all__ = ["Table5Protocol"]
