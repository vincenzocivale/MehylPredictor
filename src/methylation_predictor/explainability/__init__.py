"""Gene-level explainability for the RNA-conditioned methylation models.

See docs/EXPLAINABILITY.md for the method note and scripts/explain.py for the
CLI entrypoint. Kept as its own subpackage (like ``benchmark/methylprophet``)
rather than folded into ``rna_training`` -- it is read-only diagnostics over a
frozen checkpoint, not part of the train/eval loop.
"""
from .integrated_gradients import GeneAttribution, integrated_gradients_rna

__all__ = ["GeneAttribution", "integrated_gradients_rna"]
