"""Stable names for the two current full-data candidate architectures.

This file intentionally contains no new model logic in phase 2a.  It creates a
paper-facing import boundary before the large implementation move.  Phase 2b
will move the implementations behind these names while keeping the historical
imports available as compatibility shims until old checkpoints are migrated.
"""

from ..models import FunctionalConcatIterativeRNAModel, FunctionalConcatMASModel

SingleRetrievalPredictor = FunctionalConcatMASModel
IterativeRetrievalPredictor = FunctionalConcatIterativeRNAModel
