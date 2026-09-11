"""Paper-facing model namespace.

During the repo-v2 migration the implementations still live in the historical
``methylation_predictor.models`` module.  The stable names exported here are the
API that the refactor will preserve while implementations are moved into
smaller modules.
"""

from .reference import IterativeRetrievalPredictor, SingleRetrievalPredictor

__all__ = [
    "SingleRetrievalPredictor",
    "IterativeRetrievalPredictor",
]
