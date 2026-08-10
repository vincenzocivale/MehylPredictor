"""Full E2-E4 MethylProphet-comparable experiment suite.

This package is intentionally additive: it does not alter the legacy trainer used
by the already-running E1 benchmark.  It provides the missing mixed-source data
path and frozen NTv3 feature expansion required by E2/E3, plus a genome-wide
Array E4 path.
"""

from .feature_store import CompositeFeatureStore, SortedIndex

__all__ = ["CompositeFeatureStore", "SortedIndex"]
