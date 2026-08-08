"""Canonical RNA-to-DNAm training implementation.

`RNA2DNAmModel` is exposed lazily (PEP 562) so that importing anything else
from this package -- in particular `methylation_predictor.tcga_canonical`,
the data/protocol layer -- does not force `torch` to load first. This
matters beyond import speed: `torch`'s bundled `libstdc++` can shadow the
newer `libstdc++` symbol version `pyarrow` needs (GLIBCXX_3.4.31) if it
loads first in the process, silently breaking every parquet read for the
rest of the process. Data-layer code must not depend on the model or on
torch being importable at all.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import RNA2DNAmModel

__all__ = ["RNA2DNAmModel"]


def __getattr__(name: str):
    if name == "RNA2DNAmModel":
        from .models import RNA2DNAmModel

        return RNA2DNAmModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
