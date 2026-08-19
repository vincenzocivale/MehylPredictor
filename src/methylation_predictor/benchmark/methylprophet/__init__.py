"""TCGA chr1 MethylProphet Table-5-compatible benchmark."""
from .protocol import Table5Protocol
from .trainer import MethylProphetTrainer

__all__ = ["Table5Protocol", "MethylProphetTrainer"]
