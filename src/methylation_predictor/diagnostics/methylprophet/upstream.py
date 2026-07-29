"""The sole boundary to the read-only MethylProphet submodule."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


def upstream_root() -> Path:
    root = Path(__file__).resolve().parents[4] / "third_party" / "MethylProphet"
    if not (root / ".git").exists():
        raise FileNotFoundError(f"missing MethylProphet submodule: {root}")
    return root


def import_upstream() -> Path:
    """Make upstream importable without copying its source into this project."""
    root = upstream_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def assert_clean() -> str:
    """Return the pinned commit, rejecting a dirty upstream worktree."""
    root = upstream_root()
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], check=True,
                            capture_output=True, text=True).stdout
    if status:
        raise RuntimeError("MethylProphet submodule is dirty; diagnostics require read-only upstream")
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def quantize_gene_expression(vector: np.ndarray, bins: int = 51) -> np.ndarray:
    """Invoke upstream preprocessing without constructing its file-backed dataset."""
    import_upstream()
    from src.data.data_preprocessor import DataPreprocessor  # type: ignore[import-not-found]
    adapter = object.__new__(DataPreprocessor)
    adapter.num_gene_expr_bins = bins
    return DataPreprocessor.quantize_and_normalize_gene_expr(
        adapter, {"gene_expr": np.asarray(vector, dtype=np.float32).copy()}
    )["gene_expr"]
