#!/usr/bin/env python3
"""Materialise the locked Experiment-2 block split as CpG ID manifests."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from methylation_predictor.genomic_encoder.static_prior import _partition


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--block-size", type=int, default=5_000_000); p.add_argument("--seed", type=int, default=20260728)
    a = p.parse_args(); frame = pd.read_parquet(a.input)
    split = _partition(frame, "block", "cpg_idx", "chromosome", "position", a.block_size, a.seed)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    for name in ("train", "validation", "test"):
        frame.loc[split == name, ["cpg_idx"]].sort_values("cpg_idx").to_parquet(out / f"{name}_cpg_manifest.parquet", index=False)


if __name__ == "__main__":
    main()
