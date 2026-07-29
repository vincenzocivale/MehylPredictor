#!/usr/bin/env python3
"""Combine shard-level fixed-RNA MP outputs into a verified locus prior table."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glob", required=True); p.add_argument("--manifest", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); files = sorted(glob.glob(a.glob))
    if not files: raise ValueError("glob matched no worker outputs")
    expected = pd.read_parquet(a.manifest, columns=["cpg_idx"])
    if expected.cpg_idx.duplicated().any(): raise ValueError("manifest has duplicate CpG IDs")
    raw = pd.concat([pd.read_parquet(x) for x in files], ignore_index=True)
    fixed = raw[raw.condition == "fixed_train_mean"][["cpg_idx", "pred_methyl"]].copy()
    if fixed.cpg_idx.duplicated().any(): raise ValueError("worker outputs have duplicate fixed predictions")
    joined = expected.merge(fixed, on="cpg_idx", how="left", validate="one_to_one")
    if joined.pred_methyl.isna().any(): raise ValueError(f"missing fixed predictions for {joined.pred_methyl.isna().sum()} CpGs")
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); joined.to_parquet(out, index=False)
    out.with_suffix(".json").write_text(json.dumps({"inputs": files, "manifest": a.manifest, "n_cpg": len(joined)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
