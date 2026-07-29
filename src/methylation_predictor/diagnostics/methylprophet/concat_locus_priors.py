#!/usr/bin/env python3
"""Concatenate verified disjoint locus-prior tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--inputs", nargs="+", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); out = pd.concat([pd.read_parquet(x, columns=["cpg_idx", "pred_methyl"]) for x in a.inputs], ignore_index=True)
    if out.cpg_idx.duplicated().any() or out.pred_methyl.isna().any(): raise ValueError("inputs are not disjoint complete one-row-per-CpG tables")
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); out.to_parquet(a.output, index=False)


if __name__ == "__main__": main()
