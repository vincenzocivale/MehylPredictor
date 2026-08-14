#!/usr/bin/env python3
"""Render the final Table-5 headline JSON as paper-friendly CSV/Markdown."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ORDER = [
    "train_cpg_x_val_sample",
    "val_cpg_x_train_sample",
    "val_cpg_x_val_sample",
]
LABEL = {
    "train_cpg_x_val_sample": "Train CpG - Val Sample",
    "val_cpg_x_train_sample": "Val CpG - Train Sample",
    "val_cpg_x_val_sample": "Val CpG - Val Sample",
}
METRICS = ("mas_pcc", "mac_pcc", "mse", "mae")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    raw = json.loads(Path(args.input).read_text())
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    rows = []
    for view in ORDER:
        ours = raw["ours"][view]
        mp = raw["methylprophet_table5_published"][view]
        delta = raw["delta_ours_minus_methylprophet_published"][view]
        for model, values in (("MethylPredictor", ours), ("MethylProphet published", mp)):
            rows.append({
                "view": LABEL[view], "model": model,
                **{metric: float(values[metric]) for metric in METRICS},
            })
        rows.append({
            "view": LABEL[view], "model": "Delta (OURS - MP)",
            **{metric: float(delta[metric]) for metric in METRICS},
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "table5_comparison.csv", index=False)
    (out / "table5_comparison.md").write_text(frame.to_markdown(index=False, floatfmt=".6f") + "\n")
    print(frame.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"csv={out / 'table5_comparison.csv'}")
    print(f"markdown={out / 'table5_comparison.md'}")


if __name__ == "__main__":
    main()
