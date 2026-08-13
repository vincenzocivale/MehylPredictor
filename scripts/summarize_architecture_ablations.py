#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


PRIMARY_VIEW = "val_cpg_x_val_sample"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", required=True)
    args = p.parse_args()
    root = Path(args.run_root)

    rows = []
    for path in sorted((root / "runs").glob("*/evaluation/headline.json")):
        payload = json.loads(path.read_text())
        variant = path.parents[1].name
        primary_entry = payload["views"][PRIMARY_VIEW]
        # evaluate_current_model_vs_methylprophet.py nests OURS under "ours";
        # the old mixed-suite runner wrote metrics directly in the view.
        primary = primary_entry.get("ours", primary_entry)
        rows.append({
            "variant": variant,
            "best_epoch": payload["best_epoch"],
            "mse": primary["mse"],
            "mae": primary["mae"],
            "mas_pcc": primary["mas_pcc"],
            "mac_pcc": primary["mac_pcc"],
            "skill_vs_prior": primary["skill_vs_prior"],
        })
    if not rows:
        raise SystemExit(f"no completed ablations under {root / 'runs'}")

    baseline = next((r for r in rows if r["variant"] == "baseline"), None)
    for row in rows:
        row["delta_mas_pcc_vs_baseline"] = (
            row["mas_pcc"] - baseline["mas_pcc"] if baseline is not None else float("nan")
        )
        row["delta_mse_vs_baseline"] = (
            row["mse"] - baseline["mse"] if baseline is not None else float("nan")
        )
    rows.sort(key=lambda x: x["mas_pcc"], reverse=True)

    columns = [
        "variant", "best_epoch", "mas_pcc", "delta_mas_pcc_vs_baseline",
        "mse", "delta_mse_vs_baseline", "mac_pcc", "skill_vs_prior", "mae",
    ]
    out = root / "summary.tsv"
    with out.open("w") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(str(row[c]) for c in columns) + "\n")

    print("\t".join(columns))
    for row in rows:
        print("\t".join(f"{row[c]:.6f}" if isinstance(row[c], float) else str(row[c]) for c in columns))
    print(f"\nsummary: {out}")


if __name__ == "__main__":
    main()
