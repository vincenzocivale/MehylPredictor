#!/usr/bin/env python3
"""Compare one E0 architecture candidate against the frozen current E0 baseline.

This script is descriptive only: it does not declare a candidate a success or
promote it to the default architecture. Lower MSE/MAE and higher PCC/skill are
better. The comparison uses the already-generated three-view headline JSONs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

VIEWS = (
    "train_cpg_x_val_sample",
    "val_cpg_x_train_sample",
    "val_cpg_x_val_sample",
)
METRICS = ("mse", "mae", "mas_pcc", "mac_pcc", "skill_vs_prior")
LOWER_IS_BETTER = {"mse", "mae"}


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if "views" not in payload:
        raise ValueError(f"{path} does not contain a 'views' mapping")
    return payload


def _ours(view: dict, source: Path, name: str) -> dict:
    if "ours" not in view:
        raise ValueError(f"{source}: view {name!r} does not contain 'ours'")
    return view["ours"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--label", default="candidate")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    baseline = _load(args.baseline)
    candidate = _load(args.candidate)

    report = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "candidate_label": args.label,
        "selection_note": (
            "Descriptive comparison only. This file does not automatically "
            "promote the ablation to the canonical architecture."
        ),
        "views": {},
    }

    mse_rel_changes = []
    print(f"{'view':30s} {'metric':15s} {'baseline':>12s} {'candidate':>12s} {'delta':>12s}")
    print("-" * 88)

    for view_name in VIEWS:
        if view_name not in baseline["views"] or view_name not in candidate["views"]:
            raise ValueError(f"missing required view {view_name!r}")
        b = _ours(baseline["views"][view_name], args.baseline, view_name)
        c = _ours(candidate["views"][view_name], args.candidate, view_name)

        view_report = {}
        for metric in METRICS:
            bv = float(b[metric])
            cv = float(c[metric])
            delta = cv - bv
            item = {
                "baseline": bv,
                "candidate": cv,
                "delta_candidate_minus_baseline": delta,
                "direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
                "candidate_better": (cv < bv) if metric in LOWER_IS_BETTER else (cv > bv),
            }
            if metric in LOWER_IS_BETTER and bv != 0:
                rel = 100.0 * delta / bv
                item["relative_change_pct"] = rel
                if metric == "mse":
                    mse_rel_changes.append(rel)
            view_report[metric] = item
            print(f"{view_name:30s} {metric:15s} {bv:12.6f} {cv:12.6f} {delta:12.6f}")
        report["views"][view_name] = view_report

    report["summary"] = {
        "mse_views_improved": sum(
            int(report["views"][view]["mse"]["candidate_better"]) for view in VIEWS
        ),
        "mse_views_total": len(VIEWS),
        "mean_relative_mse_change_pct": (
            sum(mse_rel_changes) / len(mse_rel_changes) if mse_rel_changes else None
        ),
        "double_ood_mse_improved": report["views"]["val_cpg_x_val_sample"]["mse"]["candidate_better"],
        "double_ood_mas_pcc_improved": report["views"]["val_cpg_x_val_sample"]["mas_pcc"]["candidate_better"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote: {args.output}")


if __name__ == "__main__":
    main()
