#!/usr/bin/env python3
"""Collect new RNA encoder comparator runs into results/reference."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results/reference/ablations/rna_encoder_comparison_2026_09"
ARMS = ("gene_pathway", "bulkformer_147m")


def root(value):
    if value: return Path(value).resolve()
    if os.environ.get("METHYL_DATA_ROOT"): return Path(os.environ["METHYL_DATA_ROOT"]).resolve()
    local = REPO_ROOT / "MethylPredictionData"
    return local.resolve() if local.is_dir() else Path("/dune/DATASETS/MethylPredictionData")


def load(path):
    try: return json.loads(path.read_text())
    except Exception: return None


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--data-root"); ap.add_argument("--mode", choices=["development", "final"], default="development"); ap.add_argument("--seed", type=int, default=17); args = ap.parse_args()
    base = root(args.data_root) / "experiments/runs/locus_cls_joint/chr1"; rows = []
    for arm in ARMS:
        rd = base / f"rnaenc-{arm}-{args.mode}-seed{args.seed}"; summary = load(rd / "training/summary.json")
        if not summary: continue
        rec = {"arm": arm, "mode": args.mode, "seed": args.seed, "run_id": rd.name, "run_dir": str(rd), "best_epoch": summary.get("best_epoch"), "epochs_run": summary.get("epochs_run")}
        if args.mode == "development":
            history = load(rd / "training/history.json") or []; be = summary.get("best_epoch")
            row = next((x for x in history if x.get("epoch") == be), None) or {}
            rec["views"] = row.get("development"); rec["headline_mas_pcc"] = summary.get("best_inner_double_ood_mas_pcc")
        else:
            ev = load(rd / "evaluation/chr1/metrics.json")
            if not ev: continue
            rec["views"] = ev.get("views"); rec["headline_mas_pcc"] = (ev.get("metrics") or {}).get("mas_pcc"); rec["headline_mse"] = (ev.get("metrics") or {}).get("mse")
        rows.append(rec)
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {"study": "rna_encoder_comparison_2026_09", "protocol": {"scope": "TCGA chr1", "mode": args.mode, "seed": args.seed, "fixed_harness": "mean branch ON; raw product OFF; same loss/training budget"}, "runs": rows, "reported_context": {"methylprophet_gene_pathway_table9": {"train_cpg_x_val_sample_mas_pcc": 0.5371, "val_cpg_x_train_sample_mas_pcc": 0.4194, "val_cpg_x_val_sample_mas_pcc": 0.3959}}}
    (RESULTS / f"summary_{args.mode}.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
    lines = [f"# RNA encoder comparison — {args.mode}", "", "| encoder | seed | best epoch | headline MAS-PCC | headline MSE |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        mas=r.get("headline_mas_pcc"); mse=r.get("headline_mse")
        lines.append(f"| `{r['arm']}` | {r['seed']} | {r.get('best_epoch')} | {'—' if mas is None else f'{mas:.4f}'} | {'—' if mse is None else f'{mse:.5f}'} |")
    (RESULTS / f"summary_{args.mode}.md").write_text("\n".join(lines)+"\n")
    print(f"[collect] {len(rows)} run(s) -> {RESULTS}"); return 0 if rows else 1


if __name__ == "__main__": raise SystemExit(main())
