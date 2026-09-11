#!/usr/bin/env python3
"""Collect functional-matched RNA encoder runs into a fresh result ledger."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = (
    REPO_ROOT
    / "results"
    / "reference"
    / "appendix"
    / "ablations"
    / "functional_rna_encoder_comparison_2026_09"
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from functional_rna_encoder_suite import ARMS, SEEDS, STUDY  # noqa: E402


def resolve_root(value: str | None) -> Path:
    if value:
        return Path(value).resolve()
    if os.environ.get("METHYL_DATA_ROOT"):
        return Path(os.environ["METHYL_DATA_ROOT"]).resolve()
    local = REPO_ROOT / "MethylPredictionData"
    if local.is_dir():
        return local.resolve()
    return Path("/dune/DATASETS/MethylPredictionData")


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def collect_one(base: Path, arm, seed: int, mode: str):
    rd = base / arm.run_id(seed)
    summary = load_json(rd / "training" / "summary.json")
    if not summary:
        return None

    record = {
        "arm": arm.name,
        "recipe": arm.recipe,
        "cache_kind": arm.cache_kind,
        "mode": mode,
        "seed": seed,
        "run_id": rd.name,
        "run_dir": str(rd),
        "best_epoch": summary.get("best_epoch"),
        "epochs_run": summary.get("epochs_run"),
    }

    if mode == "development":
        history = load_json(rd / "training" / "history.json") or []
        best_epoch = summary.get("best_epoch")
        row = next(
            (x for x in history if x.get("epoch") == best_epoch),
            None,
        ) or {}
        record["views"] = row.get("development")
        record["headline_mas_pcc"] = summary.get(
            "best_inner_double_ood_mas_pcc"
        )
        record["headline_mse"] = None
    else:
        ev = load_json(rd / "evaluation" / "chr1" / "metrics.json")
        if not ev:
            return None
        record["views"] = ev.get("views")
        metrics = ev.get("metrics") or {}
        record["headline_mas_pcc"] = metrics.get("mas_pcc")
        record["headline_mse"] = metrics.get("mse")

    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root")
    ap.add_argument(
        "--mode",
        choices=["development", "final"],
        default="development",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="collect one seed only; default collects 17,29,43",
    )
    args = ap.parse_args()

    root = resolve_root(args.data_root)
    base = (
        root
        / "experiments"
        / "runs"
        / "locus_cls_joint"
        / "chr1"
    )
    seeds = (args.seed,) if args.seed is not None else SEEDS

    rows = []
    for arm in ARMS:
        for seed in seeds:
            rec = collect_one(base, arm, seed, args.mode)
            if rec is not None:
                rows.append(rec)

    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "study": STUDY,
        "protocol": {
            "scope": "TCGA chr1",
            "mode": args.mode,
            "seeds": list(seeds),
            "fixed_harness": (
                "same functional CpG encoder, J0 retrieval, mean proxy, "
                "loss, optimizer and batching; only RNA encoder changes"
            ),
            "reference_recipe": "configs/models/main.yaml",
        },
        "runs": rows,
    }

    yaml_path = RESULTS / f"summary_{args.mode}.yaml"
    yaml_path.write_text(
        yaml.safe_dump(payload, sort_keys=False)
    )

    lines = [
        f"# Functional RNA encoder comparison — chr1 {args.mode}",
        "",
        "| encoder | seed | best epoch | headline MAS-PCC | headline MSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        mas = row.get("headline_mas_pcc")
        mse = row.get("headline_mse")
        mas_text = "—" if mas is None else f"{mas:.4f}"
        mse_text = "—" if mse is None else f"{mse:.5f}"
        lines.append(
            f"| `{row['arm']}` | {row['seed']} | "
            f"{row.get('best_epoch')} | {mas_text} | {mse_text} |"
        )

    md_path = RESULTS / f"summary_{args.mode}.md"
    md_path.write_text("\n".join(lines) + "\n")

    print(f"[collect] {len(rows)} run(s) -> {RESULTS}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
