#!/usr/bin/env python3
"""Collect the controlled Q0-Q3 development experiment into versioned results.

This is architecture-selection evidence, NOT an official MethylProphet test result.
It intentionally reads the best inner-development epoch chosen by
val_cpg_x_val_sample MAS-PCC and never evaluates the official held-out split.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY = "query_representation_2026_09"
LEDGER = REPO_ROOT / "results" / "reference" / "ablations" / STUDY
ARMS = (
    ("q0_ntv3", "ntv3"),
    ("q1_mean_only", "mean_only"),
    ("q2_hybrid_detached", "hybrid_detached"),
    ("q3_hybrid_joint", "hybrid_joint"),
)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def normalized_config(config: dict) -> dict:
    config = json.loads(json.dumps(config))
    lc = config.setdefault("locus_cls", {})
    lc.pop("query_source", None)
    dev = config.setdefault("development", {})
    dev.pop("split_seed", None)
    return config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--split-seed", type=int, default=17)
    args = ap.parse_args()

    records = []
    normalized = []
    for arm, query_source in ARMS:
        rid = f"query-representation-{arm}-seed{args.seed}"
        run_dir = Path(args.output_root) / "runs" / "locus_cls_joint" / "chr1" / rid
        summary_path = run_dir / "training" / "summary.json"
        history_path = run_dir / "training" / "history.json"
        config_path = run_dir / "config.resolved.yaml"
        best_path = run_dir / "checkpoints" / "best.pt"
        if not (summary_path.is_file() and history_path.is_file() and config_path.is_file() and best_path.is_file()):
            print(f"[collect-query] incomplete: {arm}")
            continue
        summary = read_json(summary_path)
        history = read_json(history_path)
        best_epoch = int(summary["best_epoch"])
        rows = [r for r in history if int(r.get("epoch", -1)) == best_epoch and r.get("development")]
        if len(rows) != 1:
            raise RuntimeError(f"{arm}: cannot find unique development row at best_epoch={best_epoch}")
        config = yaml.safe_load(config_path.read_text()) or {}
        actual_query = (config.get("locus_cls") or {}).get("query_source", "ntv3")
        actual_split = int((config.get("development") or {}).get("split_seed", args.seed))
        if actual_query != query_source:
            raise RuntimeError(f"{arm}: query_source provenance mismatch: {actual_query} != {query_source}")
        if actual_split != args.split_seed:
            raise RuntimeError(f"{arm}: split_seed mismatch: {actual_split} != {args.split_seed}")
        normalized.append((arm, normalized_config(config)))
        record = {
            "study": STUDY,
            "arm": arm,
            "query_source": query_source,
            "seed": args.seed,
            "development_split_seed": args.split_seed,
            "run_id": rid,
            "run_dir": str(run_dir),
            "best_epoch": best_epoch,
            "epochs_run": summary.get("epochs_run"),
            "views": rows[0]["development"],
            "checkpoint": str(best_path),
            "checkpoint_sha256": sha256(best_path),
            "resolved_config_sha256": sha256(config_path),
        }
        records.append(record)

    if not records:
        print("no completed query-representation runs found")
        return 1

    # Strict one-factor-at-a-time audit: after removing query_source and the
    # explicitly fixed split_seed, every resolved config must be identical.
    control = normalized[0][1]
    for arm, config in normalized[1:]:
        if config != control:
            raise RuntimeError(f"{arm}: resolved config differs from Q0 beyond query_source/split_seed")

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        git_commit = None

    q0 = next((r for r in records if r["arm"] == "q0_ntv3"), None)
    q0_mas = None if q0 is None else q0["views"]["val_cpg_x_val_sample"]["mas_pcc"]
    rows = []
    for r in records:
        v = r["views"]
        double = v["val_cpg_x_val_sample"]
        unseen_seen = v["val_cpg_x_train_sample"]
        seen_unseen = v["train_cpg_x_val_sample"]
        rows.append({
            "arm": r["arm"],
            "query_source": r["query_source"],
            "seed": r["seed"],
            "best_epoch": r["best_epoch"],
            "inner_double_ood_mas_pcc": float(double["mas_pcc"]),
            "inner_double_ood_mse": float(double["mse"]),
            "inner_unseen_cpg_seen_sample_mas_pcc": float(unseen_seen["mas_pcc"]),
            "inner_unseen_cpg_seen_sample_mse": float(unseen_seen["mse"]),
            "inner_seen_cpg_unseen_sample_mas_pcc": float(seen_unseen["mas_pcc"]),
            "inner_seen_cpg_unseen_sample_mse": float(seen_unseen["mse"]),
            "delta_double_ood_mas_vs_q0": None if q0_mas is None else float(double["mas_pcc"] - q0_mas),
            "checkpoint_sha256": r["checkpoint_sha256"],
            "run_id": r["run_id"],
        })
    rows.sort(key=lambda x: x["inner_double_ood_mas_pcc"], reverse=True)
    winner = rows[0]

    LEDGER.mkdir(parents=True, exist_ok=True)
    (LEDGER / "runs").mkdir(exist_ok=True)
    for r in records:
        (LEDGER / "runs" / f"{r['arm']}__seed{r['seed']}.json").write_text(json.dumps(r, indent=2) + "\n")

    payload = {
        "study": STUDY,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_collection": git_commit,
        "status": "architecture_selection_only",
        "protocol": {
            "scope": "TCGA chr1",
            "mode": "development",
            "selection_metric": "inner val_cpg_x_val_sample MAS-PCC",
            "training_seed": args.seed,
            "fixed_development_split_seed": args.split_seed,
            "recipe": "configs/models/rna_methylation_locus_attention.yaml",
            "controlled_factor": "locus-attention query_source only",
        },
        "winner": winner["arm"],
        "winner_query_source": winner["query_source"],
        "runs": rows,
        "next_step": "Retrain the selected query_source in mode=final on the full official training pool and evaluate the official split; then confirm with multiple seeds/fresh scope.",
    }
    (LEDGER / "summary.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))

    lines = [
        f"# {STUDY}", "",
        "Controlled architecture-selection experiment. **Do not report these as official test numbers.**",
        "All arms use the same final locus-attention recipe, training seed, data and fixed inner split; only the query representation changes.", "",
        "| arm | query | best epoch | double-OOD MAS-PCC | double-OOD MSE | Δ MAS vs Q0 | unseen-CpG/seen-sample MAS |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = row["delta_double_ood_mas_vs_q0"]
        lines.append(
            f"| `{row['arm']}` | `{row['query_source']}` | {row['best_epoch']} | "
            f"{row['inner_double_ood_mas_pcc']:.4f} | {row['inner_double_ood_mse']:.5f} | "
            f"{('—' if delta is None else f'{delta:+.4f}')} | {row['inner_unseen_cpg_seen_sample_mas_pcc']:.4f} |"
        )
    lines += ["", f"**Selected on inner double-OOD MAS-PCC:** `{winner['arm']}` (`{winner['query_source']}`).", "",
              "Next: one final-protocol confirmation of the selected architecture; do not choose again on the official split."]
    (LEDGER / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"[collect-query] {len(records)} run(s) -> {LEDGER}")
    print(f"[collect-query] winner: {winner['arm']} {winner['inner_double_ood_mas_pcc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
