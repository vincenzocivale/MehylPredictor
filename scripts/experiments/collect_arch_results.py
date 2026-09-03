#!/usr/bin/env python3
"""Collect architecture-novelty runs (shared-backbone engine) into a single
version-controlled ledger.

Retargeted 2026-09-04 alongside run_arch_suite.py -- see arch_suite.py's module
docstring for why. Not part of the stable CLI; delete alongside the rest of the
suite once the study concludes.

Reads finished runs out of one or more output roots and writes, under
results/reference/ablations/architecture_novelty_2026_09/:

    runs/<arm>__seed<N>.json   one self-describing record per completed run
    summary.yaml               machine-readable rollup, sorted by MAS-PCC
    summary.md                 the same table, with the promotion verdict

A run counts as complete only when BOTH training/summary.json (the trainer's
own end-of-run summary) AND evaluation/chr1/metrics.json (this suite's own
evaluate.py call against the official split) exist -- unlike the retired
two-stage engine, training alone does not produce an official-split number here.

    python scripts/experiments/collect_arch_results.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from arch_suite import (  # noqa: E402
    ARMS_BY_NAME, CANONICAL_IS_CONVERGED, CANONICAL_MAS_PCC, CANONICAL_MSE, REPO_ROOT, STUDY, data_paths,
)

LEDGER = REPO_ROOT / "results" / "reference" / "ablations" / STUDY


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def collect_run(run_dir: Path) -> dict | None:
    training_summary = _read_json(run_dir / "training" / "summary.json")
    evaluation = _read_json(run_dir / "evaluation" / "chr1" / "metrics.json")
    if not training_summary or not evaluation or "metrics" not in evaluation:
        return None
    try:
        resolved_ckpt = None  # locus_cls_joint has no config.resolved.yaml; recipe lives in training config below
    except Exception:
        resolved_ckpt = None
    del resolved_ckpt

    name = run_dir.name
    prefix = f"arch-{STUDY}-shared-"
    arm_name, _, seed_part = name.removeprefix(prefix).rpartition("-seed")
    arm = ARMS_BY_NAME.get(arm_name)

    return {
        "study": STUDY,
        "engine": "matched_chr1_shared_backbone",
        "arm": arm_name,
        "stage": arm.stage if arm else None,
        "question": arm.question if arm else None,
        "control_arm": arm.control if arm else None,
        "seed": int(seed_part) if seed_part.isdigit() else None,
        "run_id": name,
        "run_dir": str(run_dir),
        "recipe": arm.recipe if arm else None,
        "architecture": evaluation.get("architecture") or evaluation.get("model"),
        "checkpoint": evaluation.get("checkpoint"),
        "checkpoint_epoch": evaluation.get("checkpoint_epoch"),
        "training": {
            "mode": training_summary.get("mode"),
            "best_epoch": training_summary.get("best_epoch"),
            "epochs_planned": training_summary.get("epochs_planned"),
            "epochs_run": training_summary.get("epochs_run"),
            "elapsed_seconds": training_summary.get("elapsed_seconds"),
        },
        "metrics": evaluation.get("metrics"),
        "architecture_diagnostics": None,  # filled below if history.json has one
    }


def _attach_diagnostics(record: dict) -> None:
    history = _read_json(Path(record["run_dir"]) / "training" / "history.json") or []
    for row in reversed(history):
        if isinstance(row, dict) and row.get("architecture_diagnostics"):
            record["architecture_diagnostics"] = row["architecture_diagnostics"]
            return


def collect(output_roots: list[str]) -> list[dict]:
    records: dict[tuple[str, int | None], dict] = {}
    for root in output_roots:
        base = Path(root) / "runs" / "locus_cls_joint" / "chr1"
        if not base.is_dir():
            print(f"[collect] no chr1 runs under {base}", file=sys.stderr)
            continue
        for run_dir in sorted(base.glob(f"arch-{STUDY}-shared-*")):
            record = collect_run(run_dir)
            if record is None:
                print(f"[collect] skipping incomplete run: {run_dir.name}", file=sys.stderr)
                continue
            _attach_diagnostics(record)
            records[(record["arm"], record["seed"])] = record
    return [records[key] for key in sorted(records, key=lambda k: (k[0], k[1] or 0))]


def _mas_pcc(record: dict) -> float | None:
    m = record.get("metrics") or {}
    return float(m["mas_pcc"]) if "mas_pcc" in m else None


def noise_floor(records: list[dict]) -> dict:
    values = [
        _mas_pcc(r) for r in records
        if r["arm"] == "seed_variance_reference" and _mas_pcc(r) is not None
    ]
    if len(values) < 2:
        return {"n_seeds": len(values), "values": values, "mean": None, "stdev": None, "threshold": None}
    stdev = statistics.stdev(values)
    return {
        "n_seeds": len(values), "values": values, "mean": statistics.mean(values), "stdev": stdev,
        "threshold": 2.0 * stdev,
    }


def verdict(delta: float | None, threshold: float | None) -> str:
    if delta is None:
        return "no result"
    if threshold is None:
        return "unjudged (run stage 0-noise-floor first)"
    if delta > threshold:
        return "PROMOTE"
    if delta < -threshold:
        return "regression"
    return "no measurable effect"


def write_ledger(records: list[dict]) -> dict:
    (LEDGER / "runs").mkdir(parents=True, exist_ok=True)
    for record in records:
        name = f"{record['arm']}__seed{record['seed']}.json"
        (LEDGER / "runs" / name).write_text(json.dumps(record, indent=2, sort_keys=False) + "\n")

    floor = noise_floor(records)
    threshold = floor["threshold"]
    baseline = floor["mean"] if floor["mean"] is not None else CANONICAL_MAS_PCC
    baseline_is_converged = floor["mean"] is not None or CANONICAL_IS_CONVERGED

    rows = []
    for record in records:
        mas = _mas_pcc(record)
        m = record.get("metrics") or {}
        rows.append({
            "arm": record["arm"], "stage": record["stage"], "seed": record["seed"],
            "official_val_cpg_x_val_sample_mas_pcc": mas,
            "official_val_cpg_x_val_sample_mse": float(m["mse"]) if "mse" in m else None,
            "delta_vs_reference": None if mas is None else round(mas - baseline, 5),
            "verdict": verdict(None if mas is None else mas - baseline, threshold),
            "control_arm": record["control_arm"],
            "architecture_diagnostics": record["architecture_diagnostics"],
            "checkpoint_epoch": record["checkpoint_epoch"],
            "run_id": record["run_id"],
        })
    rows.sort(key=lambda r: (-(r["official_val_cpg_x_val_sample_mas_pcc"] or -9.0), r["arm"]))

    summary = {
        "study": STUDY, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "engine": "matched_chr1_shared_backbone", "mode": "final",
            "split": "MethylProphet Table-5 official chr1 Array views (evaluate_official_split)",
            "headline_view": "official_val_cpg_x_val_sample",
            "budget": "80 epochs, lr 5e-5, constant",
        },
        "reference": {
            "mas_pcc": baseline, "mse": CANONICAL_MSE if floor["mean"] is None else None,
            "converged": baseline_is_converged,
            "source": (
                "results/reference/ablations.yaml::rung_b_official_final (epoch 47/80, lower bound)"
                if floor["mean"] is None
                else "this study's own seed_variance_reference (run to convergence)"
            ),
        },
        "noise_floor": floor,
        "promotion_rule": "delta vs the reference must exceed 2 x seed SD",
        "runs": rows,
    }
    (LEDGER / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))

    lines = [
        f"# {STUDY} — summary (shared-backbone engine)",
        "",
        "Generated by `scripts/experiments/collect_arch_results.py`; do not hand-edit.",
        "See `README.md` in this directory for the arm taxonomy and the 2026-09-04 retarget note.",
        "",
        "All arms: `matched_chr1_shared_backbone` engine, `mode=final`, evaluated on the official",
        "MethylProphet Table-5 chr1 Array views via `evaluate_official_split`.",
        "",
    ]
    if threshold is None:
        lines += [
            f"> **No converged noise floor yet.** `rung_b_official_final` (epoch 47/80, still improving) is a "
            f"> lower bound of {CANONICAL_MAS_PCC}, not a converged reference. Run `--stages 0-noise-floor` "
            "> (3 seeds, full 80-epoch budget) before reading any delta below as an effect.",
            "",
        ]
    else:
        lines += [
            f"**Noise floor / converged reference** ({floor['n_seeds']} seeds, full 80-epoch budget): mean "
            f"{floor['mean']:.4f}, SD {floor['stdev']:.4f}. An arm counts as an improvement only if it beats "
            f"the mean by more than 2 SD = **{threshold:.4f}**.",
            "",
        ]
    lines += [
        "| arm | stage | seed | MAS-PCC | MSE | Δ vs reference | verdict |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        mas = "—" if row["official_val_cpg_x_val_sample_mas_pcc"] is None else f"{row['official_val_cpg_x_val_sample_mas_pcc']:.4f}"
        mse = "—" if row["official_val_cpg_x_val_sample_mse"] is None else f"{row['official_val_cpg_x_val_sample_mse']:.5f}"
        delta = "—" if row["delta_vs_reference"] is None else f"{row['delta_vs_reference']:+.4f}"
        lines.append(f"| `{row['arm']}` | {row['stage']} | {row['seed']} | {mas} | {mse} | {delta} | {row['verdict']} |")
    diagnostics_rows = [r for r in rows if r["architecture_diagnostics"]]
    if diagnostics_rows:
        lines += [
            "", "## Signal-propagation diagnostics (mHC *Amax Gain Magnitude*)", "",
            "Max abs row sum (forward) and column sum (backward) of the composite residual mapping across",
            "depth. Exactly 1 for a doubly stochastic (mHC) composite; unbounded for plain HC.", "",
            "| arm | forward | backward |", "|---|---:|---:|",
        ]
        for row in diagnostics_rows:
            d = row["architecture_diagnostics"]
            lines.append(f"| `{row['arm']}` | {d.get('forward_amax_gain', float('nan')):.4f} | "
                         f"{d.get('backward_amax_gain', float('nan')):.4f} |")
    missing = [name for name in ARMS_BY_NAME if name not in {r["arm"] for r in rows}]
    if missing:
        lines += ["", "## Not yet run", "", *(f"- `{name}`" for name in missing)]
    (LEDGER / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", action="append", dest="output_roots")
    ap.add_argument("--data-root")
    args = ap.parse_args()

    roots = args.output_roots or [data_paths(args.data_root)["output_root"]]
    records = collect(roots)
    if not records:
        print("no completed runs found; nothing written")
        return 1
    summary = write_ledger(records)
    floor = summary["noise_floor"]
    print(f"[collect] {len(records)} run(s) -> {LEDGER}")
    if floor["threshold"] is not None:
        print(f"[collect] noise floor: mean {floor['mean']:.4f}, SD {floor['stdev']:.4f}, "
              f"promotion threshold {floor['threshold']:.4f}")
    else:
        print("[collect] no converged noise floor yet: run --stages 0-noise-floor first")
    for row in summary["runs"][:5]:
        print(f"  {row['arm']:<40} {row['official_val_cpg_x_val_sample_mas_pcc']}  {row['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
