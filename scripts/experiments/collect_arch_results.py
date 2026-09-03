#!/usr/bin/env python3
"""Collect architecture-novelty runs into a single version-controlled ledger.

Not part of the stable CLI -- delete alongside the rest of the suite once the
study concludes and its verdict is folded into results/reference/ablations.yaml.

Reads finished runs out of one or more output roots (a run directory is a
self-contained record: metadata.json, config.resolved.yaml, evaluation, history)
and writes, under results/reference/ablations/architecture_novelty_2026_09/:

    runs/<arm>__seed<N>.json   one self-describing record per completed run
    summary.yaml               machine-readable rollup, sorted by headline metric
    summary.md                 the same table, with the promotion verdict

Runs from different machines merge by construction: the run id is deterministic
per (arm, seed) and each record carries the host, the git commit, the checkpoint
sha256 and the resolved config, so a record is interpretable without the machine
that produced it. Re-running the collector is idempotent -- a record is keyed by
(arm, seed) and simply overwritten.

    python scripts/experiments/collect_arch_results.py
    python scripts/experiments/collect_arch_results.py \
        --output-root /dune/.../experiments --output-root /mnt/other/experiments
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
    ARMS_BY_NAME, CANONICAL_MAS_PCC, CANONICAL_MSE, HEADLINE_VIEW, REPO_ROOT, STAGES, STUDY,
    data_paths, run_id,
)

LEDGER = REPO_ROOT / "results" / "reference" / "ablations" / STUDY
VIEWS = ("train_cpg_x_val_sample", "val_cpg_x_train_sample", "val_cpg_x_val_sample")


def _read_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def collect_run(run_dir: Path) -> dict | None:
    """Build one self-describing record, or None if the run has not finished."""
    if not (run_dir / ".done").is_file():
        return None
    metrics = _read_json(run_dir / "evaluation" / "chr1" / "metrics.json")
    if not metrics or "ours" not in metrics:
        return None
    metadata = _read_json(run_dir / "metadata.json") or {}
    manifest = _read_json(run_dir / "evaluation" / "chr1" / "manifest.json") or {}
    history = _read_json(run_dir / "training" / "history.json") or []
    try:
        resolved = yaml.safe_load((run_dir / "config.resolved.yaml").read_text()) or {}
    except Exception:
        resolved = {}

    name = run_dir.name
    arm_name, _, seed_part = name.removeprefix(f"arch-{STUDY}-").rpartition("-seed")
    arm = ARMS_BY_NAME.get(arm_name)
    training = resolved.get("training", {})

    epochs = [row for row in history if isinstance(row, dict) and "epoch" in row]
    diagnostics = next(
        (row["architecture_diagnostics"] for row in reversed(epochs) if row.get("architecture_diagnostics")),
        None,
    )
    return {
        "study": STUDY,
        "arm": arm_name,
        "stage": arm.stage if arm else None,
        "question": arm.question if arm else None,
        "control_arm": arm.control if arm else None,
        "seed": int(seed_part) if seed_part.isdigit() else None,
        "run_id": name,
        "run_dir": str(run_dir),
        # Protocol identity: these are what make the number comparable to 0.5613.
        "engine": metadata.get("engine"),
        "mode": metadata.get("mode"),
        "protocol": metrics.get("protocol"),
        "architecture": metrics.get("architecture"),
        "sources": metrics.get("sources"),
        "recipe": arm.recipe if arm else None,
        # Provenance: enough to re-run this exact unit on any machine.
        "host": metadata.get("host"),
        "git_commit": metadata.get("git_commit"),
        "created_at_utc": metadata.get("created_at_utc"),
        "checkpoint": manifest.get("checkpoint"),
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "training": {
            "epochs": training.get("epochs"),
            "learning_rate": training.get("learning_rate"),
            "scheduler": training.get("scheduler"),
            "seed": training.get("seed"),
        },
        "model_config": resolved.get("model"),
        "loss_config": resolved.get("loss"),
        "metrics": {view: metrics["ours"].get(view) for view in VIEWS if view in metrics["ours"]},
        "epochs_completed": len(epochs),
        "wall_seconds": sum(float(row.get("seconds", 0.0)) for row in epochs) or None,
        "gpu_max_memory_gb": max((float(row.get("gpu_max_memory_gb", 0.0)) for row in epochs), default=None),
        # mHC's Amax Gain Magnitude, when the arm has a multi-stream trunk. This is
        # the stability evidence and is reportable whether or not MAS-PCC moved.
        "architecture_diagnostics": diagnostics,
    }


def collect(output_roots: list[str]) -> list[dict]:
    records: dict[tuple[str, int | None], dict] = {}
    for root in output_roots:
        base = Path(root) / "runs" / "rna_methylation" / "chr1"
        if not base.is_dir():
            print(f"[collect] no chr1 runs under {base}", file=sys.stderr)
            continue
        for run_dir in sorted(base.glob(f"arch-{STUDY}-*")):
            record = collect_run(run_dir)
            if record is None:
                print(f"[collect] skipping incomplete run: {run_dir.name}", file=sys.stderr)
                continue
            records[(record["arm"], record["seed"])] = record
    return [records[key] for key in sorted(records, key=lambda k: (k[0], k[1] or 0))]


def _headline(record: dict) -> float | None:
    view = record["metrics"].get(HEADLINE_VIEW)
    return float(view["mas_pcc"]) if view and "mas_pcc" in view else None


def noise_floor(records: list[dict]) -> dict:
    """Seed SD of the unmodified canonical recipe -- the yardstick for every delta.

    Until this exists, no arm's delta can be called an effect: the suite it
    supersedes (fusion_mechanism_2026_08 and friends) reported single-seed deltas
    against an unmeasured noise level.
    """
    values = [
        _headline(r) for r in records
        if r["arm"] == "seed_variance_canonical" and _headline(r) is not None
    ]
    if len(values) < 2:
        return {"n_seeds": len(values), "values": values, "mean": None, "stdev": None, "threshold": None}
    stdev = statistics.stdev(values)
    return {
        "n_seeds": len(values),
        "values": values,
        "mean": statistics.mean(values),
        "stdev": stdev,
        # Promotion rule: an arm must beat the canonical mean by more than two
        # standard deviations of the null before it is called an improvement.
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

    rows = []
    for record in records:
        mas = _headline(record)
        view = record["metrics"].get(HEADLINE_VIEW) or {}
        rows.append({
            "arm": record["arm"],
            "stage": record["stage"],
            "seed": record["seed"],
            "val_cpg_x_val_sample_mas_pcc": mas,
            "val_cpg_x_val_sample_mse": float(view["mse"]) if "mse" in view else None,
            "delta_vs_canonical": None if mas is None else round(mas - baseline, 5),
            "verdict": verdict(None if mas is None else mas - baseline, threshold),
            "control_arm": record["control_arm"],
            "architecture_diagnostics": record["architecture_diagnostics"],
            "host": record["host"],
            "run_id": record["run_id"],
        })
    rows.sort(key=lambda r: (-(r["val_cpg_x_val_sample_mas_pcc"] or -9.0), r["arm"]))

    summary = {
        "study": STUDY,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "engine": "matched_chr1",
            "mode": "final",
            "split": "MethylProphet Table-5 official chr1 Array views",
            "headline_view": HEADLINE_VIEW,
            "budget": "80 epochs, lr 5e-5, constant",
        },
        "reference": {
            "canonical_mas_pcc": CANONICAL_MAS_PCC,
            "canonical_mse": CANONICAL_MSE,
            "source": "results/reference/rna_methylation/chr1.yaml",
        },
        "noise_floor": floor,
        "promotion_rule": "delta vs the canonical seed mean must exceed 2 x seed SD",
        "runs": rows,
    }
    (LEDGER / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))

    lines = [
        f"# {STUDY} — summary",
        "",
        "Generated by `scripts/experiments/collect_arch_results.py`; do not hand-edit.",
        "See `README.md` in this directory for the arm taxonomy and what each arm asks.",
        "",
        "All arms: matched_chr1 engine, `mode=final`, official MethylProphet Table-5 chr1",
        "Array views, 80 epochs, lr 5e-5 constant — the same protocol as the canonical",
        f"reference ({CANONICAL_MAS_PCC} MAS-PCC / {CANONICAL_MSE} MSE) and as",
        "`ablations.yaml::fusion_mechanism_2026_08`, so the numbers are directly comparable.",
        "",
    ]
    if threshold is None:
        lines += [
            "> **No noise floor yet.** Run `--stages 0-noise-floor` (3 seeds of the unmodified",
            "> canonical recipe) before reading any delta below as an effect.",
            "",
        ]
    else:
        lines += [
            f"**Noise floor** ({floor['n_seeds']} seeds of the canonical recipe): mean "
            f"{floor['mean']:.4f}, SD {floor['stdev']:.4f}. An arm counts as an improvement "
            f"only if it beats the mean by more than 2 SD = **{threshold:.4f}**.",
            "",
        ]
    lines += [
        "| arm | stage | seed | MAS-PCC | MSE | Δ vs canonical | verdict |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        mas = "—" if row["val_cpg_x_val_sample_mas_pcc"] is None else f"{row['val_cpg_x_val_sample_mas_pcc']:.4f}"
        mse = "—" if row["val_cpg_x_val_sample_mse"] is None else f"{row['val_cpg_x_val_sample_mse']:.5f}"
        delta = "—" if row["delta_vs_canonical"] is None else f"{row['delta_vs_canonical']:+.4f}"
        lines.append(
            f"| `{row['arm']}` | {row['stage']} | {row['seed']} | {mas} | {mse} | {delta} | {row['verdict']} |"
        )
    diagnostics_rows = [r for r in rows if r["architecture_diagnostics"]]
    if diagnostics_rows:
        lines += [
            "",
            "## Signal-propagation diagnostics (mHC *Amax Gain Magnitude*)",
            "",
            "Max abs row sum (forward) and column sum (backward) of the composite residual",
            "mapping across depth. Exactly 1 for a doubly stochastic (mHC) composite;",
            "unbounded for plain HC — this is the stability claim, independent of MAS-PCC.",
            "",
            "| arm | forward | backward |",
            "|---|---:|---:|",
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
    ap.add_argument("--output-root", action="append", dest="output_roots",
                    help="run output root; repeatable to merge machines (default: METHYL_DATA_ROOT's)")
    ap.add_argument("--data-root", help="overrides METHYL_DATA_ROOT when deriving the default output root")
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
        print("[collect] no noise floor yet: run --stages 0-noise-floor before judging any delta")
    for row in summary["runs"][:5]:
        print(f"  {row['arm']:<40} {row['val_cpg_x_val_sample_mas_pcc']}  {row['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
