#!/usr/bin/env python3
"""Collect mean_contribution_2026_09 into version-controlled paper results."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mean_contribution_suite import ARMS, REPO_ROOT, SEEDS, STUDY, data_paths  # noqa: E402

LEDGER = REPO_ROOT / "results" / "reference" / STUDY
LOWER_BETTER = {"mse", "mae", "prior_mse"}


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def _run_dir(root: str, arm, seed: int) -> Path:
    return Path(root) / "runs" / "locus_cls_joint" / "chr1" / arm.run_id(seed)


def collect_one(root: str, arm, seed: int) -> dict | None:
    run_dir = _run_dir(root, arm, seed)
    training = _read_json(run_dir / "training" / "summary.json")
    evaluation = _read_json(run_dir / "evaluation" / "chr1" / "metrics.json")
    diagnostics = _read_json(run_dir / "evaluation" / "chr1" / "mean_diagnostics.json")
    if not training or not evaluation or not diagnostics:
        return None
    checkpoint = run_dir / "checkpoints" / "best.pt"
    resolved = run_dir / "config.resolved.yaml"
    recipe = REPO_ROOT / arm.recipe
    return {
        "study": STUDY,
        "arm": arm.name,
        "question": arm.question,
        "seed": seed,
        "run_id": arm.run_id(seed),
        "run_dir": str(run_dir),
        "recipe": arm.recipe,
        "recipe_sha256": _sha256(recipe),
        "resolved_config_sha256": _sha256(resolved),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_epoch": evaluation.get("checkpoint_epoch"),
        "training": training,
        "official_views": evaluation.get("views"),
        "headline_metrics": evaluation.get("metrics"),
        "diagnostics": diagnostics,
    }


def _mean_sd(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "stdev": statistics.stdev(values) if len(values) >= 2 else None,
        "values": values,
    }


def _metric(records: list[dict], arm: str, view: str, metric: str) -> dict:
    values = []
    for record in records:
        if record["arm"] != arm:
            continue
        metrics = (record.get("official_views") or {}).get(view) or {}
        if metric in metrics:
            values.append(float(metrics[metric]))
    return _mean_sd(values)


def _paired_benefit(records: list[dict], a: str, b: str, view: str, metric: str) -> dict:
    by = {(r["arm"], r["seed"]): r for r in records}
    values = []
    for seed in SEEDS:
        ra, rb = by.get((a, seed)), by.get((b, seed))
        if not ra or not rb:
            continue
        va = float(ra["official_views"][view][metric])
        vb = float(rb["official_views"][view][metric])
        values.append((vb - va) if metric in LOWER_BETTER else (va - vb))
    out = _mean_sd(values)
    out["definition"] = "positive means first arm is better; lower-better metrics are sign-flipped"
    return out


def _get_path(record: dict, path: tuple[str, ...]) -> float:
    node = record
    for key in path:
        node = node[key]
    return float(node)


def _paired_relative_reduction(
    records: list[dict],
    a: str,
    b: str,
    path: tuple[str, ...],
) -> dict:
    by = {(r["arm"], r["seed"]): r for r in records}
    values = []
    for seed in SEEDS:
        ra, rb = by.get((a, seed)), by.get((b, seed))
        if not ra or not rb:
            continue
        va, vb = _get_path(ra, path), _get_path(rb, path)
        if vb != 0:
            values.append(100.0 * (vb - va) / vb)
    out = _mean_sd(values)
    out["definition"] = "percent reduction achieved by first arm relative to second; positive is better"
    return out




def _decile_map(record: dict, view: str) -> dict[int, dict]:
    rows = (((record.get("diagnostics") or {}).get("views") or {}).get(view) or {}).get("variance_deciles") or []
    return {int(row["decile"]): row for row in rows}


def _paired_decile_reduction(records: list[dict], comparator: str, view: str) -> list[dict]:
    by = {(r["arm"], r["seed"]): r for r in records}
    out = []
    for decile in range(1, 11):
        mse_values, bias_values, variance_values = [], [], []
        for seed in SEEDS:
            full = by.get(("full_reference", seed))
            comp = by.get((comparator, seed))
            if not full or not comp:
                continue
            frow = _decile_map(full, view).get(decile)
            crow = _decile_map(comp, view).get(decile)
            if not frow or not crow:
                continue
            if crow["mse_mean"] != 0:
                mse_values.append(100.0 * (crow["mse_mean"] - frow["mse_mean"]) / crow["mse_mean"])
            if crow["abs_bias_mean"] != 0:
                bias_values.append(100.0 * (crow["abs_bias_mean"] - frow["abs_bias_mean"]) / crow["abs_bias_mean"])
            variance_values.append(float(frow["variance_median"]))
        out.append({
            "decile": decile,
            "variance_median": _mean_sd(variance_values),
            "mse_reduction_pct": _mean_sd(mse_values),
            "abs_bias_reduction_pct": _mean_sd(bias_values),
        })
    return out


def _format_mean_sd(stats: dict, digits: int) -> str:
    if stats["mean"] is None:
        return "—"
    if stats["stdev"] is None:
        return f"{stats['mean']:.{digits}f}"
    return f"{stats['mean']:.{digits}f} ± {stats['stdev']:.{digits}f}"


def write_outputs(records: list[dict], dataset_diag: dict | None) -> None:
    LEDGER.mkdir(parents=True, exist_ok=True)
    (LEDGER / "runs").mkdir(exist_ok=True)
    for record in records:
        path = LEDGER / "runs" / f"{record['arm']}__seed{record['seed']}.json"
        path.write_text(json.dumps(record, indent=2) + "\n")
    if dataset_diag:
        (LEDGER / "dataset_variance.json").write_text(json.dumps(dataset_diag, indent=2) + "\n")

    views = ("train_cpg_x_val_sample", "val_cpg_x_train_sample", "val_cpg_x_val_sample")
    arm_summary = {}
    for arm in ARMS:
        arm_summary[arm.name] = {
            view: {
                metric: _metric(records, arm.name, view, metric)
                for metric in ("mas_pcc", "mac_pcc", "mse", "mae", "skill_vs_prior")
            }
            for view in views
        }

    effects = {
        "proxy_supervision_full_vs_no_mean_supervision": {},
        "branch_capacity_no_supervision_vs_no_mean_branch": {},
        "total_mean_contribution_full_vs_no_mean_branch": {},
    }
    for view in views:
        effects["proxy_supervision_full_vs_no_mean_supervision"][view] = {
            "mas_pcc_benefit": _paired_benefit(records, "full_reference", "no_mean_supervision", view, "mas_pcc"),
            "mac_pcc_benefit": _paired_benefit(records, "full_reference", "no_mean_supervision", view, "mac_pcc"),
            "mse_benefit": _paired_benefit(records, "full_reference", "no_mean_supervision", view, "mse"),
            "mse_reduction_pct": _paired_relative_reduction(
                records,
                "full_reference",
                "no_mean_supervision",
                ("official_views", view, "mse"),
            ),
        }
        effects["branch_capacity_no_supervision_vs_no_mean_branch"][view] = {
            "mas_pcc_benefit": _paired_benefit(records, "no_mean_supervision", "no_mean_branch", view, "mas_pcc"),
            "mse_benefit": _paired_benefit(records, "no_mean_supervision", "no_mean_branch", view, "mse"),
            "mse_reduction_pct": _paired_relative_reduction(
                records,
                "no_mean_supervision",
                "no_mean_branch",
                ("official_views", view, "mse"),
            ),
        }
        effects["total_mean_contribution_full_vs_no_mean_branch"][view] = {
            "mas_pcc_benefit": _paired_benefit(records, "full_reference", "no_mean_branch", view, "mas_pcc"),
            "mse_benefit": _paired_benefit(records, "full_reference", "no_mean_branch", view, "mse"),
            "mse_reduction_pct": _paired_relative_reduction(
                records,
                "full_reference",
                "no_mean_branch",
                ("official_views", view, "mse"),
            ),
        }

    bias_effects = {}
    for view in ("val_cpg_x_train_sample", "val_cpg_x_val_sample"):
        bias_effects[view] = {}
        for comp in ("no_mean_supervision", "no_mean_branch"):
            bias_effects[view][comp] = _paired_relative_reduction(
                records,
                "full_reference",
                comp,
                ("diagnostics", "views", view, "locus_bias", "median_abs_bias"),
            )

    variance_decile_effects = {
        view: {
            comparator: _paired_decile_reduction(records, comparator, view)
            for comparator in ("no_mean_supervision", "no_mean_branch")
        }
        for view in ("val_cpg_x_train_sample", "val_cpg_x_val_sample")
    }

    probe = {}
    for arm_name in ("full_reference", "no_mean_supervision"):
        pearsons, r2s = [], []
        for record in records:
            if record["arm"] != arm_name:
                continue
            probe_row = record["diagnostics"].get("h_mean_linear_probe_train_cpg_to_val_cpg") or {}
            metrics = probe_row.get("metrics") or {}
            if metrics.get("pearson") is not None:
                pearsons.append(float(metrics["pearson"]))
            if metrics.get("r2") is not None:
                r2s.append(float(metrics["r2"]))
        probe[arm_name] = {"pearson": _mean_sd(pearsons), "r2": _mean_sd(r2s)}

    summary = {
        "study": STUDY,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "collector_git_head": _git_head(),
        "protocol": {
            "scope": "TCGA chr1",
            "split": "verified MethylProphet Table-5 official split",
            "seeds": list(SEEDS),
            "reference_recipe": "configs/models/rna_methylation_locus_attention.yaml",
            "design": {
                "full_reference": "mean branch ON, aux_weight=0.15",
                "no_mean_supervision": "mean branch ON, aux_weight=0.0",
                "no_mean_branch": "mean branch OFF, aux_weight=0.0",
            },
            "primary_mean_claim_metric": "MSE and locus-level bias on views with official val CpGs",
            "secondary_metric": "MAS-PCC: translation-insensitive within a CpG",
        },
        "completed_runs": len(records),
        "arm_summary": arm_summary,
        "paired_effects": effects,
        "unseen_cpg_locus_bias_reduction": bias_effects,
        "variance_decile_effects": variance_decile_effects,
        "h_mean_linear_probe": probe,
        "dataset_variance": dataset_diag,
    }
    (LEDGER / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "arm", "seed", "view", "mas_pcc", "mac_pcc", "mse", "mae",
        "skill_vs_prior", "median_abs_locus_bias", "checkpoint_epoch", "run_id",
    ])
    for record in sorted(records, key=lambda x: (x["arm"], x["seed"])):
        for view in views:
            metrics = record["official_views"][view]
            bias = ((record["diagnostics"].get("views") or {}).get(view) or {}).get("locus_bias", {})
            writer.writerow([
                record["arm"], record["seed"], view,
                metrics.get("mas_pcc"), metrics.get("mac_pcc"), metrics.get("mse"),
                metrics.get("mae"), metrics.get("skill_vs_prior"), bias.get("median_abs_bias"),
                record.get("checkpoint_epoch"), record["run_id"],
            ])
    (LEDGER / "paper_table.csv").write_text(buf.getvalue())

    decile_buf = io.StringIO()
    decile_writer = csv.writer(decile_buf)
    decile_writer.writerow([
        "view", "comparator", "decile", "target_variance_median",
        "mse_reduction_pct_full_vs_comparator", "abs_bias_reduction_pct_full_vs_comparator",
    ])
    for view, by_comp in variance_decile_effects.items():
        for comparator, rows in by_comp.items():
            for row in rows:
                decile_writer.writerow([
                    view,
                    comparator,
                    row["decile"],
                    row["variance_median"]["mean"],
                    row["mse_reduction_pct"]["mean"],
                    row["abs_bias_reduction_pct"]["mean"],
                ])
    (LEDGER / "variance_decile_effects.csv").write_text(decile_buf.getvalue())

    lines = [
        f"# {STUDY} — paper summary",
        "",
        "Generated by `scripts/experiments/collect_mean_contribution.py`; do not hand-edit generated tables.",
        "",
        "## Experimental isolation",
        "",
        "- **Full**: locus-attention K=64, mean branch present, `aux_weight=0.15`.",
        "- **No mean supervision**: identical architecture, `aux_weight=0`.",
        "- **No mean branch**: same RNA encoder/training recipe, CpG-only mean branch removed.",
        "- Paired seeds: 17, 29, 43.",
        "",
        "Primary evidence: MSE and locus-level bias on unseen CpGs. MAS-PCC is retained as a secondary endpoint.",
        "",
        "## Official double-OOD view",
        "",
        "| arm | MAS-PCC mean±SD | MAC-PCC mean±SD | MSE mean±SD |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = arm_summary[arm.name]["val_cpg_x_val_sample"]
        lines.append(
            f"| `{arm.name}` | {_format_mean_sd(row['mas_pcc'], 4)} | "
            f"{_format_mean_sd(row['mac_pcc'], 4)} | {_format_mean_sd(row['mse'], 5)} |"
        )

    lines += ["", "## Paired effect of biological mean supervision", ""]
    for view in ("val_cpg_x_train_sample", "val_cpg_x_val_sample"):
        mse_pct = effects["proxy_supervision_full_vs_no_mean_supervision"][view]["mse_reduction_pct"]["mean"]
        bias_pct = bias_effects[view]["no_mean_supervision"]["mean"]
        mse_text = "—" if mse_pct is None else f"{mse_pct:.2f}%"
        bias_text = "—" if bias_pct is None else f"{bias_pct:.2f}%"
        lines.append(f"- `{view}`: MSE reduction = {mse_text}; median |locus bias| reduction = {bias_text}.")

    lines += ["", "## Representation test", ""]
    for arm_name in ("full_reference", "no_mean_supervision"):
        pearson = probe[arm_name]["pearson"]["mean"]
        ptext = "—" if pearson is None else f"{pearson:.4f}"
        lines.append(f"- `{arm_name}` h_mean linear-probe Pearson on official val CpGs: {ptext}.")

    if dataset_diag:
        d = dataset_diag["train_cpg_x_train_sample"]
        lines += [
            "",
            "## Dataset variance decomposition",
            "",
            f"Between-locus mean differences account for **{100*d['fraction_between_locus']:.2f}%** "
            f"of total observed-cell variance in the official training matrix; within-locus patient "
            f"variation accounts for **{100*d['fraction_within_locus']:.2f}%**.",
        ]

    lines += [
        "",
        "## Files",
        "",
        "- `paper_table.csv`: flat table for manuscript/table scripts.",
        "- `variance_decile_effects.csv`: mean-sensitive effect size from low- to high-variance CpGs.",
        "- `summary.yaml`: aggregate metrics and paired effects.",
        "- `runs/*.json`: per-run checkpoint/config hashes and diagnostics.",
        "- `dataset_variance.json`: dataset-only variance decomposition, when computed.",
    ]
    (LEDGER / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", action="append", dest="output_roots")
    ap.add_argument("--data-root")
    ap.add_argument(
        "--dataset-diagnostics",
        help="dataset_variance.json produced by analyze_mean_contribution.py dataset",
    )
    args = ap.parse_args()
    roots = args.output_roots or [data_paths(args.data_root)["output_root"]]

    records = []
    for arm in ARMS:
        for seed in SEEDS:
            found = None
            for output_root in roots:
                found = collect_one(output_root, arm, seed)
                if found:
                    break
            if found:
                records.append(found)
            else:
                print(f"[collect-mean] incomplete/missing: {arm.name} seed={seed}", file=sys.stderr)

    if not records:
        print("no complete mean-contribution runs found")
        return 1
    dataset_diag = _read_json(Path(args.dataset_diagnostics)) if args.dataset_diagnostics else None
    write_outputs(records, dataset_diag)
    print(f"[collect-mean] {len(records)} complete run(s) -> {LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
