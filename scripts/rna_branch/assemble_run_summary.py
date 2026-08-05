#!/usr/bin/env python3
"""Fase: final deliverable assembly. Reads every stage's output JSON and
writes run_summary.json plus a Markdown summary table, and appends the final
result section to docs/tcga_matrix_model.md."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--best-epoch", type=int, required=True)
    args = p.parse_args()

    run_dir = args.run_dir
    dev_metrics = _load(run_dir / "development" / "metrics.json")
    final_metrics = _load(run_dir / "final_refit" / "metrics.json")
    full_eval = _load(run_dir / "evaluation" / "full_evaluation.json")
    mp_comparison = _load(run_dir / "evaluation" / "methylprophet_official_comparison.json")
    bootstrap = _load(run_dir / "evaluation" / "hierarchical_bootstrap.json")
    efficiency = _load(run_dir / "evaluation" / "efficiency_benchmark.json")

    double_ood = (full_eval or {}).get("double_ood", {})
    summary = {
        "run_name": "concat_genomewide_fullcoverage_seed17_v1",
        "git_commit": _git_head(),
        "protocol": {
            "stage_3a_development": {
                "purpose": "early stopping / best_epoch selection, strictly inside a nested "
                           "dev split carved from official train (never touches official val)",
                "best_epoch": dev_metrics.get("best_epoch") if dev_metrics else None,
                "best_validation_mse": dev_metrics.get("best_validation_mse") if dev_metrics else None,
            },
            "stage_3b_final_refit": {
                "purpose": "fresh init, seed 17, full official train_sample x train_cpg, "
                           "exactly best_epoch full-coverage epochs, no early stopping",
                "best_epoch_used": args.best_epoch,
                "num_parameters": final_metrics.get("num_parameters") if final_metrics else None,
                "elapsed_seconds": final_metrics.get("elapsed_seconds") if final_metrics else None,
            },
            "official_test_usage": (
                "official val_sample x val_cpg (validation ∪ test labels) was not referenced by "
                "either training stage -- confirmed by construction: Stage 3A's config points at "
                "the nested dev-split manifests (dev_fit/dev_heldout labels only), Stage 3B's "
                "config points at the original manifests but only ever calls "
                "bundle.*_indices('train') for its sampling pool and 'train' again (in-sample "
                "sanity check only) for its own internal validation metric -- 'validation'/'test' "
                "labels are read for the FIRST time in Stage 5 (Fase 7 evaluation), after both "
                "training stages have fully exited and their checkpoints are frozen."
            ),
        },
        "primary_result_double_ood": {
            "samples": double_ood.get("samples"),
            "cpgs": double_ood.get("cpgs"),
            "observed_rows": double_ood.get("rows"),
            "mse": double_ood.get("mse"),
            "mae": double_ood.get("mae"),
            "skill_vs_prior": double_ood.get("skill_vs_prior"),
            "dynamic_skill": double_ood.get("dynamic_skill"),
            "dynamic_pearson": double_ood.get("dynamic_pearson"),
            "patient_dynamic_pearson_median": double_ood.get("patient_dynamic_pearson_median"),
            "locus_dynamic_pearson_median": double_ood.get("locus_dynamic_pearson_median"),
            "within_cancer_skill": double_ood.get("within_cancer_skill"),
            "dynamic_amplitude_ratio": double_ood.get("dynamic_amplitude_ratio"),
            "inference_seconds": double_ood.get("inference_seconds"),
            "peak_vram_gb": double_ood.get("peak_vram_gb"),
        },
        "methylprophet_chr1_only_comparison": {
            "note": "MethylProphet's released checkpoint/predictions cover chr1 CpGs only -- this "
                    "comparison is restricted to the measured intersection, NOT the genome-wide result",
            **({} if mp_comparison is None else mp_comparison),
        },
        "hierarchical_bootstrap_available": bootstrap is not None,
        "efficiency_benchmark_available": efficiency is not None,
        "artifacts": {
            "final_checkpoint": str(run_dir / "final_refit" / "best.pt"),
            "development_metrics": str(run_dir / "development" / "metrics.json"),
            "final_refit_metrics": str(run_dir / "final_refit" / "metrics.json"),
            "full_evaluation": str(run_dir / "evaluation" / "full_evaluation.json"),
            "methylprophet_official_comparison": str(run_dir / "evaluation" / "methylprophet_official_comparison.json"),
            "hierarchical_bootstrap": str(run_dir / "evaluation" / "hierarchical_bootstrap.json"),
            "efficiency_benchmark": str(run_dir / "evaluation" / "efficiency_benchmark.json"),
        },
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")

    lines = [
        "# concat_genomewide_fullcoverage_seed17_v1 -- run summary",
        "",
        f"Git commit: `{summary['git_commit']}`",
        "",
        "## Primary result (official double-OOD: val_sample x val_cpg, full, uncapped)",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for key, value in summary["primary_result_double_ood"].items():
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        "## Protocol",
        f"- Stage 3A best_epoch: {summary['protocol']['stage_3a_development']['best_epoch']}",
        f"- Stage 3B trained for: {summary['protocol']['stage_3b_final_refit']['best_epoch_used']} full-coverage epochs",
        f"- {summary['protocol']['official_test_usage']}",
        "",
        "## MethylProphet comparison (chr1-only, see methylprophet_official_comparison.json)",
        f"- {summary['methylprophet_chr1_only_comparison'].get('note', '')}",
        "",
    ]
    (run_dir / "run_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
