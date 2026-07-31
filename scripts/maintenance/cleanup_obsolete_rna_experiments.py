#!/usr/bin/env python3
"""Remove superseded RNA representation-search files conservatively.

The default invocation is a dry run.  Use ``--apply`` after reviewing the
printed paths.  Current BulkRNABert extraction/audit code, R5.2 provenance, and
core RNA-branch training code are deliberately preserved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OBSOLETE_PATHS = (
    "configs/rna_branch/stage_t_gene_token_first_tranche_grid.yaml",
    "docs/rna_representation_execution.md",
    "docs/rna_representation_experiments.md",
    "docs/rna_representation_results.md",
    "docs/stage_t_gene_token_execution.md",
    "scripts/rna_branch/build_gene_module_matrix.py",
    "scripts/rna_branch/prepare_hallmark_modules.py",
    "scripts/rna_branch/run_representation_experiments.py",
    "src/methylation_predictor/rna_branch/extract_ntv3_gene_embeddings.py",
    "src/methylation_predictor/rna_branch/merge_ntv3_gene_embeddings.py",
    "src/methylation_predictor/rna_branch/prepare_gene_manifest.py",
    "tests/test_gene_token_stage_t.py",
)

HISTORICAL_GRID_PATHS = (
    "configs/rna_branch/signal_grid.yaml",
    "configs/rna_branch/stage_b_base.yaml",
    "configs/rna_branch/stage_b_baseline_grid.yaml",
    "configs/rna_branch/stage_b_bottleneck_grid.yaml",
    "configs/rna_branch/stage_b_pca_grid.yaml",
    "configs/rna_branch/stage_b_pca_mlp_grid.yaml",
    "configs/rna_branch/stage_b_random_projection_grid.yaml",
    "configs/rna_branch/stage_c_base.yaml",
    "configs/rna_branch/stage_c_c3_grid.yaml",
    "configs/rna_branch/stage_c_phase0_grid.yaml",
    "configs/rna_branch/stage_c_phase1_grid.yaml",
)


def repository_root(start: Path) -> Path:
    for path in (start.resolve(), *start.resolve().parents):
        if (path / ".git").exists() and (path / "src").exists():
            return path
    raise RuntimeError("could not locate the repository root")


def cleanup(root: Path, apply: bool, include_historical_grids: bool) -> dict[str, object]:
    paths = list(OBSOLETE_PATHS)
    if include_historical_grids:
        paths.extend(HISTORICAL_GRID_PATHS)
    existing: list[str] = []
    absent: list[str] = []
    removed: list[str] = []
    for relative in paths:
        target = root / relative
        if not target.exists():
            absent.append(relative)
            continue
        if target.is_dir():
            raise ValueError(f"refusing to remove directory: {relative}")
        existing.append(relative)
        if apply:
            target.unlink()
            removed.append(relative)
    return {
        "root": str(root),
        "mode": "apply" if apply else "dry-run",
        "include_historical_grids": include_historical_grids,
        "existing": existing,
        "absent": absent,
        "removed": removed,
        "preserved": [
            "docs/rna_r5_bulkrnabert_results.md",
            "scripts/rna_branch/extract_bulkrnabert_torch.py",
            "scripts/rna_branch/make_representation_configs.py",
            "scripts/rna_branch/audit_bulkrnabert_inputs.py",
            "scripts/rna_branch/audit_bulkrnabert_gene_overlap.py",
            "src/methylation_predictor/rna_branch/models.py",
            "src/methylation_predictor/rna_branch/trainer.py",
            "tests/test_rna_representation_execution.py",
            "tests/test_rna_representation_extensions.py",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--include-historical-grids",
        action="store_true",
        help="also remove old Stage B/C search grids; omitted by default because they predate R0-R5",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = repository_root(args.root)
    result = cleanup(root, args.apply, args.include_historical_grids)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
