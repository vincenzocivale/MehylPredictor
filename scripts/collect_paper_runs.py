#!/usr/bin/env python3
"""Collect standardized paper-run records into a compact results/paper ledger."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml

from methylation_predictor.artifact_uri import is_uri


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = REPO_ROOT / "configs" / "data" / "paper_chr1.yaml"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected YAML mapping")
    return payload


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def resolve_data_root(value: str | None) -> Path:
    import os

    raw = value or os.environ.get("METHYL_DATA_ROOT")
    if not raw:
        raise ValueError(
            "--data-root is required unless METHYL_DATA_ROOT is set"
        )
    return Path(raw).expanduser().resolve()


def find_records(output_root: Path, scope: str) -> list[Path]:
    runs = output_root / "runs"
    if not runs.is_dir():
        return []
    return sorted(runs.glob(f"*/{scope}/*/paper/record.json"))


def validate_record(record: dict[str, Any], path: Path) -> None:
    if record.get("schema_version") != 1:
        raise ValueError(
            f"{path}: unsupported schema_version {record.get('schema_version')!r}"
        )
    if record.get("kind") != "methylpredictor.paper_run":
        raise ValueError(f"{path}: not a methylpredictor.paper_run record")
    evaluation = record.get("evaluation") or {}
    views = evaluation.get("views") or {}
    expected = {
        "train_cpg_x_val_sample",
        "val_cpg_x_train_sample",
        "val_cpg_x_val_sample",
    }
    if set(views) != expected:
        raise ValueError(
            f"{path}: official views are not exactly {sorted(expected)}"
        )


def curate_record(record: dict[str, Any]) -> dict[str, Any]:
    """Strip machine-specific absolute local paths, keeping only portable
    ``methyl-data://`` artifact URIs and content hashes.

    This is the schema written into ``results/paper/**`` (Git-curated). The
    full record (with local absolute paths) stays only under
    ``METHYL_DATA_ROOT`` as ``paper/record.json``.
    """
    uris = record.get("artifact_uris") or {}
    evaluation = record.get("evaluation") or {}
    training = record.get("training") or {}
    checkpoint = record.get("checkpoint") or {}
    resolved_config = record.get("resolved_config") or {}
    return {
        "schema_version": record.get("schema_version"),
        "kind": record.get("kind"),
        "study": record.get("study"),
        "arm": record.get("arm"),
        "model": record.get("model"),
        "scope": record.get("scope"),
        "seed": record.get("seed"),
        "run_id": record.get("run_id"),
        "campaign": record.get("campaign"),
        "git": record.get("git"),
        "recipe": record.get("recipe"),
        "data_profile": {
            "path": (record.get("data_profile") or {}).get("path"),
            "sha256": (record.get("data_profile") or {}).get("sha256"),
        },
        "best_epoch": checkpoint.get("epoch"),
        "headline_view": evaluation.get("headline_view"),
        "headline_metrics": evaluation.get("headline_metrics"),
        "evaluation_views": evaluation.get("views"),
        "checkpoint_sha256": checkpoint.get("sha256"),
        "training_summary_sha256": training.get("sha256"),
        "resolved_config_sha256": resolved_config.get("sha256"),
        "artifact_uris": {
            "run": uris.get("run"),
            "checkpoint": uris.get("checkpoint"),
            "resolved_config": uris.get("resolved_config"),
            "training_summary": uris.get("training_summary"),
            "evaluation": uris.get("evaluation"),
        },
    }


def assert_no_forbidden_local_paths(curated: dict[str, Any], path_hint: str) -> None:
    """Curated repo results must never carry machine-specific absolute
    filesystem paths (e.g. /dune/..., /data2/..., /raid/...)."""

    def _walk(value: Any, where: str) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                _walk(v, f"{where}.{k}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                _walk(v, f"{where}[{i}]")
        elif isinstance(value, str):
            if value.startswith("/") and not is_uri(value):
                raise ValueError(
                    f"{path_hint}: forbidden absolute local path at {where}: {value!r}"
                )

    _walk(curated, "record")


def flatten_record(curated: dict[str, Any]) -> dict[str, Any]:
    metrics = curated.get("headline_metrics") or {}
    return {
        "study": curated.get("study"),
        "arm": curated.get("arm"),
        "model": curated.get("model"),
        "scope": curated.get("scope"),
        "seed": curated.get("seed"),
        "run_id": curated.get("run_id"),
        "git_commit": (curated.get("git") or {}).get("commit"),
        "git_dirty": (curated.get("git") or {}).get("dirty"),
        "recipe": (curated.get("recipe") or {}).get("path"),
        "recipe_sha256": (curated.get("recipe") or {}).get("sha256"),
        "checkpoint_sha256": curated.get("checkpoint_sha256"),
        "checkpoint_epoch": curated.get("best_epoch"),
        "mas_pcc": metrics.get("mas_pcc"),
        "mac_pcc": metrics.get("mac_pcc"),
        "mse": metrics.get("mse"),
        "mae": metrics.get("mae"),
        "prior_mse": metrics.get("prior_mse"),
        "skill_vs_prior": metrics.get("skill_vs_prior"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default=str(DEFAULT_PROFILE.relative_to(REPO_ROOT)),
    )
    ap.add_argument("--data-root")
    ap.add_argument(
        "--output-dir",
        default="results/paper",
        help="repo-relative curated output directory",
    )
    ap.add_argument(
        "--study",
        help="optional exact study filter",
    )
    args = ap.parse_args()

    profile_path = (REPO_ROOT / args.profile).resolve()
    profile = _read_yaml(profile_path)
    if profile.get("schema_version") != 1:
        raise SystemExit("unsupported paper data-profile schema")

    root = resolve_data_root(args.data_root)
    output_rel = Path(profile["paths"]["output_root"])
    output_root = (
        output_rel if output_rel.is_absolute() else root / output_rel
    )
    scope = profile["scope"]

    records = []
    sources = []
    seen = set()
    for path in find_records(output_root, scope):
        record = _read_json(path)
        validate_record(record, path)
        if args.study is not None and record.get("study") != args.study:
            continue
        key = (record.get("scope"), record.get("run_id"))
        if key in seen:
            raise ValueError(f"duplicate paper run key {key}")
        seen.add(key)
        records.append(record)
        sources.append(str(path.resolve()))

    if not records:
        raise SystemExit("no standardized paper records found")

    records.sort(
        key=lambda r: (
            str(r.get("study") or ""),
            str(r.get("arm") or ""),
            -1 if r.get("seed") is None else int(r["seed"]),
            str(r.get("run_id") or ""),
        )
    )
    curated_records = [curate_record(r) for r in records]
    for curated, source in zip(curated_records, sources):
        assert_no_forbidden_local_paths(curated, source)
    rows = [flatten_record(r) for r in curated_records]

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    try:
        output_dir.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise SystemExit("--output-dir must live inside the repository") from exc
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl = output_dir / "runs.jsonl"
    jsonl.write_text(
        "".join(json.dumps(r, sort_keys=False) + "\n" for r in curated_records)
    )

    csv_path = output_dir / "runs.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    per_run_paths = []
    for curated in curated_records:
        study = curated.get("study") or "unknown_study"
        arm = curated.get("arm") or "unknown_arm"
        seed = curated.get("seed")
        seed_name = f"seed{seed}" if seed is not None else "seed_none"
        per_run_path = output_dir / study / arm / f"{seed_name}.json"
        per_run_path.parent.mkdir(parents=True, exist_ok=True)
        per_run_path.write_text(json.dumps(curated, indent=2, sort_keys=False) + "\n")
        per_run_paths.append(str(per_run_path.relative_to(REPO_ROOT)))

    registry = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_from_git_commit": _git_head(),
        "scope": scope,
        "study_filter": args.study,
        "n_runs": len(curated_records),
        "runs": [
            {
                "study": c.get("study"),
                "arm": c.get("arm"),
                "seed": c.get("seed"),
                "run_id": c.get("run_id"),
                "path": p,
            }
            for c, p in zip(curated_records, per_run_paths)
        ],
    }
    (output_dir / "registry.json").write_text(
        json.dumps(registry, indent=2) + "\n"
    )

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_from_git_commit": _git_head(),
        "scope": scope,
        "study_filter": args.study,
        "n_runs": len(curated_records),
        "sources": sources,
        "outputs": {
            "jsonl": str(jsonl.relative_to(REPO_ROOT)),
            "csv": str(csv_path.relative_to(REPO_ROOT)),
            "registry": str((output_dir / "registry.json").relative_to(REPO_ROOT)),
            "per_run": per_run_paths,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print(
        f"[paper-collect] {len(curated_records)} run(s) -> "
        f"{output_dir.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
