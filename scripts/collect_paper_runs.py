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


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    metrics = (record.get("evaluation") or {}).get("headline_metrics") or {}
    return {
        "study": record.get("study"),
        "arm": record.get("arm"),
        "model": record.get("model"),
        "scope": record.get("scope"),
        "seed": record.get("seed"),
        "run_id": record.get("run_id"),
        "git_commit": (record.get("git") or {}).get("commit"),
        "git_dirty": (record.get("git") or {}).get("dirty"),
        "recipe": (record.get("recipe") or {}).get("path"),
        "recipe_sha256": (record.get("recipe") or {}).get("sha256"),
        "checkpoint_sha256": (
            (record.get("checkpoint") or {}).get("sha256")
            if record.get("checkpoint")
            else None
        ),
        "checkpoint_epoch": (
            (record.get("checkpoint") or {}).get("epoch")
            if record.get("checkpoint")
            else None
        ),
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
    rows = [flatten_record(r) for r in records]

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    try:
        output_dir.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise SystemExit("--output-dir must live inside the repository") from exc
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl = output_dir / "runs.jsonl"
    jsonl.write_text(
        "".join(json.dumps(r, sort_keys=False) + "\n" for r in records)
    )

    csv_path = output_dir / "runs.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_from_git_commit": _git_head(),
        "scope": scope,
        "study_filter": args.study,
        "n_runs": len(records),
        "sources": sources,
        "outputs": {
            "jsonl": str(jsonl.relative_to(REPO_ROOT)),
            "csv": str(csv_path.relative_to(REPO_ROOT)),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print(
        f"[paper-collect] {len(records)} run(s) -> "
        f"{output_dir.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
