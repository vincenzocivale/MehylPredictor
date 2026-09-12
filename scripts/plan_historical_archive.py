#!/usr/bin/env python3
"""DRY-RUN-FIRST historical run slim-archive planner.

Scans historical run directories (e.g. ``experiments/runs/locus_cls_joint/**``)
under ``--data-root`` and extracts *compact* provenance -- never the heavy
checkpoint/log payloads -- from:

  * metadata.json
  * config.resolved.yaml
  * training/summary.json
  * evaluation/*/metrics.json
  * checkpoint metadata/hash (only if inexpensive: file size + optional sha256
    when --hash-checkpoints is passed; this can be slow for many large runs)

This program NEVER deletes, moves, renames, or truncates anything under
--data-root. By default it only reports. Pass --write-archive to additionally
*copy* (never move) the compact provenance bundle into
``experiments/archive/historical_run_metadata/<run>/`` under --data-root --
that is purely additive and never touches the original run directory.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SUBDIR = Path("experiments") / "archive" / "historical_run_metadata"

REQUIRED_FOR_ARCHIVEABLE = ("metadata.json",)


def _read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text())


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for base, _dirs, names in os.walk(path):
        for name in names:
            try:
                total += (Path(base) / name).stat().st_size
            except OSError:
                continue
    return total


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class RunPlan:
    original_relative_path: str
    classification: str  # ARCHIVEABLE | INCOMPLETE | MISSING_METADATA
    reasons: list[str]
    estimated_reclaimable_bytes: int
    provenance: dict[str, Any]


def discover_historical_runs(data_root: Path, storage_key: str) -> list[Path]:
    root = data_root / "experiments" / "runs" / storage_key
    if not root.is_dir():
        return []
    runs: list[Path] = []
    for scope in sorted(p for p in root.iterdir() if p.is_dir()):
        for run in sorted(p for p in scope.iterdir() if p.is_dir()):
            runs.append(run)
    return runs


def extract_provenance(run_dir: Path, *, hash_checkpoints: bool) -> dict[str, Any]:
    provenance: dict[str, Any] = {}

    metadata_path = run_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            provenance["metadata"] = _read_json(metadata_path)
        except Exception as exc:  # noqa: BLE001
            provenance["metadata_error"] = str(exc)

    config_path = run_dir / "config.resolved.yaml"
    if config_path.is_file():
        try:
            provenance["config_resolved"] = _read_yaml(config_path)
        except Exception as exc:  # noqa: BLE001
            provenance["config_resolved_error"] = str(exc)

    summary_path = run_dir / "training" / "summary.json"
    if summary_path.is_file():
        try:
            provenance["training_summary"] = _read_json(summary_path)
        except Exception as exc:  # noqa: BLE001
            provenance["training_summary_error"] = str(exc)

    evaluation_dir = run_dir / "evaluation"
    evaluations: dict[str, Any] = {}
    if evaluation_dir.is_dir():
        for metrics_path in sorted(evaluation_dir.glob("*/metrics.json")):
            scope_name = metrics_path.parent.name
            try:
                evaluations[scope_name] = _read_json(metrics_path)
            except Exception as exc:  # noqa: BLE001
                evaluations[scope_name] = {"error": str(exc)}
    if evaluations:
        provenance["evaluation"] = evaluations

    checkpoints_dir = run_dir / "checkpoints"
    checkpoint_meta: dict[str, Any] = {}
    if checkpoints_dir.is_dir():
        for ckpt in sorted(checkpoints_dir.glob("*.pt")):
            try:
                entry: dict[str, Any] = {"size_bytes": ckpt.stat().st_size}
            except OSError:
                entry = {"size_bytes": None}
            if hash_checkpoints:
                try:
                    entry["sha256"] = _sha256(ckpt)
                except OSError as exc:  # noqa: BLE001
                    entry["sha256_error"] = str(exc)
            checkpoint_meta[ckpt.name] = entry
    if checkpoint_meta:
        provenance["checkpoints"] = checkpoint_meta

    return provenance


def classify_run(
    run_dir: Path,
    provenance: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    has_metadata = "metadata" in provenance and "metadata_error" not in provenance
    has_summary = "training_summary" in provenance
    has_evaluation = bool(provenance.get("evaluation"))

    if not has_metadata:
        reasons.append("metadata.json missing or unreadable")
        return "MISSING_METADATA", reasons

    if not has_summary:
        reasons.append("training/summary.json missing")
    if not has_evaluation:
        reasons.append("no evaluation/*/metrics.json found")

    if not has_summary or not has_evaluation:
        return "INCOMPLETE", reasons

    return "ARCHIVEABLE", ["metadata, training summary, and evaluation all present"]


def plan_run(run_dir: Path, data_root: Path, *, hash_checkpoints: bool) -> RunPlan:
    provenance = extract_provenance(run_dir, hash_checkpoints=hash_checkpoints)
    classification, reasons = classify_run(run_dir, provenance)
    size = _dir_size_bytes(run_dir)
    # Compact provenance itself is negligible; nearly the whole directory
    # becomes reclaimable once slim-archived (checkpoints/logs dominate size).
    reclaimable = size if classification == "ARCHIVEABLE" else 0
    return RunPlan(
        original_relative_path=str(run_dir.resolve().relative_to(data_root.resolve())),
        classification=classification,
        reasons=reasons,
        estimated_reclaimable_bytes=reclaimable,
        provenance=provenance,
    )


def write_archive_copy(plan: RunPlan, data_root: Path) -> Path:
    """Additive-only: write the compact provenance bundle to the archive
    root. Never touches or deletes the original run directory."""
    target_dir = data_root / ARCHIVE_SUBDIR / Path(plan.original_relative_path).name
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "provenance.json"
    payload = {
        "schema_version": 1,
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_relative_path": plan.original_relative_path,
        "classification": plan.classification,
        "reasons": plan.reasons,
        "provenance": plan.provenance,
    }
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    tmp.replace(target)
    return target


def _resolve_data_root(value: str | None) -> Path:
    raw = value or os.environ.get("METHYL_DATA_ROOT")
    if not raw:
        raise ValueError("--data-root is required unless METHYL_DATA_ROOT is set")
    return Path(raw).expanduser().resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root")
    ap.add_argument(
        "--storage-key",
        default="locus_cls_joint",
        help="historical run storage key under experiments/runs/<key>/<scope>/<run>",
    )
    ap.add_argument("--hash-checkpoints", action="store_true")
    ap.add_argument("--write-archive", action="store_true", help="additive-only: also write compact provenance copies")
    ap.add_argument("--output", help="optional path to write the JSON report (outside --data-root)")
    args = ap.parse_args()

    data_root = _resolve_data_root(args.data_root)
    runs = discover_historical_runs(data_root, args.storage_key)

    plans = [plan_run(r, data_root, hash_checkpoints=args.hash_checkpoints) for r in runs]

    by_class: dict[str, int] = {}
    reclaimable_total = 0
    for plan in plans:
        by_class[plan.classification] = by_class.get(plan.classification, 0) + 1
        reclaimable_total += plan.estimated_reclaimable_bytes

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "storage_key": args.storage_key,
        "n_runs": len(plans),
        "by_classification": by_class,
        "estimated_reclaimable_bytes_total": reclaimable_total,
        "runs": [
            {
                "original_relative_path": p.original_relative_path,
                "classification": p.classification,
                "reasons": p.reasons,
                "estimated_reclaimable_bytes": p.estimated_reclaimable_bytes,
            }
            for p in plans
        ],
    }

    for p in plans:
        print(
            f"[{p.classification}] {p.original_relative_path} "
            f"reclaimable={p.estimated_reclaimable_bytes} reasons={p.reasons}"
        )
    print(
        f"TOTAL runs={len(plans)} by_class={by_class} "
        f"reclaimable_bytes={reclaimable_total}"
    )

    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str) + "\n")

    if args.write_archive:
        for plan in plans:
            if plan.classification == "ARCHIVEABLE":
                write_archive_copy(plan, data_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
