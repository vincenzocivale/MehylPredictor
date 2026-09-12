#!/usr/bin/env python3
"""Read-only dependency inventory for the frozen paper study surface.

The script never deletes, moves, renames, truncates, or modifies anything
under --data-root. Reports must be written outside METHYL_DATA_ROOT.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPENDENCIES = REPO_ROOT / "configs/data/paper_dependencies_v2.yaml"
DEFAULT_CONTRACT = REPO_ROOT / "configs/data/storage_v2.yaml"

ARTIFACT_CLASSES = {
    "REQUIRED_RUNTIME",
    "REQUIRED_REPRODUCIBILITY",
    "UNREFERENCED_CLEANUP_TARGET",
    "REVIEW",
}


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected YAML mapping")
    return payload


def related(a: str, b: str) -> bool:
    pa, pb = Path(a), Path(b)
    return pa == pb or pa in pb.parents or pb in pa.parents


def du_size(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        out = subprocess.check_output(
            ["du", "-sb", "--", str(path)],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return int(out.split()[0])
    except Exception:
        return None


def resolve_arm_dependencies(
    registry: dict[str, Any],
    study: str,
    arm: str,
) -> dict[str, list[str]]:
    spec = registry["studies"][study]["arms"][arm]

    external: list[str] = []
    for set_name in spec.get("dependency_sets") or []:
        external.extend(registry["dependency_sets"][set_name])
    external.extend(spec.get("external_inputs") or [])

    repo = list(spec.get("repo_inputs") or [])

    # Stable unique order.
    external = list(dict.fromkeys(external))
    repo = list(dict.fromkeys(repo))
    return {"external": external, "repo": repo}


def required_files_for(
    input_spec: dict[str, Any],
    consumer_contract: str | None,
) -> list[str]:
    files = list(input_spec.get("required_files") or [])
    if consumer_contract:
        extra = (
            (input_spec.get("consumer_extra_files") or {})
            .get(consumer_contract)
            or {}
        )
        files.extend(extra.get("files") or [])
    return list(dict.fromkeys(files))


def inspect_dependency(
    *,
    name: str,
    spec: dict[str, Any],
    root: Path,
    consumer_contract: str | None = None,
) -> dict[str, Any]:
    path = root / spec["path"]
    expected_kind = spec.get("kind", "directory")
    exists = path.exists()
    kind_ok = (
        exists
        and (
            (expected_kind == "file" and path.is_file())
            or (expected_kind == "directory" and path.is_dir())
        )
    )

    required_files = required_files_for(spec, consumer_contract)
    missing_files = [
        rel for rel in required_files if not (path / rel).is_file()
    ] if exists and expected_kind == "directory" else list(required_files)

    valid = exists and kind_ok and not missing_files
    return {
        "name": name,
        "path": spec["path"],
        "retention": spec.get("retention"),
        "roles": list(spec.get("roles") or []),
        "expected_kind": expected_kind,
        "exists": exists,
        "kind_ok": kind_ok,
        "required_files": required_files,
        "missing_files": missing_files,
        "valid": valid,
    }


def build_arm_matrix(
    registry: dict[str, Any],
    data_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    repo_rows: list[dict[str, Any]] = []

    for study, study_spec in registry["studies"].items():
        for arm, arm_spec in study_spec["arms"].items():
            deps = resolve_arm_dependencies(registry, study, arm)
            consumer = arm_spec.get("consumer_contract")

            for name in deps["external"]:
                spec = registry["external_inputs"][name]
                status = inspect_dependency(
                    name=name,
                    spec=spec,
                    root=data_root,
                    consumer_contract=consumer,
                )
                rows.append({
                    "study": study,
                    "arm": arm,
                    "recipe": arm_spec["recipe"],
                    "seeds": list(arm_spec.get("seeds") or []),
                    "dependency": name,
                    **status,
                })

            for name in deps["repo"]:
                spec = registry["repo_inputs"][name]
                path = REPO_ROOT / spec["path"]
                repo_rows.append({
                    "study": study,
                    "arm": arm,
                    "recipe": arm_spec["recipe"],
                    "dependency": name,
                    "path": spec["path"],
                    "exists": path.exists(),
                    "valid": (
                        path.is_file()
                        if spec.get("kind") == "file"
                        else path.is_dir()
                    ),
                })

    return rows, repo_rows


def discover_top_level_artifacts(data_root: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []

    derived = data_root / "derived"
    if derived.is_dir():
        rows.extend(
            (str(p.relative_to(data_root)), "derived")
            for p in sorted(derived.iterdir())
        )

    datasets = data_root / "datasets"
    if datasets.is_dir():
        rows.extend(
            (str(p.relative_to(data_root)), "dataset")
            for p in sorted(datasets.iterdir())
        )

    reference = data_root / "reference"
    if reference.is_dir():
        rows.extend(
            (str(p.relative_to(data_root)), "reference")
            for p in sorted(reference.iterdir())
        )

    weights = data_root / "model_weights_cache"
    if weights.is_dir():
        rows.extend(
            (str(p.relative_to(data_root)), "model_weights_cache")
            for p in sorted(weights.iterdir())
        )

    return rows


def classify_artifact(
    path: str,
    registry: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[str, list[str]]:
    runtime_hits = []
    repro_hits = []

    for name, spec in registry["external_inputs"].items():
        dep_path = str(spec["path"])
        if not related(path, dep_path):
            continue
        if spec.get("retention") == "runtime":
            runtime_hits.append(name)
        elif spec.get("retention") == "reproducibility":
            repro_hits.append(name)

    if runtime_hits:
        return (
            "REQUIRED_RUNTIME",
            [f"paper dependency: {name}" for name in sorted(runtime_hits)],
        )
    if repro_hits:
        return (
            "REQUIRED_REPRODUCIBILITY",
            [
                f"reproducibility dependency: {name}"
                for name in sorted(repro_hits)
            ],
        )

    cleanup_targets = (
        contract.get("cleanup_target", {})
        .get("review_then_delete_if_unreferenced", [])
    )
    if any(related(path, target) for target in cleanup_targets):
        return (
            "UNREFERENCED_CLEANUP_TARGET",
            ["listed by Data/Results V2 for cleanup review"],
        )

    return "REVIEW", ["not referenced by frozen paper dependency matrix"]


def build_inventory(
    *,
    data_root: Path,
    registry_path: Path,
    contract_path: Path,
    include_sizes: bool,
) -> dict[str, Any]:
    registry = read_yaml(registry_path)
    contract = read_yaml(contract_path)

    if registry.get("schema_version") != 1:
        raise ValueError("unsupported dependency registry schema")
    if contract.get("status") != "frozen":
        raise ValueError("Data/Results V2 contract must be frozen")

    matrix, repo_matrix = build_arm_matrix(registry, data_root)

    external_status: dict[str, dict[str, Any]] = {}
    for name, spec in registry["external_inputs"].items():
        external_status[name] = inspect_dependency(
            name=name,
            spec=spec,
            root=data_root,
        )

    # Add consumer-specific CpG-prior validation separately because the same
    # prior cache has a stricter current file contract for that baseline.
    prior_spec = registry["external_inputs"]["prior_cache"]
    external_status["prior_cache__cpg_prior_contract"] = inspect_dependency(
        name="prior_cache",
        spec=prior_spec,
        root=data_root,
        consumer_contract="cpg_prior",
    )

    artifacts = []
    for rel, kind in discover_top_level_artifacts(data_root):
        classification, reasons = classify_artifact(
            rel,
            registry,
            contract,
        )
        full = data_root / rel
        artifacts.append({
            "path": rel,
            "kind": kind,
            "classification": classification,
            "reasons": reasons,
            "size_bytes": du_size(full) if include_sizes else None,
        })

    counts = {key: 0 for key in sorted(ARTIFACT_CLASSES)}
    bytes_by_class = {key: 0 for key in sorted(ARTIFACT_CLASSES)}
    for row in artifacts:
        counts[row["classification"]] += 1
        if row["size_bytes"] is not None:
            bytes_by_class[row["classification"]] += row["size_bytes"]

    missing_runtime = sorted(
        name
        for name, status in external_status.items()
        if status.get("retention") == "runtime"
        and not status["valid"]
    )
    missing_repro = sorted(
        name
        for name, status in external_status.items()
        if status.get("retention") == "reproducibility"
        and not status["valid"]
    )
    missing_repo = sorted(
        {
            row["dependency"]
            for row in repo_matrix
            if not row["valid"]
        }
    )

    return {
        "schema_version": 1,
        "kind": "methylpredictor.paper_dependency_inventory",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root.resolve()),
        "safety": {
            "read_only": True,
            "destructive_actions_allowed": False,
        },
        "dependency_registry": str(
            registry_path.relative_to(REPO_ROOT)
        ),
        "contract": str(contract_path.relative_to(REPO_ROOT)),
        "external_inputs": external_status,
        "arm_matrix": matrix,
        "repo_input_matrix": repo_matrix,
        "artifact_inventory": artifacts,
        "summary": {
            "study_count": len(registry["studies"]),
            "arm_count": sum(
                len(s["arms"]) for s in registry["studies"].values()
            ),
            "external_input_count": len(registry["external_inputs"]),
            "missing_runtime_dependencies": missing_runtime,
            "missing_reproducibility_dependencies": missing_repro,
            "missing_repo_dependencies": missing_repo,
            "artifact_count_by_classification": counts,
            "bytes_by_classification": bytes_by_class,
            "cleanup_target_count": counts[
                "UNREFERENCED_CLEANUP_TARGET"
            ],
        },
    }


def write_reports(payload: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)

    (output / "dependency_inventory.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": payload["schema_version"],
                "kind": payload["kind"],
                "generated_at_utc": payload["generated_at_utc"],
                "data_root": payload["data_root"],
                "safety": payload["safety"],
                "summary": payload["summary"],
            },
            indent=2,
        )
        + "\n"
    )

    matrix_fields = [
        "study",
        "arm",
        "recipe",
        "seeds",
        "dependency",
        "path",
        "retention",
        "roles",
        "expected_kind",
        "exists",
        "kind_ok",
        "required_files",
        "missing_files",
        "valid",
    ]
    with (output / "dependency_matrix.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=matrix_fields)
        writer.writeheader()
        for row in payload["arm_matrix"]:
            flat = dict(row)
            for key in ("seeds", "roles", "required_files", "missing_files"):
                flat[key] = "|".join(str(x) for x in flat.get(key) or [])
            writer.writerow({k: flat.get(k) for k in matrix_fields})

    artifact_fields = [
        "path",
        "kind",
        "classification",
        "reasons",
        "size_bytes",
    ]
    with (output / "artifact_inventory.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=artifact_fields)
        writer.writeheader()
        for row in payload["artifact_inventory"]:
            flat = dict(row)
            flat["reasons"] = " | ".join(flat["reasons"])
            writer.writerow(flat)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root")
    ap.add_argument(
        "--registry",
        default=str(DEFAULT_DEPENDENCIES.relative_to(REPO_ROOT)),
    )
    ap.add_argument(
        "--contract",
        default=str(DEFAULT_CONTRACT.relative_to(REPO_ROOT)),
    )
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-sizes", action="store_true")
    args = ap.parse_args()

    raw_root = args.data_root or os.environ.get("METHYL_DATA_ROOT")
    if not raw_root:
        ap.error(
            "--data-root is required unless METHYL_DATA_ROOT is set"
        )
    data_root = Path(raw_root).expanduser().resolve()
    if not data_root.is_dir():
        ap.error(f"data root does not exist: {data_root}")

    output = Path(args.output).expanduser().resolve()
    try:
        output.relative_to(data_root)
    except ValueError:
        pass
    else:
        ap.error("--output must live outside --data-root")

    registry = (REPO_ROOT / args.registry).resolve()
    contract = (REPO_ROOT / args.contract).resolve()

    payload = build_inventory(
        data_root=data_root,
        registry_path=registry,
        contract_path=contract,
        include_sizes=not args.no_sizes,
    )
    write_reports(payload, output)

    s = payload["summary"]
    print(
        f"[paper-deps] studies={s['study_count']} "
        f"arms={s['arm_count']}"
    )
    print(
        "[paper-deps] missing runtime:",
        s["missing_runtime_dependencies"] or "none",
    )
    print(
        "[paper-deps] missing reproducibility:",
        s["missing_reproducibility_dependencies"] or "none",
    )
    print(
        "[paper-deps] missing repo:",
        s["missing_repo_dependencies"] or "none",
    )
    for cls, count in sorted(
        s["artifact_count_by_classification"].items()
    ):
        print(f"  {cls}: {count}")
    print(f"[paper-deps] reports -> {output}")
    print("[paper-deps] READ ONLY: no data was modified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
