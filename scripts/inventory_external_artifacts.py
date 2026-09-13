#!/usr/bin/env python3
"""Dry-run dependency/size inventory for external MethylPredictor artifacts.

This program NEVER deletes, moves, truncates, renames, or modifies anything
under --data-root. It only reads filesystem state and writes reports to
--output, which must live outside --data-root.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = REPO_ROOT / "configs" / "external_cleanup_policy.yaml"
DEFAULT_PROFILE = REPO_ROOT / "configs" / "data" / "paper_chr1.yaml"

CLASSIFICATIONS = {"KEEP", "PROTECT", "REVIEW", "DELETE_CANDIDATE"}


@dataclass
class Artifact:
    path: str
    kind: str
    size_bytes: int | None
    classification: str
    reasons: list[str]
    referenced_by: list[str]
    exists: bool = True


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected YAML mapping")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _relative(path: Path, data_root: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(data_root.resolve()))
    except ValueError:
        return None


def _related(a: str, b: str) -> bool:
    pa, pb = Path(a), Path(b)
    return pa == pb or pa in pb.parents or pb in pa.parents


def _du_size(path: Path) -> int | None:
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
        pass

    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return None

    total = 0
    try:
        for base, _, names in os.walk(path):
            for name in names:
                try:
                    total += (Path(base) / name).stat().st_size
                except OSError:
                    continue
        return total
    except OSError:
        return None


def _profile_refs(profile_path: Path, data_root: Path) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}
    if not profile_path.is_file():
        return refs

    payload = _read_yaml(profile_path)

    # Only runtime INPUTS are dependency evidence. Destination paths such as
    # output_root must never protect every artifact written underneath them.
    input_keys = {
        "canonical_root",
        "prepared_root",
        "registry",
        "rna_cache",
        "prior_cache",
        "cpg_targets_dir",
        "functional_atlas",
        "annotation_cache",
        # New genome-wide locus feature store (paths.locus_store), replacing
        # the legacy functional_atlas/annotation_cache pair for fresh runs --
        # see docs/data/LOCUS_FEATURES.md. Without this key, derived/locus_
        # features_v1 was spuriously classified REVIEW/unreferenced even
        # though it is an active, required paper-runtime dependency.
        "locus_store",
    }

    paths = payload.get("paths") or {}
    # functional_atlas/annotation_cache may live under a legacy_frozen block
    # (see configs/data/paper_chr1.yaml) rather than paths, once locus_store
    # supersedes them for fresh runs; check both so either generation of
    # profile is picked up as reference evidence.
    legacy_frozen = payload.get("legacy_frozen") or {}
    paths = {**legacy_frozen, **paths}
    for key in sorted(input_keys):
        raw = paths.get(key)
        if raw is None:
            continue

        p = Path(raw)
        resolved = p if p.is_absolute() else data_root / p
        rel = _relative(resolved, data_root)
        if rel is not None:
            refs.setdefault(rel, []).append(
                f"profile:{profile_path.name}:{key}"
            )

    return refs


def _paper_record_paths(data_root: Path) -> list[Path]:
    runs = data_root / "experiments" / "runs"
    if not runs.is_dir():
        return []
    return sorted(runs.glob("*/*/*/paper/record.json"))


def _add_path_ref(
    refs: dict[str, list[str]],
    raw: Any,
    data_root: Path,
    label: str,
) -> None:
    if not isinstance(raw, str) or not raw:
        return
    rel = _relative(Path(raw), data_root)
    if rel is not None:
        refs.setdefault(rel, []).append(label)


def _paper_refs(
    data_root: Path,
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    refs: dict[str, list[str]] = {}
    records: list[dict[str, Any]] = []

    for path in _paper_record_paths(data_root):
        try:
            record = _read_json(path)
        except Exception as exc:
            records.append(
                {"path": str(path), "valid": False, "error": str(exc)}
            )
            continue

        if (
            record.get("schema_version") != 1
            or record.get("kind") != "methylpredictor.paper_run"
        ):
            records.append(
                {
                    "path": str(path),
                    "valid": False,
                    "error": "unsupported paper record schema/kind",
                }
            )
            continue

        label = f"paper_record:{path}"

        runtime = record.get("runtime") or {}
        _add_path_ref(refs, runtime.get("run_dir"), data_root, label)

        for section in ("training", "checkpoint", "resolved_config"):
            node = record.get(section)
            if isinstance(node, dict):
                _add_path_ref(refs, node.get("path"), data_root, label)

        evaluation = record.get("evaluation") or {}
        _add_path_ref(refs, evaluation.get("path"), data_root, label)

        data_profile = record.get("data_profile") or {}
        resolved = data_profile.get("resolved_paths") or {}
        if isinstance(resolved, dict):
            for key, raw in resolved.items():
                _add_path_ref(
                    refs,
                    raw,
                    data_root,
                    f"{label}:data:{key}",
                )

        records.append(
            {
                "path": str(path),
                "valid": True,
                "run_id": record.get("run_id"),
                "study": record.get("study"),
                "arm": record.get("arm"),
            }
        )

    return refs, records


def _merge_refs(*groups: dict[str, list[str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for group in groups:
        for path, labels in group.items():
            out.setdefault(path, []).extend(labels)
    return {k: sorted(set(v)) for k, v in out.items()}


def _explicit_keeps(policy: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for section in ("always_keep", "paper_runtime_keep"):
        for row in policy.get(section) or []:
            out.setdefault(str(row["path"]), []).append(
                str(row.get("reason") or section)
            )
    return out


def _patterns(policy: dict[str, Any]) -> list[str]:
    groups = policy.get("candidate_name_patterns") or {}
    return [
        str(pattern).lower()
        for values in groups.values()
        for pattern in (values or [])
    ]


def _reference_hits(
    artifact_path: str,
    refs: dict[str, list[str]],
) -> list[str]:
    hits: list[str] = []
    for ref_path, labels in refs.items():
        if _related(artifact_path, ref_path):
            hits.extend(labels)
    return sorted(set(hits))


def _keep_hits(
    artifact_path: str,
    keeps: dict[str, list[str]],
) -> list[str]:
    hits: list[str] = []
    for keep_path, reasons in keeps.items():
        if _related(artifact_path, keep_path):
            hits.extend(reasons)
    return sorted(set(hits))


def discover_artifacts(data_root: Path) -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []

    derived = data_root / "derived"
    if derived.is_dir():
        found.extend((p, "derived") for p in sorted(derived.iterdir()))

    runs = data_root / "experiments" / "runs"
    if runs.is_dir():
        for storage in sorted(p for p in runs.iterdir() if p.is_dir()):
            for scope in sorted(p for p in storage.iterdir() if p.is_dir()):
                for run in sorted(p for p in scope.iterdir() if p.is_dir()):
                    found.append((run, "run"))

    searches = data_root / "experiments" / "searches"
    if searches.is_dir():
        for model in sorted(p for p in searches.iterdir() if p.is_dir()):
            for scope in sorted(p for p in model.iterdir() if p.is_dir()):
                for search in sorted(p for p in scope.iterdir() if p.is_dir()):
                    found.append((search, "search"))

    for root_name in ("model_weights_cache", "reference"):
        base = data_root / root_name
        if base.is_file():
            found.append((base, root_name))
        elif base.is_dir():
            found.extend(
                (p, root_name) for p in sorted(base.iterdir())
            )

    canonical = data_root / "datasets" / "methylprophet_repro_v1"
    if canonical.exists():
        found.append((canonical, "canonical"))

    unique: dict[str, tuple[Path, str]] = {}
    for path, kind in found:
        unique.setdefault(str(path.resolve()), (path, kind))
    return sorted(unique.values(), key=lambda row: str(row[0]))


def classify(
    *,
    relative_path: str,
    kind: str,
    explicit_keeps: dict[str, list[str]],
    all_refs: dict[str, list[str]],
    paper_refs: dict[str, list[str]],
    patterns: list[str],
    architecture_selected: bool,
    fresh_paper_runs_verified: bool,
) -> tuple[str, list[str], list[str]]:
    all_hits = _reference_hits(relative_path, all_refs)
    paper_hits = _reference_hits(relative_path, paper_refs)
    keep_hits = _keep_hits(relative_path, explicit_keeps)

    if paper_hits:
        return (
            "KEEP",
            ["referenced by standardized final paper record"],
            all_hits,
        )

    if keep_hits:
        return "KEEP", keep_hits, all_hits

    if all_hits:
        return (
            "KEEP",
            ["referenced by current paper data profile"],
            all_hits,
        )

    if kind in {"run", "search"}:
        if not architecture_selected:
            return (
                "PROTECT",
                ["architecture selection not locked"],
                all_hits,
            )
        if not fresh_paper_runs_verified:
            return (
                "PROTECT",
                ["fresh final paper runs not verified"],
                all_hits,
            )

    matched = sorted(
        {p for p in patterns if p in relative_path.lower()}
    )
    gates_open = architecture_selected and fresh_paper_runs_verified

    if matched and gates_open:
        return (
            "DELETE_CANDIDATE",
            [
                "unreferenced and matches explicit candidate pattern(s): "
                + ", ".join(matched),
                "candidate only; destructive actions remain disabled",
            ],
            all_hits,
        )

    reasons = ["not referenced by current/final paper contract"]
    if matched:
        reasons.insert(
            0,
            "matches historical/scratch pattern but safety gates are closed",
        )
    return "REVIEW", reasons, all_hits


def build_inventory(
    *,
    data_root: Path,
    policy_path: Path,
    profile_path: Path,
    architecture_selected: bool,
    fresh_paper_runs_verified: bool,
    include_sizes: bool,
) -> dict[str, Any]:
    policy = _read_yaml(policy_path)
    if policy.get("schema_version") != 1:
        raise ValueError("unsupported cleanup policy schema")
    if (
        (policy.get("policy") or {}).get("destructive_actions_allowed")
        is not False
    ):
        raise ValueError(
            "cleanup policy must explicitly disable destructive actions"
        )
    if fresh_paper_runs_verified and not architecture_selected:
        raise ValueError(
            "--fresh-paper-runs-verified requires --architecture-selected"
        )

    profile_refs = _profile_refs(profile_path, data_root)
    paper_refs, paper_records = _paper_refs(data_root)
    all_refs = _merge_refs(profile_refs, paper_refs)
    keeps = _explicit_keeps(policy)
    patterns = _patterns(policy)

    artifacts: list[Artifact] = []
    for path, kind in discover_artifacts(data_root):
        rel = _relative(path, data_root)
        if rel is None:
            continue
        cls, reasons, refs = classify(
            relative_path=rel,
            kind=kind,
            explicit_keeps=keeps,
            all_refs=all_refs,
            paper_refs=paper_refs,
            patterns=patterns,
            architecture_selected=architecture_selected,
            fresh_paper_runs_verified=fresh_paper_runs_verified,
        )
        artifacts.append(
            Artifact(
                path=rel,
                kind=kind,
                size_bytes=_du_size(path) if include_sizes else None,
                classification=cls,
                reasons=reasons,
                referenced_by=refs,
            )
        )

    counts = {name: 0 for name in sorted(CLASSIFICATIONS)}
    sizes = {name: 0 for name in sorted(CLASSIFICATIONS)}
    unknown = {name: 0 for name in sorted(CLASSIFICATIONS)}
    for artifact in artifacts:
        counts[artifact.classification] += 1
        if artifact.size_bytes is None:
            unknown[artifact.classification] += 1
        else:
            sizes[artifact.classification] += artifact.size_bytes

    layout_anomalies = [
        a.path
        for a in artifacts
        if a.path.startswith("experiments/runs/runs/")
    ]

    return {
        "schema_version": 1,
        "kind": "methylpredictor.external_cleanup_inventory",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root.resolve()),
        "safety": {
            "dry_run_only": True,
            "destructive_actions_allowed": False,
            "architecture_selected": architecture_selected,
            "fresh_paper_runs_verified": fresh_paper_runs_verified,
            "valid_paper_record_count": sum(
                1 for r in paper_records if r.get("valid")
            ),
            "invalid_paper_record_count": sum(
                1 for r in paper_records if not r.get("valid")
            ),
        },
        "references": {
            "profile_paths": profile_refs,
            "paper_record_paths": paper_refs,
            "paper_records": paper_records,
        },
        "summary": {
            "artifact_count": len(artifacts),
            "count_by_classification": counts,
            "bytes_by_classification": sizes,
            "unknown_size_count_by_classification": unknown,
            "layout_anomaly_count": len(layout_anomalies),
            "layout_anomalies": layout_anomalies,
        },
        "artifacts": [asdict(a) for a in artifacts],
    }


def write_reports(payload: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)

    (output / "inventory.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    fields = [
        "path",
        "kind",
        "size_bytes",
        "classification",
        "reasons",
        "referenced_by",
        "exists",
    ]
    with (output / "inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["artifacts"]:
            flat = dict(row)
            flat["reasons"] = " | ".join(flat["reasons"])
            flat["referenced_by"] = " | ".join(flat["referenced_by"])
            writer.writerow(flat)

    summary = {
        "schema_version": payload["schema_version"],
        "kind": payload["kind"],
        "generated_at_utc": payload["generated_at_utc"],
        "data_root": payload["data_root"],
        "safety": payload["safety"],
        "summary": payload["summary"],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument(
        "--policy",
        default=str(DEFAULT_POLICY.relative_to(REPO_ROOT)),
    )
    ap.add_argument(
        "--profile",
        default=str(DEFAULT_PROFILE.relative_to(REPO_ROOT)),
    )
    ap.add_argument("--output", required=True)
    ap.add_argument("--architecture-selected", action="store_true")
    ap.add_argument("--fresh-paper-runs-verified", action="store_true")
    ap.add_argument("--no-sizes", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    policy = (REPO_ROOT / args.policy).resolve()
    profile = (REPO_ROOT / args.profile).resolve()

    if not data_root.is_dir():
        ap.error(f"data root does not exist: {data_root}")
    if not policy.is_file():
        ap.error(f"policy does not exist: {policy}")
    if not profile.is_file():
        ap.error(f"profile does not exist: {profile}")

    try:
        output.relative_to(data_root)
    except ValueError:
        pass
    else:
        ap.error("--output must live outside --data-root")

    payload = build_inventory(
        data_root=data_root,
        policy_path=policy,
        profile_path=profile,
        architecture_selected=args.architecture_selected,
        fresh_paper_runs_verified=args.fresh_paper_runs_verified,
        include_sizes=not args.no_sizes,
    )
    write_reports(payload, output)

    summary = payload["summary"]
    print(f"[cleanup-inventory] artifacts={summary['artifact_count']}")
    for cls, count in sorted(
        summary["count_by_classification"].items()
    ):
        print(
            f"  {cls}: {count} artifact(s), "
            f"{summary['bytes_by_classification'][cls]} known bytes"
        )
    if summary.get("layout_anomaly_count"):
        print(
            "[cleanup-inventory] WARNING: unexpected run layout(s):",
            flush=True,
        )
        for path in summary["layout_anomalies"]:
            print(f"  {path}", flush=True)

    print(f"[cleanup-inventory] reports -> {output}")
    print("[cleanup-inventory] DRY RUN ONLY: nothing was deleted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
