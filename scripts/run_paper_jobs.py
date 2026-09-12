#!/usr/bin/env python3
"""Deterministic multi-machine paper job runner.

Reads the single job matrix in ``configs/paper_studies.yaml``, expands it into
a stable, ordered job list, shards it deterministically across machines/GPUs,
validates each job's data dependencies, and drives ``scripts/paper_experiment.py``
for jobs that are not already complete.

No database and no Slurm dependency -- everything needed to resume or re-shard
comes from the deterministic ordering plus each job's own
``paper/record.json`` completion marker under ``METHYL_DATA_ROOT``.

Examples
--------
List every job (no execution):

    python scripts/run_paper_jobs.py --list

Run one shard of three on GPU 0:

    python scripts/run_paper_jobs.py --shard 0/3 --gpu 0
    python scripts/run_paper_jobs.py --shard 1/3 --gpu 0
    python scripts/run_paper_jobs.py --shard 2/3 --gpu 0

Filter to one study/arm:

    python scripts/run_paper_jobs.py --study main --arm main --dry-run
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDIES_CONFIG = REPO_ROOT / "configs" / "paper_studies.yaml"
STORAGE_CONFIG = REPO_ROOT / "configs" / "data" / "storage_v2.yaml"
DEFAULT_PROFILE = REPO_ROOT / "configs" / "data" / "paper_chr1.yaml"
PAPER_EXPERIMENT = REPO_ROOT / "scripts" / "paper_experiment.py"

BLOCKED_DEPENDENCY = "bulkrnabert"


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return payload


@dataclass(frozen=True)
class Job:
    study: str
    arm: str
    seed: int | None
    scope: str
    recipe: str
    run_id: str
    required_dependencies: tuple[str, ...] = field(default_factory=tuple)

    @property
    def key(self) -> str:
        seed_part = "noseed" if self.seed is None else f"seed{self.seed}"
        return f"{self.study}/{self.arm}/{seed_part}"


def load_job_matrix(
    studies_path: Path = STUDIES_CONFIG,
    scope: str = "chr1",
) -> list[Job]:
    """Expand ``configs/paper_studies.yaml`` into a deterministically ordered
    job list. Ordering is stable (study, arm, seed) so sharding is
    reproducible across processes/machines regardless of dict iteration
    order or filesystem state."""
    payload = _read_yaml(studies_path)
    if payload.get("schema_version") != 1:
        raise ValueError(f"{studies_path}: unsupported schema_version")

    jobs: list[Job] = []
    for study, study_def in payload["studies"].items():
        seeds = study_def.get("seeds") or []
        for arm, arm_def in study_def["arms"].items():
            recipe = arm_def["recipe"]
            deps = tuple(arm_def.get("required_dependencies") or ())
            if arm_def.get("no_seed"):
                run_id = arm
                jobs.append(Job(study, arm, None, scope, recipe, run_id, deps))
                continue
            for seed in seeds:
                run_id = f"{arm}-seed{seed}"
                jobs.append(Job(study, arm, seed, scope, recipe, run_id, deps))

    jobs.sort(key=lambda j: (j.study, j.arm, -1 if j.seed is None else j.seed))
    return jobs


def shard_jobs(jobs: list[Job], shard_index: int, shard_count: int) -> list[Job]:
    """Deterministic contiguous-modulo sharding over the stable job order.

    Job ``i`` (0-indexed in the stable sorted order) goes to shard
    ``i % shard_count``. This is stable across machines/processes as long as
    they all read the same ``configs/paper_studies.yaml``.
    """
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if not (0 <= shard_index < shard_count):
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    return [job for i, job in enumerate(jobs) if i % shard_count == shard_index]


def _resolve_data_root(value: str | None) -> Path:
    raw = value or os.environ.get("METHYL_DATA_ROOT")
    if not raw:
        raise ValueError("--data-root is required unless METHYL_DATA_ROOT is set")
    return Path(raw).expanduser().resolve()


def _dependency_paths(
    storage: dict[str, Any], profile: dict[str, Any], dependency: str
) -> list[Path]:
    """Resolve a named dependency group to the local paths it requires."""
    inputs = storage["paper_inputs"]
    if dependency == "chr1_runtime":
        chr1 = inputs["chr1_runtime"]
        return [Path(p) for p in chr1.values()]
    if dependency == "bulkformer_147m":
        return [Path(inputs["comparator_inputs"]["bulkformer_147m"])]
    if dependency == "bulkrnabert":
        return [Path(inputs["comparator_inputs"]["bulkrnabert"])]
    if dependency == "gene_pathway_membership":
        return [Path(inputs["repo_inputs"]["gene_pathway_membership"])]
    raise ValueError(f"unknown dependency name: {dependency!r}")


def check_dependencies(
    job: Job, data_root: Path, storage: dict[str, Any], profile: dict[str, Any]
) -> tuple[str, list[str]]:
    """Return (status, missing) where status is one of READY / BLOCKED."""
    missing: list[str] = []
    for dep in job.required_dependencies:
        for rel in _dependency_paths(storage, profile, dep):
            local = rel if rel.is_absolute() else data_root / rel
            if not local.exists():
                missing.append(f"{dep}:{rel}")
    if missing and any(m.startswith(BLOCKED_DEPENDENCY) for m in missing):
        return "BLOCKED", missing
    if missing:
        return "MISSING_DEPENDENCY", missing
    return "READY", missing


def run_dir_for(job: Job, profile: dict[str, Any], data_root: Path) -> Path:
    output_root = Path(profile["paths"]["output_root"])
    if not output_root.is_absolute():
        output_root = data_root / output_root
    key = (
        "cpg_prior"
        if job.study == "cpg_prior"
        else profile["storage"]["rna_key"]
    )
    return output_root / "runs" / key / job.scope / job.run_id


def is_complete(job: Job, profile: dict[str, Any], data_root: Path) -> bool:
    return (run_dir_for(job, profile, data_root) / "paper" / "record.json").is_file()


def build_command(job: Job, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(PAPER_EXPERIMENT),
        "--profile",
        args.profile,
        "--recipe",
        job.recipe,
        "--run-id",
        job.run_id,
        "--study",
        job.study,
        "--arm",
        job.arm,
    ]
    if job.seed is not None:
        cmd += ["--seed", str(job.seed)]
    if args.data_root:
        cmd += ["--data-root", args.data_root]
    if args.allow_dirty:
        cmd.append("--allow-dirty")
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--studies", default=str(STUDIES_CONFIG.relative_to(REPO_ROOT)))
    ap.add_argument("--profile", default=str(DEFAULT_PROFILE.relative_to(REPO_ROOT)))
    ap.add_argument("--storage-config", default=str(STORAGE_CONFIG.relative_to(REPO_ROOT)))
    ap.add_argument("--data-root")
    ap.add_argument("--scope", default="chr1")
    ap.add_argument("--shard", default="0/1", help="SHARD_INDEX/SHARD_COUNT, e.g. 1/3")
    ap.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES value for launched jobs")
    ap.add_argument("--study", help="filter to one study")
    ap.add_argument("--arm", help="filter to one arm")
    ap.add_argument("--seed", type=int, help="filter to one seed")
    ap.add_argument("--list", action="store_true", help="print the resolved job table and exit")
    ap.add_argument("--dry-run", action="store_true", help="print commands without executing")
    ap.add_argument("--force", action="store_true", help="rerun even if already complete")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    try:
        shard_index_s, shard_count_s = args.shard.split("/")
        shard_index, shard_count = int(shard_index_s), int(shard_count_s)
    except ValueError as exc:
        ap.error(f"--shard must be INDEX/COUNT, got {args.shard!r} ({exc})")

    studies_path = (REPO_ROOT / args.studies).resolve()
    profile_path = (REPO_ROOT / args.profile).resolve()
    storage_path = (REPO_ROOT / args.storage_config).resolve()

    jobs = load_job_matrix(studies_path, scope=args.scope)
    if args.study:
        jobs = [j for j in jobs if j.study == args.study]
    if args.arm:
        jobs = [j for j in jobs if j.arm == args.arm]
    if args.seed is not None:
        jobs = [j for j in jobs if j.seed == args.seed]

    all_jobs_for_shard = load_job_matrix(studies_path, scope=args.scope)
    shard = shard_jobs(all_jobs_for_shard, shard_index, shard_count)
    shard_keys = {j.key for j in shard}
    jobs = [j for j in jobs if j.key in shard_keys]

    data_root = _resolve_data_root(args.data_root)
    profile = _read_yaml(profile_path)
    storage = _read_yaml(storage_path)

    env = dict(os.environ)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    exit_code = 0
    for job in jobs:
        status, missing = check_dependencies(job, data_root, storage, profile)
        complete = is_complete(job, profile, data_root)

        if args.list or args.dry_run:
            note = f"missing={missing}" if missing else ""
            state = "COMPLETE" if complete else status
            print(f"[{job.key}] state={state} recipe={job.recipe} run_id={job.run_id} {note}".rstrip())
            if args.list:
                continue

        if complete and not args.force:
            print(f"[{job.key}] already complete, skipping (pass --force to rerun)")
            continue

        if status == "BLOCKED":
            print(f"[{job.key}] BLOCKED: missing required BulkRNABert dependency: {missing}")
            exit_code = exit_code or 3
            continue
        if status == "MISSING_DEPENDENCY":
            print(f"[{job.key}] MISSING_DEPENDENCY: {missing}")
            exit_code = exit_code or 2
            continue

        cmd = build_command(job, args)
        if args.dry_run:
            print("  +", " ".join(cmd))
            continue

        print(f"[{job.key}] launching: {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
        if result.returncode != 0:
            exit_code = result.returncode

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
