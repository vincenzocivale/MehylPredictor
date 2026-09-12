#!/usr/bin/env python3
"""Deterministic multi-machine runner for the frozen paper job matrix."""
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
STUDIES_CONFIG = REPO_ROOT / "configs/paper_studies.yaml"
STORAGE_CONFIG = REPO_ROOT / "configs/data/storage_v2.yaml"
DEFAULT_PROFILE = REPO_ROOT / "configs/data/paper_chr1.yaml"
PAPER_EXPERIMENT = REPO_ROOT / "scripts/paper_experiment.py"


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected YAML mapping")
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
    rna_cache: str | None = None

    @property
    def key(self) -> str:
        seed_part = "noseed" if self.seed is None else f"seed{self.seed}"
        return f"{self.study}/{self.arm}/{seed_part}"


def load_job_matrix(
    studies_path: Path = STUDIES_CONFIG,
    scope: str = "chr1",
) -> list[Job]:
    payload = _read_yaml(studies_path)
    if payload.get("schema_version") != 1:
        raise ValueError(f"{studies_path}: unsupported schema_version")

    jobs: list[Job] = []
    for study, study_def in payload["studies"].items():
        seeds = study_def.get("seeds") or []
        for arm, arm_def in study_def["arms"].items():
            recipe = arm_def["recipe"]
            deps = tuple(arm_def.get("required_dependencies") or ())
            rna_cache = arm_def.get("rna_cache")
            if arm_def.get("no_seed"):
                run_id = f"{study}__{arm}"
                jobs.append(
                    Job(
                        study, arm, None, scope, recipe, run_id, deps, rna_cache
                    )
                )
                continue
            for seed in seeds:
                run_id = f"{study}__{arm}__seed{seed}"
                jobs.append(
                    Job(
                        study,
                        arm,
                        int(seed),
                        scope,
                        recipe,
                        run_id,
                        deps,
                        rna_cache,
                    )
                )

    jobs.sort(
        key=lambda j: (
            j.study,
            j.arm,
            -1 if j.seed is None else j.seed,
        )
    )
    return jobs


def shard_jobs(
    jobs: list[Job], shard_index: int, shard_count: int
) -> list[Job]:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            "shard_index must satisfy 0 <= shard_index < shard_count"
        )
    return [
        job
        for index, job in enumerate(jobs)
        if index % shard_count == shard_index
    ]


def _resolve_data_root(value: str | None) -> Path:
    raw = value or os.environ.get("METHYL_DATA_ROOT")
    if not raw:
        raise ValueError(
            "--data-root is required unless METHYL_DATA_ROOT is set"
        )
    return Path(raw).expanduser().resolve()


def _data_dependency_paths(
    storage: dict[str, Any],
    dependency: str,
) -> list[Path]:
    inputs = storage["paper_inputs"]
    if dependency == "chr1_runtime":
        return [
            Path(value)
            for value in inputs["chr1_runtime"].values()
        ]
    if dependency == "bulkformer_147m":
        return [
            Path(inputs["comparator_inputs"]["bulkformer_147m"])
        ]
    if dependency == "bulkrnabert":
        return [
            Path(inputs["comparator_inputs"]["bulkrnabert"])
        ]
    if dependency == "gene_pathway_membership":
        return []
    raise ValueError(f"unknown dependency name: {dependency!r}")


def _repo_dependency_paths(
    storage: dict[str, Any],
    dependency: str,
) -> list[Path]:
    if dependency != "gene_pathway_membership":
        return []
    return [
        REPO_ROOT
        / storage["paper_inputs"]["repo_inputs"][
            "gene_pathway_membership"
        ]
    ]


def resolve_rna_cache(
    job: Job,
    storage: dict[str, Any],
    data_root: Path,
) -> Path | None:
    if job.rna_cache is None:
        return None

    inputs = storage["paper_inputs"]
    if job.rna_cache == "canonical":
        rel = inputs["chr1_runtime"]["rna_cache"]
    elif job.rna_cache == "bulkformer_147m":
        rel = inputs["comparator_inputs"]["bulkformer_147m"]
    elif job.rna_cache == "bulkrnabert":
        rel = inputs["comparator_inputs"]["bulkrnabert"]
    else:
        raise ValueError(
            f"unknown RNA cache selector {job.rna_cache!r}"
        )
    return (data_root / rel).resolve()


def check_dependencies(
    job: Job,
    data_root: Path,
    storage: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[str, list[str]]:
    del profile
    missing: list[str] = []

    for dep in job.required_dependencies:
        for rel in _data_dependency_paths(storage, dep):
            local = rel if rel.is_absolute() else data_root / rel
            if not local.exists():
                missing.append(f"{dep}:{rel}")

        for local in _repo_dependency_paths(storage, dep):
            if not local.exists():
                missing.append(
                    f"{dep}:{local.relative_to(REPO_ROOT)}"
                )

    if any(item.startswith("bulkrnabert:") for item in missing):
        return "BLOCKED", missing
    if missing:
        return "MISSING_DEPENDENCY", missing
    return "READY", []


def run_dir_for(
    job: Job,
    profile: dict[str, Any],
    data_root: Path,
) -> Path:
    output_root = Path(profile["paths"]["output_root"])
    if not output_root.is_absolute():
        output_root = data_root / output_root
    storage_key = (
        profile["storage"]["cpg_prior_key"]
        if job.study == "cpg_prior"
        else profile["storage"]["rna_key"]
    )
    return (
        output_root
        / "runs"
        / storage_key
        / job.scope
        / job.run_id
    )


def is_complete(
    job: Job,
    profile: dict[str, Any],
    data_root: Path,
) -> bool:
    return (
        run_dir_for(job, profile, data_root)
        / "paper"
        / "record.json"
    ).is_file()


def build_command(
    job: Job,
    args: argparse.Namespace,
    *,
    storage: dict[str, Any],
    data_root: Path,
) -> list[str]:
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

    # Always pass an explicit cache for trainable RNA arms. This is critical
    # for frozen-embedding comparators: dependency validation alone is not
    # sufficient; paper_experiment must receive the comparator cache.
    rna_cache = resolve_rna_cache(job, storage, data_root)
    if rna_cache is not None:
        cmd += ["--rna-cache", str(rna_cache)]

    if args.data_root:
        cmd += ["--data-root", args.data_root]
    if args.allow_dirty:
        cmd.append("--allow-dirty")
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--studies",
        default=str(STUDIES_CONFIG.relative_to(REPO_ROOT)),
    )
    ap.add_argument(
        "--profile",
        default=str(DEFAULT_PROFILE.relative_to(REPO_ROOT)),
    )
    ap.add_argument(
        "--storage-config",
        default=str(STORAGE_CONFIG.relative_to(REPO_ROOT)),
    )
    ap.add_argument("--data-root")
    ap.add_argument("--scope", default="chr1")
    ap.add_argument(
        "--shard",
        default="0/1",
        help="INDEX/COUNT, e.g. 1/3",
    )
    ap.add_argument(
        "--gpu",
        help="CUDA_VISIBLE_DEVICES value for launched jobs",
    )
    ap.add_argument("--study")
    ap.add_argument("--arm")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    try:
        shard_i, shard_n = (
            int(x) for x in args.shard.split("/", 1)
        )
    except ValueError as exc:
        ap.error(f"--shard must be INDEX/COUNT: {exc}")

    studies_path = (REPO_ROOT / args.studies).resolve()
    profile_path = (REPO_ROOT / args.profile).resolve()
    storage_path = (REPO_ROOT / args.storage_config).resolve()

    all_jobs = load_job_matrix(studies_path, scope=args.scope)
    selected = shard_jobs(all_jobs, shard_i, shard_n)

    if args.study:
        selected = [j for j in selected if j.study == args.study]
    if args.arm:
        selected = [j for j in selected if j.arm == args.arm]
    if args.seed is not None:
        selected = [j for j in selected if j.seed == args.seed]

    data_root = _resolve_data_root(args.data_root)
    profile = _read_yaml(profile_path)
    storage = _read_yaml(storage_path)

    final_key = storage["final_storage"]["rna_key"]
    if profile["storage"]["rna_key"] != final_key:
        raise SystemExit(
            "paper profile storage key is inconsistent with "
            f"Data/Results V2: {profile['storage']['rna_key']!r} "
            f"!= {final_key!r}"
        )

    env = dict(os.environ)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    exit_code = 0
    for job in selected:
        status, missing = check_dependencies(
            job,
            data_root,
            storage,
            profile,
        )
        complete = is_complete(job, profile, data_root)

        rna_cache = resolve_rna_cache(job, storage, data_root)
        if args.list or args.dry_run:
            state = "COMPLETE" if complete else status
            suffix = (
                f" rna_cache={rna_cache}"
                if rna_cache is not None
                else ""
            )
            if missing:
                suffix += f" missing={missing}"
            print(
                f"[{job.key}] state={state} "
                f"run_id={job.run_id} recipe={job.recipe}"
                f"{suffix}"
            )
            if args.list:
                continue

        if complete and not args.force:
            print(
                f"[{job.key}] already complete; skipping"
            )
            continue

        if status == "BLOCKED":
            print(
                f"[{job.key}] BLOCKED: {missing}"
            )
            exit_code = exit_code or 3
            continue
        if status == "MISSING_DEPENDENCY":
            print(
                f"[{job.key}] MISSING_DEPENDENCY: {missing}"
            )
            exit_code = exit_code or 2
            continue

        cmd = build_command(
            job,
            args,
            storage=storage,
            data_root=data_root,
        )
        if args.dry_run:
            print("  +", " ".join(cmd))
            continue

        print(
            f"[{job.key}] launching: {' '.join(cmd)}",
            flush=True,
        )
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
        )
        if result.returncode != 0:
            exit_code = result.returncode

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
