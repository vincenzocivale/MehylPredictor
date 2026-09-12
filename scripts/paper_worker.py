#!/usr/bin/env python3
"""Dynamic shared-dune worker for the final paper experiment campaign.

Each worker atomically claims the first missing runnable job. The shared
experiments.json file records pending/running/completed/failed/blocked jobs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import traceback
from typing import Any

import yaml

from methylation_predictor.experiment_coordination import (
    SharedExperimentRegistry,
    directory_lock,
)
from methylation_predictor.telegram_notify import (
    format_job_message,
    send_telegram,
    telegram_credentials,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts/run_paper_jobs.py"
PUBLISH_PATH = REPO_ROOT / "scripts/publish_paper_result.py"
COORD_PATH = REPO_ROOT / "configs/experiment_coordination.yaml"
STUDIES_PATH = REPO_ROOT / "configs/paper_studies.yaml"
PROFILE_PATH = REPO_ROOT / "configs/data/paper_chr1.yaml"
STORAGE_PATH = REPO_ROOT / "configs/data/storage_v2.yaml"


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected YAML mapping")
    return payload


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def git_head() -> str:
    return git("rev-parse", "HEAD").stdout.strip()


def git_branch() -> str:
    return git("branch", "--show-current").stdout.strip()


def git_dirty() -> bool:
    return bool(git("status", "--porcelain").stdout.strip())


def runner_module():
    spec = importlib.util.spec_from_file_location(
        "_paper_runner_worker",
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def data_root(value: str | None) -> Path:
    raw = value or os.environ.get("METHYL_DATA_ROOT")
    if not raw:
        raise ValueError("--data-root required unless METHYL_DATA_ROOT is set")
    return Path(raw).expanduser().resolve()


def catalog_entry(job) -> dict[str, Any]:
    return {
        "job_key": job.key,
        "study": job.study,
        "arm": job.arm,
        "seed": job.seed,
        "scope": job.scope,
        "recipe": job.recipe,
        "run_id": job.run_id,
        "required_dependencies": list(job.required_dependencies),
        "rna_cache": job.rna_cache,
    }


def run_dir(runner, job, profile, root: Path) -> Path:
    return runner.run_dir_for(job, profile, root)


def record_path(runner, job, profile, root: Path) -> Path:
    return run_dir(runner, job, profile, root) / "paper" / "record.json"


def headline_metrics(path: Path) -> dict[str, Any] | None:
    try:
        return (
            (json.loads(path.read_text()).get("evaluation") or {})
            .get("headline_metrics")
        )
    except Exception:
        return None


def order_jobs(jobs, coord: dict[str, Any]):
    queue = coord.get("queue") or {}
    study_order = list(queue.get("study_order") or [])
    seed_order = list(queue.get("seed_order") or [])
    study_rank = {value: i for i, value in enumerate(study_order)}
    seed_rank = {int(value): i for i, value in enumerate(seed_order)}
    return sorted(
        jobs,
        key=lambda j: (
            study_rank.get(j.study, 10_000),
            j.arm,
            seed_rank.get(j.seed, 10_000) if j.seed is not None else 10_000,
            j.key,
        ),
    )


def sync_branch(remote: str, branch: str) -> None:
    if git_dirty():
        raise RuntimeError("Git working tree is dirty")
    if git_branch() != branch:
        raise RuntimeError(
            f"worker must run on {branch!r}; current={git_branch()!r}"
        )
    git("fetch", remote, branch)
    result = git("merge", "--ff-only", f"{remote}/{branch}", check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)


def validate_campaign_code(base_commit: str) -> None:
    current = git_head()
    if current == base_commit:
        return
    ancestor = git(
        "merge-base",
        "--is-ancestor",
        base_commit,
        current,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(
            "checkout is not descended from campaign base commit"
        )
    changed = [
        line.strip()
        for line in git(
            "diff",
            "--name-only",
            f"{base_commit}..{current}",
        ).stdout.splitlines()
        if line.strip()
    ]
    disallowed = [
        path
        for path in changed
        if not path.startswith("results/paper/")
    ]
    if disallowed:
        raise RuntimeError(
            "code/config changed after campaign start: "
            + ", ".join(disallowed)
        )


class Heartbeat:
    def __init__(self, claim, interval: float):
        self.claim = claim
        self.interval = interval
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self) -> None:
        while not self.stop.wait(self.interval):
            try:
                self.claim.heartbeat()
            except Exception as exc:
                print(f"[worker] heartbeat warning: {exc}", flush=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join(timeout=max(2.0, self.interval + 1.0))


def stream_process(cmd: list[str], env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log:
        log.write(
            f"\n===== {datetime.now(timezone.utc).isoformat()} =====\n"
        )
        log.write("COMMAND: " + " ".join(cmd) + "\n")
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return process.wait()


def notify(
    coord: dict[str, Any],
    message: str,
    *,
    disabled: bool,
) -> None:
    if disabled or not (coord.get("telegram") or {}).get("enabled", True):
        return
    token, chat = telegram_credentials()
    if not token or not chat:
        print("[telegram] credentials missing; notification skipped", flush=True)
        return
    send_telegram(
        message,
        token=token,
        chat_id=chat,
        timeout=float((coord.get("telegram") or {}).get("timeout_seconds", 10)),
    )


def publish_git(
    *,
    registry: SharedExperimentRegistry,
    coord: dict[str, Any],
    record: Path,
    job,
    base_commit: str,
) -> tuple[str, str | None]:
    cfg = coord["git_publish"]
    if not cfg.get("enabled", True):
        return "disabled", None

    lock = registry.locks_root / f"{cfg.get('lock_name', 'git-push')}.lock"
    remote = str(cfg.get("remote", "origin"))
    branch = str(cfg["branch"])
    output_root = str(cfg.get("result_root", "results/paper"))

    with directory_lock(
        lock,
        wait_seconds=float(cfg.get("lock_wait_seconds", 600)),
        stale_after_seconds=float(cfg.get("stale_after_seconds", 1800)),
    ):
        sync_branch(remote, branch)
        validate_campaign_code(base_commit)

        result = subprocess.run(
            [
                sys.executable,
                str(PUBLISH_PATH),
                "--record",
                str(record),
                "--output-root",
                output_root,
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        relative = json.loads(result.stdout.strip().splitlines()[-1])["published"]

        if not git("status", "--porcelain", "--", relative).stdout.strip():
            return "already_published", git_head()

        git("add", "--", relative)
        git("commit", "-m", f"Add paper result: {job.key}")

        pushed = git("push", remote, f"HEAD:{branch}", check=False)
        if pushed.returncode != 0:
            git("fetch", remote, branch)
            git("rebase", f"{remote}/{branch}")
            pushed = git("push", remote, f"HEAD:{branch}", check=False)
            if pushed.returncode != 0:
                raise RuntimeError(
                    "git push failed after retry:\n"
                    + (pushed.stderr or pushed.stdout)
                )
        return "pushed", git_head()


def verify_campaign(
    campaign: dict[str, Any],
    *,
    coord: dict[str, Any],
) -> str:
    expected = {
        "campaign": coord["campaign"],
        "job_matrix_sha256": sha256(STUDIES_PATH),
        "profile_sha256": sha256(PROFILE_PATH),
        "storage_sha256": sha256(STORAGE_PATH),
        "git_branch": coord["git_publish"]["branch"],
    }
    mismatch = [
        key
        for key, value in expected.items()
        if campaign.get(key) != value
    ]
    if mismatch:
        raise RuntimeError(
            "campaign definition mismatch: " + ", ".join(mismatch)
        )
    base = str(campaign["base_git_commit"])
    validate_campaign_code(base)
    return base


def selected_jobs(jobs, args):
    values = list(jobs)
    if args.study:
        values = [j for j in values if j.study == args.study]
    if args.arm:
        values = [j for j in values if j.arm == args.arm]
    if args.seed is not None:
        values = [j for j in values if j.seed == args.seed]
    return values


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--study")
    ap.add_argument("--arm")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    ap.add_argument("--idle-exit-minutes", type=float, default=10.0)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-telegram", action="store_true")
    args = ap.parse_args()

    root = data_root(args.data_root)
    coord = read_yaml(COORD_PATH)
    profile = read_yaml(PROFILE_PATH)
    storage = read_yaml(STORAGE_PATH)
    runner = runner_module()

    cfg = coord["shared_state"]
    lease = coord["lease"]
    registry = SharedExperimentRegistry(
        data_root=root,
        campaign=coord["campaign"],
        relative_root=cfg["relative_root"],
        registry_file=cfg.get("registry_file", "experiments.json"),
        campaign_file=cfg.get("campaign_file", "campaign.json"),
        jobs_dir=cfg.get("jobs_dir", "jobs"),
        claims_dir=cfg.get("claims_dir", "claims"),
        logs_dir=cfg.get("logs_dir", "logs"),
        locks_dir=cfg.get("locks_dir", "locks"),
        claim_stale_after_seconds=float(lease.get("stale_after_seconds", 900)),
    )

    git_cfg = coord["git_publish"]
    remote = str(git_cfg.get("remote", "origin"))
    branch = str(git_cfg["branch"])

    # Serialize Git sync too, allowing multiple workers even from the same
    # checkout without racing on .git/index.
    startup_git_lock = registry.locks_root / f"{git_cfg.get('lock_name', 'git-push')}.lock"
    with directory_lock(
        startup_git_lock,
        wait_seconds=float(git_cfg.get("lock_wait_seconds", 600)),
        stale_after_seconds=float(git_cfg.get("stale_after_seconds", 1800)),
    ):
        sync_branch(remote, branch)

    proposed = {
        "job_matrix_sha256": sha256(STUDIES_PATH),
        "profile_sha256": sha256(PROFILE_PATH),
        "storage_sha256": sha256(STORAGE_PATH),
        "base_git_commit": git_head(),
        "git_branch": branch,
    }
    campaign = registry.ensure_campaign(proposed)
    base_commit = verify_campaign(campaign, coord=coord)

    all_jobs = order_jobs(
        runner.load_job_matrix(scope=profile["scope"]),
        coord,
    )
    jobs = selected_jobs(all_jobs, args)
    catalog = [catalog_entry(job) for job in all_jobs]
    catalog_by_key = {row["job_key"]: row for row in catalog}

    # Reconcile record files and dependency blocking with operational state.
    for job in all_jobs:
        entry = catalog_by_key[job.key]
        record = record_path(runner, job, profile, root)
        if record.is_file():
            old = registry.read_state(job.key) or {}
            registry.write_state(
                job.key,
                status="completed",
                catalog_entry=entry,
                completed_at=old.get("completed_at")
                or datetime.fromtimestamp(
                    record.stat().st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
                record_path=str(record),
                publish=old.get("publish"),
            )
            continue
        dep_status, missing = runner.check_dependencies(
            job, root, storage, profile
        )
        if dep_status != "READY":
            registry.write_state(
                job.key,
                status="blocked",
                catalog_entry=entry,
                missing_dependencies=missing,
            )
    registry.refresh_snapshot(catalog)

    # Retry missing Git publication for already completed jobs without rerunning
    # their training/evaluation.
    if not args.no_push:
        for job in jobs:
            record = record_path(runner, job, profile, root)
            if not record.is_file():
                continue
            state = registry.read_state(job.key) or {}
            publish = state.get("publish") or {}
            if publish.get("status") in {"pushed", "already_published"}:
                continue
            try:
                pstatus, commit = publish_git(
                    registry=registry,
                    coord=coord,
                    record=record,
                    job=job,
                    base_commit=base_commit,
                )
                registry.write_state(
                    job.key,
                    status="completed",
                    catalog_entry=catalog_by_key[job.key],
                    publish={
                        "status": pstatus,
                        "commit": commit,
                        "error": None,
                    },
                )
            except Exception as exc:
                registry.write_state(
                    job.key,
                    status="completed",
                    catalog_entry=catalog_by_key[job.key],
                    publish={
                        "status": "push_failed",
                        "commit": None,
                        "error": str(exc),
                    },
                )
        registry.refresh_snapshot(catalog)

    idle_started = time.monotonic()

    while True:
        claim = None
        job = None

        for candidate in jobs:
            entry = catalog_by_key[candidate.key]
            record = record_path(runner, candidate, profile, root)
            if record.is_file():
                continue

            dep_status, missing = runner.check_dependencies(
                candidate, root, storage, profile
            )
            if dep_status != "READY":
                registry.write_state(
                    candidate.key,
                    status="blocked",
                    catalog_entry=entry,
                    missing_dependencies=missing,
                )
                continue

            state = registry.read_state(candidate.key) or {}
            status = state.get("status") or "pending"
            if status == "failed" and not args.retry_failed:
                continue
            if status == "running" and registry.active_claim(candidate.key):
                continue

            owner = {
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "gpu": str(args.gpu),
            }
            claim = registry.try_claim(candidate.key, owner=owner)
            if claim is not None:
                job = candidate
                break

        registry.refresh_snapshot(catalog)

        if claim is None or job is None:
            if not args.loop:
                print("[worker] no claimable job", flush=True)
                return 0
            if not any(registry.active_claim(j.key) for j in jobs):
                print("[worker] no remaining runnable/running job", flush=True)
                return 0
            if time.monotonic() - idle_started >= args.idle_exit_minutes * 60:
                print("[worker] idle timeout", flush=True)
                return 0
            time.sleep(args.poll_seconds)
            continue

        idle_started = time.monotonic()
        entry = catalog_by_key[job.key]
        old = registry.read_state(job.key) or {}
        attempt = int(old.get("attempt") or 0) + 1
        log_path = registry.log_path(job.key, attempt)
        rdir = run_dir(runner, job, profile, root)
        record = record_path(runner, job, profile, root)

        registry.write_state(
            job.key,
            status="running",
            catalog_entry=entry,
            attempt=attempt,
            host=socket.gethostname(),
            pid=os.getpid(),
            gpu=str(args.gpu),
            started_at=datetime.now(timezone.utc).isoformat(),
            log_path=str(log_path),
            run_dir=str(rdir),
            reclaimed_stale=claim.reclaimed_stale,
            error=None,
        )
        registry.refresh_snapshot(catalog)

        if claim.reclaimed_stale:
            notify(
                coord,
                format_job_message(
                    "RECOVERED",
                    job_key=job.key,
                    run_id=job.run_id,
                    gpu=str(args.gpu),
                    detail="stale claim reclaimed by a new worker",
                ),
                disabled=args.no_telegram,
            )

        notify(
            coord,
            format_job_message(
                "STARTED",
                job_key=job.key,
                run_id=job.run_id,
                gpu=str(args.gpu),
            ),
            disabled=args.no_telegram,
        )

        started = time.monotonic()
        try:
            cmd_args = argparse.Namespace(
                profile=str(PROFILE_PATH.relative_to(REPO_ROOT)),
                data_root=str(root),
                allow_dirty=False,
            )
            cmd = runner.build_command(
                job,
                cmd_args,
                storage=storage,
                data_root=root,
            )

            # Recover interrupted training automatically when a resumable
            # checkpoint exists.
            last = rdir / "checkpoints" / "last.pt"
            training_summary = rdir / "training" / "summary.json"
            if rdir.exists() and not record.is_file():
                if not training_summary.is_file() and last.is_file():
                    cmd.append("--resume")

            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            env["METHYLPREDICTOR_CODE_COMMIT"] = base_commit
            env["METHYLPREDICTOR_CAMPAIGN"] = coord["campaign"]

            with claim, Heartbeat(
                claim,
                float(lease.get("heartbeat_seconds", 60)),
            ):
                code = stream_process(cmd, env, log_path)

            duration = time.monotonic() - started
            if code != 0:
                raise RuntimeError(
                    f"paper experiment exited with code {code}"
                )
            if not record.is_file():
                raise RuntimeError(
                    f"successful process did not create {record}"
                )

            pstatus = "disabled"
            pcommit = None
            push_error = None
            if not args.no_push:
                try:
                    pstatus, pcommit = publish_git(
                        registry=registry,
                        coord=coord,
                        record=record,
                        job=job,
                        base_commit=base_commit,
                    )
                except Exception as exc:
                    pstatus = "push_failed"
                    push_error = str(exc)

            registry.write_state(
                job.key,
                status="completed",
                catalog_entry=entry,
                attempt=attempt,
                completed_at=datetime.now(timezone.utc).isoformat(),
                duration_seconds=duration,
                record_path=str(record),
                publish={
                    "status": pstatus,
                    "commit": pcommit,
                    "error": push_error,
                },
                error=None,
            )
            registry.refresh_snapshot(catalog)

            notify(
                coord,
                format_job_message(
                    "COMPLETED",
                    job_key=job.key,
                    run_id=job.run_id,
                    gpu=str(args.gpu),
                    duration_seconds=duration,
                    headline_metrics=headline_metrics(record),
                    detail=f"git={pstatus}",
                ),
                disabled=args.no_telegram,
            )
            if push_error:
                notify(
                    coord,
                    format_job_message(
                        "PUSH_ERROR",
                        job_key=job.key,
                        run_id=job.run_id,
                        gpu=str(args.gpu),
                        detail=push_error,
                    ),
                    disabled=args.no_telegram,
                )

        except BaseException as exc:
            claim.release()
            duration = time.monotonic() - started
            detail = "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip()
            registry.write_state(
                job.key,
                status="failed",
                catalog_entry=entry,
                attempt=attempt,
                failed_at=datetime.now(timezone.utc).isoformat(),
                duration_seconds=duration,
                error=detail,
                log_path=str(log_path),
            )
            registry.refresh_snapshot(catalog)
            notify(
                coord,
                format_job_message(
                    "ERROR",
                    job_key=job.key,
                    run_id=job.run_id,
                    gpu=str(args.gpu),
                    duration_seconds=duration,
                    detail=detail,
                ),
                disabled=args.no_telegram,
            )
            print(f"[worker] {job.key} failed: {detail}", flush=True)
            if not args.loop:
                return 1

        if not args.loop:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
