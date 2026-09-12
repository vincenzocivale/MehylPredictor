"""Shared-dune coordination primitives for distributed paper experiments."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    os.replace(temp, path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def safe_slug(key: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "__", key).strip("._")
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}--{digest}"


def age_seconds(path: Path) -> float:
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except FileNotFoundError:
        return float("inf")


@contextmanager
def directory_lock(
    path: Path,
    *,
    wait_seconds: float,
    stale_after_seconds: float,
    poll_seconds: float = 1.0,
) -> Iterator[None]:
    """Short-lived cross-machine lock based on atomic directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:
            if age_seconds(path) > stale_after_seconds:
                stale = path.with_name(f"{path.name}.stale-{uuid.uuid4().hex}")
                try:
                    os.replace(path, stale)
                except (FileNotFoundError, OSError):
                    pass
                else:
                    shutil.rmtree(stale, ignore_errors=True)
                    continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for lock {path}")
            time.sleep(poll_seconds)
    try:
        yield
    finally:
        shutil.rmtree(path, ignore_errors=True)


class Claim:
    def __init__(
        self,
        *,
        path: Path,
        job_key: str,
        owner: dict[str, Any],
        reclaimed_stale: bool = False,
    ):
        self.path = path
        self.job_key = job_key
        self.owner = owner
        self.reclaimed_stale = reclaimed_stale
        self.released = False

    def heartbeat(self) -> None:
        if self.released:
            return
        atomic_json(
            self.path / "heartbeat.json",
            {
                "job_key": self.job_key,
                "heartbeat_at": utc_now(),
                **self.owner,
            },
        )

    def release(self) -> None:
        if not self.released:
            shutil.rmtree(self.path, ignore_errors=True)
            self.released = True

    def __enter__(self) -> "Claim":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class SharedExperimentRegistry:
    def __init__(
        self,
        *,
        data_root: str | Path,
        campaign: str,
        relative_root: str,
        registry_file: str = "experiments.json",
        campaign_file: str = "campaign.json",
        jobs_dir: str = "jobs",
        claims_dir: str = "claims",
        logs_dir: str = "logs",
        locks_dir: str = "locks",
        claim_stale_after_seconds: float = 900.0,
    ):
        self.data_root = Path(data_root).expanduser().resolve()
        rel = Path(relative_root)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("coordination relative_root must be safe")
        self.campaign = campaign
        self.root = self.data_root / rel
        self.registry_path = self.root / registry_file
        self.campaign_path = self.root / campaign_file
        self.jobs_root = self.root / jobs_dir
        self.claims_root = self.root / claims_dir
        self.logs_root = self.root / logs_dir
        self.locks_root = self.root / locks_dir
        self.claim_stale_after_seconds = float(claim_stale_after_seconds)
        for path in (
            self.jobs_root,
            self.claims_root,
            self.logs_root,
            self.locks_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def job_state_path(self, job_key: str) -> Path:
        return self.jobs_root / f"{safe_slug(job_key)}.json"

    def claim_path(self, job_key: str) -> Path:
        return self.claims_root / f"{safe_slug(job_key)}.lock"

    def log_path(self, job_key: str, attempt: int) -> Path:
        return self.logs_root / f"{safe_slug(job_key)}--attempt{attempt}.log"

    def read_campaign(self) -> dict[str, Any] | None:
        return read_json(self.campaign_path)

    def ensure_campaign(self, metadata: dict[str, Any]) -> dict[str, Any]:
        lock = self.locks_root / "campaign-init.lock"
        with directory_lock(lock, wait_seconds=60, stale_after_seconds=180):
            existing = self.read_campaign()
            if existing is None:
                existing = {
                    "schema_version": 1,
                    "campaign": self.campaign,
                    "created_at": utc_now(),
                    **metadata,
                }
                atomic_json(self.campaign_path, existing)
            return existing

    def read_state(self, job_key: str) -> dict[str, Any] | None:
        return read_json(self.job_state_path(job_key))

    def write_state(
        self,
        job_key: str,
        *,
        status: str,
        catalog_entry: dict[str, Any],
        **updates: Any,
    ) -> dict[str, Any]:
        path = self.job_state_path(job_key)
        old = read_json(path) or {}
        state = {
            **catalog_entry,
            **old,
            **updates,
            "job_key": job_key,
            "status": status,
            "updated_at": utc_now(),
        }
        atomic_json(path, state)
        return state

    def active_claim(self, job_key: str) -> bool:
        path = self.claim_path(job_key)
        if not path.exists():
            return False
        heartbeat = path / "heartbeat.json"
        ref = heartbeat if heartbeat.exists() else path
        return age_seconds(ref) <= self.claim_stale_after_seconds

    def try_claim(
        self,
        job_key: str,
        *,
        owner: dict[str, Any],
    ) -> Claim | None:
        path = self.claim_path(job_key)

        def create(reclaimed: bool) -> Claim | None:
            try:
                path.mkdir(parents=False)
            except FileExistsError:
                return None
            claim = Claim(
                path=path,
                job_key=job_key,
                owner=owner,
                reclaimed_stale=reclaimed,
            )
            atomic_json(
                path / "claim.json",
                {
                    "job_key": job_key,
                    "claimed_at": utc_now(),
                    "reclaimed_stale": reclaimed,
                    **owner,
                },
            )
            claim.heartbeat()
            return claim

        claim = create(False)
        if claim is not None:
            return claim

        heartbeat = path / "heartbeat.json"
        ref = heartbeat if heartbeat.exists() else path
        if age_seconds(ref) <= self.claim_stale_after_seconds:
            return None

        stale = path.with_name(f"{path.name}.stale-{uuid.uuid4().hex}")
        try:
            os.replace(path, stale)
        except (FileNotFoundError, OSError):
            return None
        shutil.rmtree(stale, ignore_errors=True)
        return create(True)

    def refresh_snapshot(
        self,
        catalog: list[dict[str, Any]],
    ) -> dict[str, Any]:
        lock = self.locks_root / "registry.lock"
        with directory_lock(lock, wait_seconds=60, stale_after_seconds=180):
            rows = []
            counts: dict[str, int] = {}
            for entry in catalog:
                state = self.read_state(entry["job_key"])
                if state is None:
                    state = {
                        **entry,
                        "status": "pending",
                        "updated_at": None,
                    }
                rows.append(state)
                status = str(state.get("status") or "pending")
                counts[status] = counts.get(status, 0) + 1
            snapshot = {
                "schema_version": 1,
                "campaign": self.campaign,
                "generated_at": utc_now(),
                "summary": {
                    "total": len(rows),
                    "by_status": counts,
                },
                "jobs": rows,
            }
            atomic_json(self.registry_path, snapshot)
            return snapshot
