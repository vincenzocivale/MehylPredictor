"""Structured, auditable storage for training searches, runs and evaluations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import yaml


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_commit(repo_root: str | Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_json(path: str | Path, payload: Any) -> None:
    _atomic_text(Path(path), json.dumps(payload, indent=2, sort_keys=False) + "\n")


def write_yaml(path: str | Path, payload: Any) -> None:
    _atomic_text(Path(path), yaml.safe_dump(payload, sort_keys=False))


@dataclass(slots=True)
class RunStore:
    root: Path
    model: str
    train_scope: str
    run_id: str
    path: Path
    is_new: bool = True

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        model: str,
        train_scope: str,
        seed: int,
        learning_rate: float,
        scheduler: str,
        epochs: int,
        run_id: str | None = None,
        nested: bool = True,
        resume: bool = False,
    ) -> "RunStore":
        if run_id is None:
            lr = f"{learning_rate:.2g}".replace("+", "")
            run_id = f"{_utc_stamp()}_lr{lr}_{scheduler}_e{epochs}_seed{seed}"
        if any(ch in run_id for ch in "/\\ "):
            raise ValueError("run_id must be path-safe")
        root_path = Path(root)
        path = root_path / "runs" / model / train_scope / run_id if nested else root_path / run_id
        existed = path.exists()
        if existed and not resume:
            raise FileExistsError(f"run directory already exists: {path}; pass resume=True to reopen it")
        if existed:
            meta_path = path / "metadata.json"
            if not meta_path.is_file():
                raise RuntimeError(f"cannot resume run without metadata.json: {path}")
            meta = json.loads(meta_path.read_text())
            if meta.get("model") != model or meta.get("training_scope") != train_scope:
                raise RuntimeError("resume run model/scope does not match requested training contract")
        path.mkdir(parents=True, exist_ok=True)
        for name in ("checkpoints", "training", "evaluation", "logs"):
            (path / name).mkdir(exist_ok=True)
        return cls(Path(root), model, train_scope, run_id, path, is_new=not existed)

    @classmethod
    def open(cls, path: str | Path) -> "RunStore":
        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text())
        return cls(path.parents[3], meta["model"], meta["training_scope"], path.name, path, is_new=False)

    def save_resolved_config(self, config: dict[str, Any]) -> None:
        write_yaml(self.path / "config.resolved.yaml", config)

    def save_metadata(self, metadata: dict[str, Any]) -> None:
        payload = {
            "schema_version": 1,
            "model": self.model,
            "training_scope": self.train_scope,
            "run_id": self.run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            **metadata,
        }
        write_json(self.path / "metadata.json", payload)

    def checkpoint(self, name: str) -> Path:
        return self.path / "checkpoints" / name

    def training_file(self, name: str) -> Path:
        return self.path / "training" / name

    def evaluation_dir(self, eval_scope: str) -> Path:
        path = self.path / "evaluation" / eval_scope
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_evaluation_manifest(
        self,
        *,
        eval_scope: str,
        checkpoint: str | Path,
        dataset_contract: str,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        output = self.evaluation_dir(eval_scope) / "manifest.json"
        payload = {
            "schema_version": 1,
            "model": self.model,
            "training_scope": self.train_scope,
            "evaluation_scope": eval_scope,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "dataset_contract": dataset_contract,
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            **(extra or {}),
        }
        write_json(output, payload)
        return output


@dataclass(slots=True)
class SearchStore:
    path: Path

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        model: str,
        scope: str,
        search_id: str | None = None,
    ) -> "SearchStore":
        search_id = search_id or _utc_stamp()
        path = Path(root) / "searches" / model / scope / search_id
        path.mkdir(parents=True, exist_ok=False)
        (path / "runs").mkdir()
        return cls(path)
