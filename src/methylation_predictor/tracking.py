"""Optional experiment tracking with a no-dependency default."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

from .config import RunConfig
from .utils import json_safe


class Tracker:
    """Small tracking interface used by the trainer."""

    enabled = False

    def log(self, values: dict[str, Any], step: int | None = None) -> None:
        del values, step

    def set_summary(self, values: dict[str, Any]) -> None:
        del values

    def log_checkpoint(self, path: str | Path, aliases: list[str] | None = None) -> None:
        del path, aliases

    def finish(self, status: str = "finished") -> None:
        del status


class WandbTracker(Tracker):
    enabled = True

    def __init__(self, config: RunConfig, model: nn.Module, output_dir: Path, resume_id: str | None = None):
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "tracking.backend=wandb requires the optional dependency 'wandb'; "
                "install requirements-rna.txt or run `pip install wandb`."
            ) from exc

        self._wandb = wandb
        tracking = config.tracking
        self.run = wandb.init(
            project=tracking.project,
            entity=tracking.entity,
            group=tracking.group,
            name=tracking.name or config.run_name,
            job_type=tracking.job_type,
            tags=tracking.tags,
            mode=tracking.mode,
            dir=str(output_dir),
            config=json_safe(config.as_dict()),
            id=resume_id,
            resume="allow" if resume_id else None,
        )
        if tracking.watch_model:
            wandb.watch(model, log="gradients", log_freq=max(1, tracking.log_every_steps))

    def log(self, values: dict[str, Any], step: int | None = None) -> None:
        self._wandb.log(json_safe(values), step=step)

    def set_summary(self, values: dict[str, Any]) -> None:
        for key, value in json_safe(values).items():
            self.run.summary[key] = value

    def log_checkpoint(self, path: str | Path, aliases: list[str] | None = None) -> None:
        path = Path(path)
        artifact = self._wandb.Artifact(name=f"{self.run.id}-checkpoint", type="model")
        artifact.add_file(str(path))
        self.run.log_artifact(artifact, aliases=aliases or ["best"])

    def finish(self, status: str = "finished") -> None:
        self._wandb.finish(exit_code=0 if status == "finished" else 1)


def create_tracker(
    config: RunConfig, model: nn.Module, output_dir: Path, resume_id: str | None = None
) -> Tracker:
    backend = config.tracking.backend.lower()
    if backend in {"none", "disabled"} or config.tracking.mode == "disabled":
        return Tracker()
    if backend == "wandb":
        return WandbTracker(config, model, output_dir, resume_id=resume_id)
    raise ValueError(f"unknown tracking backend: {config.tracking.backend!r}")
