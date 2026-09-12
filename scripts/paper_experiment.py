#!/usr/bin/env python3
"""Standard paper-facing train -> evaluate -> record API.

This wrapper deliberately sits outside ``scripts/experiments`` so the frozen
architecture-search launchers remain byte-stable. It only accepts recipes
classified as ``paper_facing`` in ``configs/experiment_surface.yaml``.

The underlying RNA runtime may still use a historical compatibility storage
key until the cleanup phase; that implementation detail is read from the data
profile rather than exposed as public API.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_SURFACE = REPO_ROOT / "configs" / "experiment_surface.yaml"
DEFAULT_PROFILE = REPO_ROOT / "configs" / "data" / "paper_chr1.yaml"
RECORD_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    )


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError as exc:
        raise ValueError(f"{path} must live inside the repository") from exc


def paper_facing_recipe_records() -> dict[str, dict[str, Any]]:
    payload = _read_yaml(EXPERIMENT_SURFACE)
    records = payload["models"]["paper_facing"]
    return {record["path"]: record for record in records}


def validate_paper_recipe(recipe: Path) -> tuple[str, dict[str, Any]]:
    relative = _repo_relative(recipe)
    records = paper_facing_recipe_records()
    if relative not in records:
        raise ValueError(
            f"{relative} is not classified as a paper_facing recipe in "
            "configs/experiment_surface.yaml"
        )
    record = records[relative]
    expected = record.get("sha256")
    actual = _sha256(recipe)
    if expected and expected != actual:
        raise ValueError(
            f"paper-facing recipe hash mismatch for {relative}: "
            f"manifest={expected}, actual={actual}"
        )
    return relative, record


def classify_recipe(recipe: Path) -> str:
    raw = _read_yaml(recipe)
    if raw.get("model") is None and raw.get("training") is None:
        return "cpg_prior"
    return "rna_methylation"


def resolve_data_root(value: str | None) -> Path:
    raw = value or os.environ.get("METHYL_DATA_ROOT")
    if not raw:
        raise ValueError(
            "--data-root is required unless METHYL_DATA_ROOT is set"
        )
    return Path(raw).expanduser().resolve()


def load_profile(profile_path: Path, data_root: Path) -> dict[str, Any]:
    raw = _read_yaml(profile_path)
    if raw.get("schema_version") != 1:
        raise ValueError(
            f"{profile_path}: unsupported schema_version "
            f"{raw.get('schema_version')!r}"
        )
    scope = raw.get("scope")
    if scope not in {"chr1", "chr123", "genomewide"}:
        raise ValueError(f"{profile_path}: unsupported scope {scope!r}")

    paths_raw = raw.get("paths") or {}
    required = {
        "canonical_root",
        "registry",
        "rna_cache",
        "prior_cache",
        "cpg_targets_dir",
        "functional_atlas",
        "annotation_cache",
        "output_root",
    }
    missing = sorted(required - set(paths_raw))
    if missing:
        raise ValueError(
            f"{profile_path}: missing path entries: {', '.join(missing)}"
        )

    resolved = {}
    for key, value in paths_raw.items():
        p = Path(value)
        resolved[key] = p if p.is_absolute() else data_root / p

    storage = raw.get("storage") or {}
    if not storage.get("rna_key"):
        raise ValueError(f"{profile_path}: storage.rna_key is required")
    if not storage.get("cpg_prior_key"):
        raise ValueError(
            f"{profile_path}: storage.cpg_prior_key is required"
        )

    official = raw.get("official_evaluation") or {}
    views = tuple(official.get("views") or ())
    expected_views = (
        "train_cpg_x_val_sample",
        "val_cpg_x_train_sample",
        "val_cpg_x_val_sample",
    )
    if views != expected_views:
        raise ValueError(
            f"{profile_path}: official_evaluation.views must be exactly "
            f"{expected_views}"
        )
    if official.get("headline_view") != "val_cpg_x_val_sample":
        raise ValueError(
            f"{profile_path}: headline_view must be val_cpg_x_val_sample"
        )

    return {
        "raw": raw,
        "scope": scope,
        "paths": {k: Path(v).resolve() for k, v in resolved.items()},
        "storage": storage,
        "official_evaluation": official,
    }


def runtime_run_dir(
    profile: dict[str, Any],
    model_kind: str,
    run_id: str,
) -> Path:
    key = (
        profile["storage"]["rna_key"]
        if model_kind == "rna_methylation"
        else profile["storage"]["cpg_prior_key"]
    )
    return (
        profile["paths"]["output_root"]
        / "runs"
        / key
        / profile["scope"]
        / run_id
    )


def evaluation_path(run_dir: Path, scope: str) -> Path:
    return run_dir / "evaluation" / scope / "metrics.json"


def paper_record_path(run_dir: Path) -> Path:
    return run_dir / "paper" / "record.json"


def _require_inputs(
    profile: dict[str, Any],
    model_kind: str,
    rna_cache_override: Path | None,
) -> None:
    paths = dict(profile["paths"])
    if rna_cache_override is not None:
        paths["rna_cache"] = rna_cache_override

    keys = ["canonical_root", "registry", "prior_cache"]
    if model_kind == "rna_methylation":
        keys += [
            "rna_cache",
            "cpg_targets_dir",
            "functional_atlas",
            "annotation_cache",
        ]
        if profile["scope"] == "chr1":
            keys.append("prepared_root")

    missing = [
        f"{key}={paths[key]}"
        for key in keys
        if key not in paths or not Path(paths[key]).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "missing required paper-experiment inputs:\n  "
            + "\n  ".join(missing)
        )


def build_train_command(
    *,
    profile: dict[str, Any],
    recipe_relative: str,
    run_id: str,
    seed: int,
    resume: bool,
    rna_cache_override: Path | None,
) -> list[str]:
    paths = profile["paths"]
    rna_cache = rna_cache_override or paths["rna_cache"]
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--model",
        "rna_methylation",
        "--scope",
        profile["scope"],
        "--recipe",
        recipe_relative,
        "--mode",
        "final",
        "--seed",
        str(seed),
        "--run-id",
        run_id,
        "--canonical-root",
        str(paths["canonical_root"]),
        "--registry",
        str(paths["registry"]),
        "--rna-cache",
        str(rna_cache),
        "--prior-cache",
        str(paths["prior_cache"]),
        "--cpg-targets-dir",
        str(paths["cpg_targets_dir"]),
        "--functional-atlas",
        str(paths["functional_atlas"]),
        "--annotation-cache",
        str(paths["annotation_cache"]),
        "--output-root",
        str(paths["output_root"]),
    ]
    if profile["scope"] == "chr1":
        cmd += ["--prepared-root", str(paths["prepared_root"])]
    if resume:
        cmd.append("--resume")
    return cmd


def build_evaluate_command(
    *,
    profile: dict[str, Any],
    model_kind: str,
    recipe_relative: str,
    run_dir: Path,
    rna_cache_override: Path | None,
) -> list[str]:
    paths = profile["paths"]
    output = evaluation_path(run_dir, profile["scope"])
    if model_kind == "cpg_prior":
        return [
            sys.executable,
            "scripts/evaluate.py",
            "--model",
            "cpg_prior",
            "--eval-scope",
            profile["scope"],
            "--canonical-root",
            str(paths["canonical_root"]),
            "--feature-cache",
            str(paths["prior_cache"]),
            "--registry",
            str(paths["registry"]),
            "--output",
            str(output),
        ]

    rna_cache = rna_cache_override or paths["rna_cache"]
    cmd = [
        sys.executable,
        "scripts/evaluate.py",
        "--model",
        "rna_methylation",
        "--checkpoint",
        str(run_dir / "checkpoints" / "best.pt"),
        "--eval-scope",
        profile["scope"],
        "--recipe",
        recipe_relative,
        "--canonical-root",
        str(paths["canonical_root"]),
        "--registry",
        str(paths["registry"]),
        "--rna-cache",
        str(rna_cache),
        "--prior-cache",
        str(paths["prior_cache"]),
        "--cpg-targets-dir",
        str(paths["cpg_targets_dir"]),
        "--functional-atlas",
        str(paths["functional_atlas"]),
        "--annotation-cache",
        str(paths["annotation_cache"]),
        "--output",
        str(output),
    ]
    if profile["scope"] == "chr1":
        cmd += ["--prepared-root", str(paths["prepared_root"])]
    return cmd


def _run(cmd: list[str]) -> None:
    print("+", shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    tmp.replace(path)


def build_record(
    *,
    profile_path: Path,
    profile: dict[str, Any],
    recipe: Path,
    recipe_relative: str,
    model_kind: str,
    run_id: str,
    seed: int | None,
    study: str | None,
    arm: str | None,
    run_dir: Path,
    rna_cache_override: Path | None,
) -> dict[str, Any]:
    ev_path = evaluation_path(run_dir, profile["scope"])
    if not ev_path.is_file():
        raise FileNotFoundError(
            f"official evaluation missing: {ev_path}"
        )
    evaluation = _read_json(ev_path)

    expected_views = tuple(profile["official_evaluation"]["views"])
    actual_views = evaluation.get("views") or {}
    missing_views = [v for v in expected_views if v not in actual_views]
    if missing_views:
        raise ValueError(
            "evaluation is missing official view(s): "
            + ", ".join(missing_views)
        )

    headline_view = profile["official_evaluation"]["headline_view"]
    headline = actual_views[headline_view]
    if evaluation.get("metrics") != headline:
        raise ValueError(
            "evaluation.metrics does not equal the configured headline view"
        )

    checkpoint = run_dir / "checkpoints" / "best.pt"
    training_summary_path = run_dir / "training" / "summary.json"
    resolved_config = run_dir / "config.resolved.yaml"

    if model_kind == "rna_methylation":
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint missing: {checkpoint}")
        if not training_summary_path.is_file():
            raise FileNotFoundError(
                f"training summary missing: {training_summary_path}"
            )

    data_paths = {
        k: str(v)
        for k, v in profile["paths"].items()
    }
    if rna_cache_override is not None:
        data_paths["rna_cache"] = str(rna_cache_override.resolve())

    record = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "kind": "methylpredictor.paper_run",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": study,
        "arm": arm,
        "model": model_kind,
        "scope": profile["scope"],
        "seed": seed,
        "run_id": run_id,
        "git": {
            "commit": _git_head(),
            "dirty": _git_dirty(),
        },
        "recipe": {
            "path": recipe_relative,
            "sha256": _sha256(recipe),
        },
        "data_profile": {
            "path": _repo_relative(profile_path),
            "sha256": _sha256(profile_path),
            "resolved_paths": data_paths,
        },
        "runtime": {
            "run_dir": str(run_dir.resolve()),
            "storage_key": run_dir.parent.parent.name,
        },
        "training": None,
        "checkpoint": None,
        "evaluation": {
            "path": str(ev_path.resolve()),
            "sha256": _sha256(ev_path),
            "headline_view": headline_view,
            "headline_metrics": headline,
            "views": actual_views,
        },
    }

    if training_summary_path.is_file():
        record["training"] = {
            "path": str(training_summary_path.resolve()),
            "sha256": _sha256(training_summary_path),
            "summary": _read_json(training_summary_path),
        }

    if checkpoint.is_file():
        record["checkpoint"] = {
            "path": str(checkpoint.resolve()),
            "sha256": _sha256(checkpoint),
            "epoch": evaluation.get("checkpoint_epoch"),
        }

    if resolved_config.is_file():
        record["resolved_config"] = {
            "path": str(resolved_config.resolve()),
            "sha256": _sha256(resolved_config),
        }
    else:
        record["resolved_config"] = None

    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default=str(DEFAULT_PROFILE.relative_to(REPO_ROOT)),
        help="repo-relative data profile",
    )
    ap.add_argument("--data-root")
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--study")
    ap.add_argument("--arm")
    ap.add_argument(
        "--stage",
        choices=["train", "evaluate", "record", "all"],
        default="all",
    )
    ap.add_argument(
        "--rna-cache",
        help="override profile RNA cache (e.g. BulkFormer/BulkRNABert cache)",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force-evaluation", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    profile_path = (REPO_ROOT / args.profile).resolve()
    recipe = (REPO_ROOT / args.recipe).resolve()
    if not profile_path.is_file():
        ap.error(f"profile does not exist: {profile_path}")
    if not recipe.is_file():
        ap.error(f"recipe does not exist: {recipe}")

    recipe_relative, _ = validate_paper_recipe(recipe)
    model_kind = classify_recipe(recipe)

    if model_kind == "rna_methylation" and args.seed is None:
        ap.error("--seed is required for trainable RNA recipes")
    if model_kind == "cpg_prior" and args.stage == "train":
        ap.error("cpg_prior has no training stage")
    if model_kind == "cpg_prior" and args.stage == "all":
        # "all" means evaluate + record for a zero-parameter baseline.
        pass

    data_root = resolve_data_root(args.data_root)
    profile = load_profile(profile_path, data_root)
    rna_override = (
        Path(args.rna_cache).expanduser().resolve()
        if args.rna_cache
        else None
    )
    run_dir = runtime_run_dir(profile, model_kind, args.run_id)

    if args.stage in {"train", "all"} and model_kind == "rna_methylation":
        if _git_dirty() and not args.allow_dirty and not args.dry_run:
            raise SystemExit(
                "Refusing final paper training from a dirty Git tree. "
                "Commit/stash changes or pass --allow-dirty for a non-final smoke run."
            )

    _require_inputs(profile, model_kind, rna_override)

    train_cmd = None
    if model_kind == "rna_methylation":
        train_cmd = build_train_command(
            profile=profile,
            recipe_relative=recipe_relative,
            run_id=args.run_id,
            seed=args.seed,
            resume=args.resume,
            rna_cache_override=rna_override,
        )
    eval_cmd = build_evaluate_command(
        profile=profile,
        model_kind=model_kind,
        recipe_relative=recipe_relative,
        run_dir=run_dir,
        rna_cache_override=rna_override,
    )

    if args.dry_run:
        print(f"# model={model_kind} run_dir={run_dir}")
        if args.stage in {"train", "all"} and train_cmd is not None:
            print(shlex.join(train_cmd))
        if args.stage in {"evaluate", "all"}:
            print(shlex.join(eval_cmd))
        if args.stage in {"record", "all"}:
            print(f"# record -> {paper_record_path(run_dir)}")
        return 0

    if args.stage in {"train", "all"} and model_kind == "rna_methylation":
        summary = run_dir / "training" / "summary.json"
        best = run_dir / "checkpoints" / "best.pt"
        complete = summary.is_file() and best.is_file()
        if complete:
            print(
                f"[paper-api] training already complete: {run_dir}",
                flush=True,
            )
        else:
            if run_dir.exists() and not args.resume:
                raise SystemExit(
                    f"incomplete run directory exists: {run_dir}\n"
                    "Inspect it and rerun with --resume if appropriate."
                )
            _run(train_cmd)

    if model_kind == "cpg_prior":
        run_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in {"evaluate", "all"}:
        ev = evaluation_path(run_dir, profile["scope"])
        if ev.is_file() and not args.force_evaluation:
            print(
                f"[paper-api] official evaluation already exists: {ev}",
                flush=True,
            )
        else:
            ev.parent.mkdir(parents=True, exist_ok=True)
            _run(eval_cmd)

    if args.stage in {"record", "all"}:
        record = build_record(
            profile_path=profile_path,
            profile=profile,
            recipe=recipe,
            recipe_relative=recipe_relative,
            model_kind=model_kind,
            run_id=args.run_id,
            seed=args.seed,
            study=args.study,
            arm=args.arm,
            run_dir=run_dir,
            rna_cache_override=rna_override,
        )
        target = paper_record_path(run_dir)
        _write_json(target, record)
        print(f"[paper-api] record -> {target}", flush=True)
        print(json.dumps(record, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
