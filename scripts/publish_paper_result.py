#!/usr/bin/env python3
"""Publish one completed paper record as a unique curated repo JSON.

No Git operation is performed here. The distributed worker owns serialized
commit/push.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = REPO_ROOT / "scripts/collect_paper_runs.py"


def load_collector():
    spec = importlib.util.spec_from_file_location(
        "_single_paper_collector",
        COLLECTOR,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def target_relative_path(curated: dict) -> Path:
    study = curated.get("study") or "unknown_study"
    arm = curated.get("arm") or "unknown_arm"
    seed = curated.get("seed")
    name = f"seed{seed}.json" if seed is not None else "seed_none.json"
    return Path(study) / arm / name


def publish_record(record_path: Path, output_root: Path) -> Path:
    collector = load_collector()
    record = collector._read_json(record_path)
    collector.validate_record(record, record_path)
    curated = collector.curate_record(record)
    collector.assert_no_forbidden_local_paths(curated, str(record_path))

    target = output_root / target_relative_path(curated)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(curated, indent=2, sort_keys=False) + "\n"
    if target.exists() and target.read_text() == content:
        return target
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(content)
    temp.replace(target)
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", required=True)
    ap.add_argument("--output-root", default="results/paper")
    args = ap.parse_args()

    record = Path(args.record).expanduser().resolve()
    output = (REPO_ROOT / args.output_root).resolve()
    try:
        output.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise SystemExit("--output-root must live inside repository") from exc

    target = publish_record(record, output)
    print(json.dumps({"published": str(target.relative_to(REPO_ROOT))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
