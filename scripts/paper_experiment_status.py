#!/usr/bin/env python3
"""Show the shared paper-experiment queue/status file on dune."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/experiment_coordination.yaml"


def render(payload: dict) -> None:
    summary = payload.get("summary") or {}
    print(
        f"campaign={payload.get('campaign')} "
        f"total={summary.get('total')} "
        f"status={summary.get('by_status')}"
    )
    print()
    for job in payload.get("jobs") or []:
        seed = "-" if job.get("seed") is None else str(job["seed"])
        print(
            f"{str(job.get('status') or 'pending'):10s} "
            f"{str(job.get('study') or ''):22s} "
            f"{str(job.get('arm') or ''):24s} "
            f"seed={seed:>2s} "
            f"host={str(job.get('host') or '-'):18s} "
            f"gpu={str(job.get('gpu') or '-'):>3s} "
            f"attempt={job.get('attempt') or 0}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--watch", type=float, metavar="SECONDS")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())
    raw = args.data_root or os.environ.get("METHYL_DATA_ROOT")
    if not raw:
        ap.error("--data-root required unless METHYL_DATA_ROOT is set")
    path = (
        Path(raw).expanduser().resolve()
        / cfg["shared_state"]["relative_root"]
        / cfg["shared_state"]["registry_file"]
    )

    while True:
        if not path.is_file():
            raise SystemExit(f"shared registry not initialized yet: {path}")
        payload = json.loads(path.read_text())
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            render(payload)
        if not args.watch:
            return 0
        time.sleep(args.watch)
        print("\033[2J\033[H", end="")


if __name__ == "__main__":
    raise SystemExit(main())
