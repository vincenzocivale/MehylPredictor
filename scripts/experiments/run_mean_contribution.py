#!/usr/bin/env python3
"""Train/evaluate/diagnose mean_contribution_2026_09 arms on official chr1."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mean_contribution_suite import ENGINE, REPO_ROOT, SEEDS, Arm, data_paths, select  # noqa: E402


def _gpu_free_gb(index: int) -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(index)],
            text=True, stderr=subprocess.DEVNULL,
        )
        return float(out.strip().splitlines()[0]) / 1024.0
    except Exception:
        return None


def _wait_gpu(index: int, minimum: float, max_wait_hours: float, poll_seconds: int) -> bool:
    deadline = time.time() + max_wait_hours * 3600
    while True:
        free = _gpu_free_gb(index)
        if free is None or free >= minimum:
            return True
        if time.time() >= deadline:
            return False
        print(f"[mean-suite] GPU {index}: {free:.1f} GB free, need {minimum:.1f}; waiting", flush=True)
        time.sleep(poll_seconds)


def _run_dir(paths: dict[str, str], arm: Arm, seed: int) -> Path:
    return Path(paths["output_root"]) / "runs" / "locus_cls_joint" / "chr1" / arm.run_id(seed)


def _eval_file(run_dir: Path) -> Path:
    return run_dir / "evaluation" / "chr1" / "metrics.json"


def _diag_file(run_dir: Path) -> Path:
    return run_dir / "evaluation" / "chr1" / "mean_diagnostics.json"


def _train_cmd(arm: Arm, seed: int, paths: dict[str, str], resume: bool) -> list[str]:
    cmd = [
        sys.executable, "scripts/train.py",
        "--model", "rna_methylation", "--scope", "chr1", "--engine", ENGINE, "--mode", "final",
        "--recipe", arm.recipe, "--seed", str(seed), "--run-id", arm.run_id(seed),
        "--canonical-root", paths["canonical_root"], "--prepared-root", paths["prepared_root"],
        "--feature-cache", paths["feature_cache"], "--rna-cache", paths["rna_cache"],
        "--registry", paths["registry"], "--cpg-targets-dir", paths["cpg_targets_dir"],
        "--output-root", paths["output_root"],
    ]
    if resume:
        cmd.append("--resume")
    return cmd


def _eval_cmd(arm: Arm, run_dir: Path, paths: dict[str, str]) -> list[str]:
    return [
        sys.executable, "scripts/evaluate.py",
        "--model", "rna_methylation", "--engine", ENGINE,
        "--checkpoint", str(run_dir / "checkpoints" / "best.pt"),
        "--eval-scope", "chr1", "--recipe", arm.recipe,
        "--canonical-root", paths["canonical_root"], "--prepared-root", paths["prepared_root"],
        "--feature-cache", paths["feature_cache"], "--rna-cache", paths["rna_cache"],
        "--registry", paths["registry"], "--cpg-targets-dir", paths["cpg_targets_dir"],
        "--output", str(_eval_file(run_dir)),
    ]


def _diag_cmd(arm: Arm, run_dir: Path, paths: dict[str, str]) -> list[str]:
    return [
        sys.executable, "scripts/experiments/analyze_mean_contribution.py", "checkpoint",
        "--checkpoint", str(run_dir / "checkpoints" / "best.pt"), "--recipe", arm.recipe,
        "--canonical-root", paths["canonical_root"], "--prepared-root", paths["prepared_root"],
        "--feature-cache", paths["feature_cache"], "--rna-cache", paths["rna_cache"],
        "--registry", paths["registry"], "--cpg-targets-dir", paths["cpg_targets_dir"],
        "--output", str(_diag_file(run_dir)),
    ]


def _parse_seeds(raw: str | None) -> tuple[int, ...]:
    return SEEDS if not raw else tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", help="comma-separated arm names; default all")
    ap.add_argument("--seeds", help="comma-separated seeds; default 17,29,43")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--data-root")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--max-wait-hours", type=float, default=24.0)
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--print-commands", action="store_true")
    ap.add_argument("--refresh-evaluation", action="store_true")
    ap.add_argument("--refresh-diagnostics", action="store_true")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    arms, seeds, paths = select(args.arms), _parse_seeds(args.seeds), data_paths(args.data_root)
    host = socket.gethostname()
    log_dir = REPO_ROOT / "logs" / "mean_contribution"
    log_dir.mkdir(parents=True, exist_ok=True)
    state_path = log_dir / f"queue_state_{host}.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}

    for arm in arms:
        for seed in seeds:
            run_dir = _run_dir(paths, arm, seed)
            train_done = (run_dir / "training" / "summary.json").is_file() and (run_dir / "checkpoints" / "best.pt").is_file()
            eval_done, diag_done = _eval_file(run_dir).is_file(), _diag_file(run_dir).is_file()
            resume = run_dir.is_dir() and not train_done
            if resume and not (run_dir / "checkpoints" / "last.pt").is_file():
                raise RuntimeError(f"{run_dir} exists but has no checkpoints/last.pt; inspect before rerunning")

            commands = []
            if not train_done:
                commands.append(("train", _train_cmd(arm, seed, paths, resume)))
            if not eval_done or args.refresh_evaluation:
                commands.append(("eval", _eval_cmd(arm, run_dir, paths)))
            if not diag_done or args.refresh_diagnostics:
                commands.append(("diag", _diag_cmd(arm, run_dir, paths)))

            if args.print_commands:
                print(f"# {arm.name} seed={seed}")
                for _, cmd in commands:
                    print(shlex.join(cmd))
                continue
            if not commands:
                print(f"[mean-suite] {arm.name} seed={seed}: complete, skipping", flush=True)
                continue
            if not _wait_gpu(args.gpu, args.min_free_gb, args.max_wait_hours, args.poll_seconds):
                return 2

            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONPATH": str(REPO_ROOT / "src")}
            unit = f"{arm.name}-seed{seed}"
            state[unit] = {
                "status": "running", "host": host, "gpu": args.gpu, "run_id": arm.run_id(seed),
                "recipe": arm.recipe, "started_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            failed = False
            for phase, cmd in commands:
                log = log_dir / f"{arm.run_id(seed)}.{phase}.log"
                print(f"[mean-suite] {unit}: {phase} -> {log}", flush=True)
                with log.open("a") as handle:
                    handle.write(f"\n=== {datetime.now(timezone.utc).isoformat()} :: {shlex.join(cmd)}\n")
                    handle.flush()
                    rc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
                if rc != 0:
                    state[unit].update({"status": f"{phase}_failed", "returncode": rc})
                    state_path.write_text(json.dumps(state, indent=2) + "\n")
                    failed = True
                    break
            if not failed:
                state[unit].update({"status": "completed", "finished_at_utc": datetime.now(timezone.utc).isoformat()})
                state_path.write_text(json.dumps(state, indent=2) + "\n")
                print(f"[mean-suite] {unit}: completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
