#!/usr/bin/env python3
"""Queue runner for the architecture-novelty suite (architecture_novelty_2026_09).

Not part of the stable CLI -- delete alongside scripts/experiments/arch_suite.py
once the study concludes (CLAUDE.md: one-off experiment scripts are deleted, not
kept as a second workflow).

Designed to be started independently on several machines that share the output
root. Each machine takes a shard, each unit of work has a deterministic run id,
and a unit whose run directory already carries the trainer's ``.done`` marker is
skipped -- so shards can overlap, a machine can be added late, and a killed
runner can simply be restarted.

    # preflight, no GPU: build every arm on CPU and check it starts at the prior
    python scripts/experiments/run_arch_suite.py --dry-run

    # machine 1 of 3
    python scripts/experiments/run_arch_suite.py --shard 1/3 --gpu 0

    # a specific stage on a machine with a different data mount
    python scripts/experiments/run_arch_suite.py --stages 2-hyper-connections \
        --data-root /mnt/methyl --gpu 1

Run --stages 0-noise-floor first: every other arm's delta is judged against the
seed SD it measures.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shlex
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from arch_suite import ARMS_BY_NAME, REPO_ROOT, STAGES, Arm, data_paths, jobs, run_id, select  # noqa: E402


def _gpu_free_gb(index: int) -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(index)],
            text=True, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    return float(out.strip().splitlines()[0]) / 1024.0


def _wait_for_gpu(index: int, min_free_gb: float, max_wait_hours: float, poll_seconds: int) -> bool:
    """Block until the GPU has room. Returns False if it never frees up.

    The suite is expected to share a machine with unrelated long-running jobs
    (an NTv3 atlas extraction has held ~16 GB here for days), so a runner that
    grabbed the GPU immediately would either OOM or evict someone.
    """
    deadline = time.time() + max_wait_hours * 3600
    while True:
        free = _gpu_free_gb(index)
        if free is None:
            print(f"[arch-suite] cannot query GPU {index}; proceeding without a memory check", flush=True)
            return True
        if free >= min_free_gb:
            return True
        if time.time() > deadline:
            return False
        print(
            f"[arch-suite] GPU {index} has {free:.1f} GB free, need {min_free_gb:.1f} GB; "
            f"waiting {poll_seconds}s",
            flush=True,
        )
        time.sleep(poll_seconds)


def _train_command(arm: Arm, seed: int, paths: dict[str, str]) -> list[str]:
    return [
        sys.executable, "scripts/train.py",
        "--model", "rna_methylation",
        "--scope", "chr1",
        "--engine", "matched_chr1",
        "--mode", "final",
        "--recipe", arm.recipe,
        "--seed", str(seed),
        "--run-id", run_id(arm, seed),
        "--canonical-root", paths["canonical_root"],
        "--prepared-root", paths["prepared_root"],
        "--feature-cache", paths["feature_cache"],
        "--rna-cache", paths["rna_cache"],
        "--registry", paths["registry"],
        "--output-root", paths["output_root"],
        *arm.extra_args,
    ]


def _run_dir(paths: dict[str, str], arm: Arm, seed: int) -> Path:
    return Path(paths["output_root"]) / "runs" / "rna_methylation" / "chr1" / run_id(arm, seed)


def dry_run(arms: tuple[Arm, ...]) -> int:
    """Build every selected arm on CPU and assert the zero-init contract.

    Catches config threading mistakes, shape bugs and capacity blowups before any
    GPU hour is spent -- an arm with 50x the canonical parameter count would be
    measuring capacity, not architecture.
    """
    import copy
    import torch
    import yaml
    from methylation_predictor.benchmark.methylprophet.config import load_config
    from methylation_predictor.config import ModelConfig
    from methylation_predictor.models import (
        ArchitectureVariantModel, VarianceNormalizedResidualModel,
        architecture_variant_label, is_architecture_variant,
    )

    base_raw = yaml.safe_load((REPO_ROOT / "configs/benchmark_methylprophet/reference.yaml").read_text())
    scratch = Path(os.environ.get("TMPDIR", "/tmp")) / "arch_suite_dry_run"
    scratch.mkdir(parents=True, exist_ok=True)
    canonical_params = sum(
        p.numel()
        for p in VarianceNormalizedResidualModel(
            25_017, 1536, ModelConfig(variance_normalized_residual=True), epsilon=1e-4
        ).parameters()
    )

    print(f"{'arm':<40} {'stage':<24} {'params':>13} {'vs canon':>9}  topology")
    print(f"{'(canonical reference)':<40} {'':<24} {canonical_params:>13,} {'  +0.0%':>9}")
    failures = 0
    for arm in arms:
        recipe = yaml.safe_load((REPO_ROOT / arm.recipe).read_text())
        base = copy.deepcopy(base_raw)
        base["model"] = copy.deepcopy(recipe["model"])
        base["loss"] = copy.deepcopy(recipe["loss"])
        base["training"] = {**base.get("training", {}), **copy.deepcopy(recipe.get("training", {}))}
        base["tracking"] = {**base.get("tracking", {}), **copy.deepcopy(recipe.get("tracking", {}))}
        resolved = scratch / f"{arm.name}.yaml"
        resolved.write_text(yaml.safe_dump(base, sort_keys=False))
        try:
            cfg = load_config(resolved)
            variant = is_architecture_variant(cfg.model)
            cls = ArchitectureVariantModel if variant else VarianceNormalizedResidualModel
            model = cls(25_017, 1536, cfg.model, epsilon=cfg.data.clip_beta_epsilon).eval()
            rna, loci = torch.randn(3, 25_017), torch.randn(5, 1536)
            prior, sigma = torch.rand(5) * 0.8 + 0.1, torch.rand(5) * 0.4 + 0.1
            kwargs = {"cpg_positions": torch.arange(5) * 91} if getattr(model, "requires_cpg_positions", False) else {}
            with torch.no_grad():
                beta = model(rna, loci, prior, sigma=sigma, **kwargs)["beta"]
            if not torch.allclose(beta, prior[None].expand(3, 5), atol=1e-5):
                raise AssertionError("does not start at the prior (zero-init contract violated)")
            total = sum(p.numel() for p in model.parameters())
            label = architecture_variant_label(cfg.model) if variant else "canonical model class"
            delta = 100.0 * (total - canonical_params) / canonical_params
            print(f"{arm.name:<40} {arm.stage:<24} {total:>13,} {delta:>+8.1f}%  {label}")
        except Exception as exc:  # noqa: BLE001 -- preflight reports, it does not abort the sweep
            failures += 1
            print(f"{arm.name:<40} {arm.stage:<24} {'FAILED':>13}            {type(exc).__name__}: {exc}")
    print()
    print(f"{len(arms)} arm(s), {len(jobs(arms))} run(s), {failures} failure(s)")
    return 1 if failures else 0


def cuda_smoke(arms: tuple[Arm, ...], gpu: int, samples: int, loci: int) -> int:
    """One forward+backward per arm on the GPU, under the real bf16 autocast.

    The --dry-run preflight runs on CPU in float32, which cannot catch the things
    that actually break on device: einsum under bf16 autocast, the float32
    Sinkhorn island inside a bf16 graph, gradient checkpointing under autocast,
    and the axial attention's masked softmax in low precision. Blocks are small,
    so this checks correctness, not the real per-step memory -- run it whenever
    the GPU has a couple of spare GB, before committing to multi-hour runs.
    """
    import copy
    import torch
    import yaml
    from methylation_predictor.benchmark.methylprophet.config import load_config
    from methylation_predictor.losses import residual_loss
    from methylation_predictor.models import (
        ArchitectureVariantModel, VarianceNormalizedResidualModel, is_architecture_variant,
    )

    if not torch.cuda.is_available():
        print("[arch-suite] CUDA not available"); return 1
    device = torch.device("cuda")
    base_raw = yaml.safe_load((REPO_ROOT / "configs/benchmark_methylprophet/reference.yaml").read_text())
    scratch = Path(os.environ.get("TMPDIR", "/tmp")) / "arch_suite_cuda_smoke"
    scratch.mkdir(parents=True, exist_ok=True)

    failures = 0
    for arm in arms:
        recipe = yaml.safe_load((REPO_ROOT / arm.recipe).read_text())
        base = copy.deepcopy(base_raw)
        base["model"] = copy.deepcopy(recipe["model"])
        base["loss"] = copy.deepcopy(recipe["loss"])
        base["training"] = {**base.get("training", {}), **copy.deepcopy(recipe.get("training", {}))}
        resolved = scratch / f"{arm.name}.yaml"
        resolved.write_text(yaml.safe_dump(base, sort_keys=False))
        try:
            cfg = load_config(resolved)
            cls = ArchitectureVariantModel if is_architecture_variant(cfg.model) else VarianceNormalizedResidualModel
            model = cls(25_017, 1536, cfg.model, epsilon=cfg.data.clip_beta_epsilon).to(device).train()
            rna = torch.randn(samples, 25_017, device=device)
            emb = torch.randn(loci, 1536, device=device)
            prior = torch.rand(loci, device=device) * 0.8 + 0.1
            sigma = torch.rand(loci, device=device) * 0.4 + 0.1
            target = torch.rand(samples, loci, device=device)
            target[0, :3] = float("nan")  # missing observations must not poison the backward pass
            kwargs = (
                {"cpg_positions": torch.arange(loci, device=device) * 91}
                if getattr(model, "requires_cpg_positions", False) else {}
            )
            torch.cuda.reset_peak_memory_stats(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(rna, emb, prior, sigma=sigma, **kwargs)
                loss, _ = residual_loss(outputs, target, prior, cfg.loss,
                                        epsilon=cfg.data.clip_beta_epsilon, sigma=sigma)
            if not torch.isfinite(loss):
                raise AssertionError("non-finite loss under bf16 autocast")
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not grads:
                raise AssertionError("no parameter received a gradient")
            bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            if bad:
                raise AssertionError(f"non-finite gradients: {', '.join(bad[:3])}")
            diagnostics = model.diagnostics() if hasattr(model, "diagnostics") else {}
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            loss_value = float(loss.detach())
            print(f"{arm.name:<40} loss={loss_value:>8.4f} peak={peak:>5.2f}GB "
                  f"grads={len(grads):>3} {diagnostics or ''}")
            del model, outputs, loss
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 -- smoke test reports every arm
            failures += 1
            print(f"{arm.name:<40} FAILED  {type(exc).__name__}: {exc}")
            torch.cuda.empty_cache()
    print()
    print(f"{len(arms)} arm(s), {failures} failure(s) "
          f"(blocks were {samples}x{loci}; real array blocks are 512x512, WGBS 32x16384)")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", help="comma-separated arm names (default: all)")
    ap.add_argument("--stages", help=f"comma-separated stages; known: {', '.join(STAGES)}")
    ap.add_argument("--shard", help="i/n -- run this machine's slice of the selected arms")
    ap.add_argument("--gpu", type=int, default=0, help="GPU index (sets CUDA_VISIBLE_DEVICES for the child)")
    ap.add_argument("--data-root", help="overrides METHYL_DATA_ROOT for a machine with a different mount")
    ap.add_argument("--min-free-gb", type=float, default=20.0,
                    help="wait until the GPU has this much free memory before starting a run")
    ap.add_argument("--max-wait-hours", type=float, default=24.0)
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true", help="CPU preflight only: build every arm, print sizes")
    ap.add_argument("--cuda-smoke", action="store_true",
                    help="one small forward+backward per arm on the GPU under real bf16 autocast")
    ap.add_argument("--smoke-samples", type=int, default=8)
    ap.add_argument("--smoke-loci", type=int, default=256)
    ap.add_argument("--print-commands", action="store_true", help="print the train.py commands and exit")
    ap.add_argument("--force", action="store_true", help="re-run units that already completed")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    arms = select(args.arms, args.stages, args.shard)
    if not arms:
        print("no arms selected"); return 1
    if args.dry_run:
        return dry_run(arms)
    if args.cuda_smoke:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        return cuda_smoke(arms, args.gpu, args.smoke_samples, args.smoke_loci)

    paths = data_paths(args.data_root)
    units = jobs(arms)
    if args.print_commands:
        for arm, seed in units:
            print(shlex.join(_train_command(arm, seed, paths)))
        return 0

    host = socket.gethostname()
    log_dir = REPO_ROOT / "logs" / "arch_suite"
    log_dir.mkdir(parents=True, exist_ok=True)
    state_path = log_dir / f"queue_state_{host}.json"
    state: dict[str, dict] = json.loads(state_path.read_text()) if state_path.is_file() else {}

    print(f"[arch-suite] host={host} gpu={args.gpu} arms={len(arms)} runs={len(units)}", flush=True)
    for arm, seed in units:
        unit = run_id(arm, seed)
        run_dir = _run_dir(paths, arm, seed)
        if (run_dir / ".done").is_file() and not args.force:
            print(f"[arch-suite] {unit}: already complete, skipping", flush=True)
            state[unit] = {**state.get(unit, {}), "status": "already_complete", "run_dir": str(run_dir)}
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            continue

        if not _wait_for_gpu(args.gpu, args.min_free_gb, args.max_wait_hours, args.poll_seconds):
            print(f"[arch-suite] {unit}: GPU never freed up within --max-wait-hours; stopping", flush=True)
            state[unit] = {"status": "gpu_unavailable", "host": host}
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            return 2

        command = _train_command(arm, seed, paths)
        if run_dir.is_dir():
            # RunStore refuses to reopen an existing run directory without --resume,
            # so a unit killed mid-training would otherwise fail forever. Resume when
            # there is a checkpoint to resume from; otherwise stop and say so rather
            # than deleting someone's partial run directory.
            if (run_dir / "checkpoints" / "latest.pt").is_file():
                print(f"[arch-suite] {unit}: incomplete run found, resuming", flush=True)
                command = command + ["--resume"]
            else:
                print(
                    f"[arch-suite] {unit}: run directory exists with no checkpoints/latest.pt "
                    f"(a failed start). Inspect and remove {run_dir} to retry; skipping.",
                    flush=True,
                )
                state[unit] = {**state.get(unit, {}), "status": "blocked_incomplete_run_dir",
                               "run_dir": str(run_dir), "host": host}
                state_path.write_text(json.dumps(state, indent=2) + "\n")
                continue
        log_path = log_dir / f"{unit}.log"
        started = datetime.now(timezone.utc)
        state[unit] = {
            "status": "running", "host": host, "platform": platform.platform(), "gpu": args.gpu,
            "arm": arm.name, "stage": arm.stage, "seed": seed, "recipe": arm.recipe,
            "run_dir": str(run_dir), "log": str(log_path),
            "command": shlex.join(command), "started_at_utc": started.isoformat(),
        }
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        print(f"[arch-suite] {unit}: starting -> {log_path}", flush=True)

        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONPATH": str(REPO_ROOT / "src")}
        with log_path.open("a") as handle:
            handle.write(f"\n=== {started.isoformat()} {host} :: {shlex.join(command)}\n")
            handle.flush()
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=REPO_ROOT)

        finished = datetime.now(timezone.utc)
        state[unit].update({
            "status": "completed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "finished_at_utc": finished.isoformat(),
            "wall_seconds": (finished - started).total_seconds(),
        })
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        print(
            f"[arch-suite] {unit}: {state[unit]['status']} in "
            f"{state[unit]['wall_seconds'] / 3600:.2f} h (rc={result.returncode})",
            flush=True,
        )

    print(f"[arch-suite] shard finished; state -> {state_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
