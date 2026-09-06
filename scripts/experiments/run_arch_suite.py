#!/usr/bin/env python3
"""Queue runner for the architecture-novelty suite (architecture_novelty_2026_09),
targeting the primary shared-backbone architecture (retargeted 2026-09-04).

Not part of the stable CLI -- delete alongside scripts/experiments/arch_suite.py
once the study concludes (CLAUDE.md: one-off experiment scripts are deleted, not
kept as a second workflow).

Two differences from the retired two-stage suite's runner, both consequences of
how ``matched_chr1_shared_backbone`` works:

  * Training and evaluation are SEPARATE steps here (``scripts/train.py`` only
    trains; the official-split number comes from a second
    ``scripts/evaluate.py --engine matched_chr1_shared_backbone`` call against
    the checkpoint it produced). Each unit of work is therefore two subprocess
    calls, run back to back.
  * incomplete runs with a valid ``checkpoints/last.pt`` are resumed by passing
    ``--resume`` to the shared-backbone trainer. Completed runs can optionally
    have their evaluation refreshed with ``--refresh-evaluation``.

    # preflight, no GPU: build every arm on CPU and check its shapes/finiteness
    python scripts/experiments/run_arch_suite.py --dry-run

    # machine 1 of 3
    python scripts/experiments/run_arch_suite.py --shard 1/3 --gpu 0

Run --stages 0-noise-floor first: it is both the seed-SD floor every other arm
is judged against AND the missing convergence run for the project's own primary-
architecture reference number (rung_b_official_final was stopped by user request
at epoch 47/80, not converged -- see results/reference/ablations.yaml).
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

from arch_suite import ARMS_BY_NAME, ENGINE, REPO_ROOT, STAGES, Arm, data_paths, jobs, run_id, select  # noqa: E402


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
        print(f"[arch-suite] GPU {index} has {free:.1f} GB free, need {min_free_gb:.1f} GB; "
              f"waiting {poll_seconds}s", flush=True)
        time.sleep(poll_seconds)


def _run_dir(paths: dict[str, str], arm: Arm, seed: int) -> Path:
    return Path(paths["output_root"]) / "runs" / "locus_cls_joint" / "chr1" / run_id(arm, seed)


def _eval_output(run_dir: Path) -> Path:
    return run_dir / "evaluation" / "chr1" / "metrics.json"


def _train_command(arm: Arm, seed: int, paths: dict[str, str], *, resume: bool = False) -> list[str]:
    command = [
        sys.executable, "scripts/train.py",
        "--model", "rna_methylation", "--scope", "chr1", "--engine", ENGINE, "--mode", "final",
        "--recipe", arm.recipe, "--seed", str(seed), "--run-id", run_id(arm, seed),
        "--canonical-root", paths["canonical_root"], "--prepared-root", paths["prepared_root"],
        "--feature-cache", paths["feature_cache"], "--rna-cache", paths["rna_cache"],
        "--registry", paths["registry"], "--cpg-targets-dir", paths["cpg_targets_dir"],
        "--output-root", paths["output_root"], *arm.extra_args,
    ]
    if resume:
        command.append("--resume")
    return command


def _eval_command(arm: Arm, seed: int, paths: dict[str, str], run_dir: Path) -> list[str]:
    checkpoint = run_dir / "checkpoints" / "best.pt"
    return [
        sys.executable, "scripts/evaluate.py",
        "--model", "rna_methylation", "--engine", ENGINE,
        "--checkpoint", str(checkpoint), "--eval-scope", "chr1",
        "--output", str(_eval_output(run_dir)),
        "--recipe", arm.recipe,
        "--canonical-root", paths["canonical_root"], "--prepared-root", paths["prepared_root"],
        "--feature-cache", paths["feature_cache"], "--rna-cache", paths["rna_cache"],
        "--registry", paths["registry"], "--cpg-targets-dir", paths["cpg_targets_dir"],
        # Must mirror _train_command's extra_args -- an arm that overrides
        # --rna-cache (e.g. enc_frozen_embedding_bulkrnabert) needs the same
        # override at eval time or evaluate_official_split rebuilds the model
        # against the wrong (default 25017-gene) cache and the checkpoint's
        # state_dict fails to load with a shape mismatch (found 2026-09-06:
        # this was the first arm to ever use extra_args, so the gap never
        # manifested before).
        *arm.extra_args,
    ]


def _is_complete(run_dir: Path) -> bool:
    return (
        (run_dir / "training" / "summary.json").is_file()
        and (run_dir / "checkpoints" / "best.pt").is_file()
        and _eval_output(run_dir).is_file()
    )


def dry_run(arms: tuple[Arm, ...]) -> int:
    """Build every selected arm on CPU: check shapes, finiteness, and the
    fusion_init_std=0 -> exactly-0.5 invariant, before spending any GPU hour."""
    import torch

    from methylation_predictor.rna_training.config import load_rna_recipe
    from methylation_predictor.models import (
        FeatureFusionArchitectureVariantModel, FeatureFusionLocusCLSModel,
        feature_fusion_variant_label, is_architecture_variant,
    )

    reference = load_rna_recipe(REPO_ROOT / "configs/models/rna_methylation_shared_backbone.yaml")
    reference_model = FeatureFusionLocusCLSModel(25_017, 1536, reference.model, trunk_hidden_dim=256, bottleneck_dim=64)
    reference_params = sum(p.numel() for p in reference_model.parameters())

    print(f"{'arm':<42} {'stage':<24} {'params':>13} {'vs reference':>13}  topology")
    print(f"{'(reference: rung_b)':<42} {'':<24} {reference_params:>13,} {'+0.0%':>13}")
    failures = 0
    for arm in arms:
        try:
            recipe = load_rna_recipe(REPO_ROOT / arm.recipe)
            lc = recipe.raw.get("locus_cls", {})
            variant = is_architecture_variant(recipe.model)
            cls = FeatureFusionArchitectureVariantModel if variant else FeatureFusionLocusCLSModel
            model = cls(
                25_017, 1536, recipe.model,
                trunk_hidden_dim=lc.get("trunk_hidden_dim", 256), bottleneck_dim=lc.get("bottleneck_dim", 64),
                trunk_dropout=lc.get("trunk_dropout", 0.1), use_mean_branch=lc.get("use_mean_branch", True),
                use_fusion_product=lc.get("use_fusion_product", False),
                fusion_init_std=0.0,  # exact invariant check: must predict 0.5 everywhere
            ).eval()
            rna, loci = torch.randn(3, 25_017), torch.randn(5, 1536)
            kwargs = {"cpg_positions": torch.arange(5) * 91} if getattr(model, "requires_cpg_positions", False) else {}
            with torch.no_grad():
                beta = model(rna, loci, **kwargs)["beta"]
            if not torch.allclose(beta, torch.full_like(beta, 0.5), atol=1e-5):
                raise AssertionError("fusion_init_std=0 does not predict exactly 0.5 (zero-init contract violated)")
            total = sum(p.numel() for p in model.parameters())
            label = feature_fusion_variant_label(recipe.model) if variant else "reference model class"
            delta = 100.0 * (total - reference_params) / reference_params
            print(f"{arm.name:<42} {arm.stage:<24} {total:>13,} {delta:>+12.1f}%  {label}")
        except Exception as exc:  # noqa: BLE001 -- preflight reports, does not abort the sweep
            failures += 1
            print(f"{arm.name:<42} {arm.stage:<24} {'FAILED':>13} {'':>13}  {type(exc).__name__}: {exc}")
    print()
    print(f"{len(arms)} arm(s), {len(jobs(arms))} run(s), {failures} failure(s)")
    return 1 if failures else 0


def cuda_smoke(arms: tuple[Arm, ...], gpu: int, samples: int, loci: int) -> int:
    """One forward+backward per arm on the GPU under real bf16 autocast, plus
    the auxiliary mean/residual losses LocusCLSJointTrainer actually applies --
    catches what the CPU/float32 dry run cannot (bf16 einsum, the float32
    Sinkhorn island inside a bf16 graph, gradient checkpointing under autocast)."""
    import torch

    from methylation_predictor.losses import beta_nll_term, locus_correlation_loss, masked_mean
    from methylation_predictor.rna_training.config import load_rna_recipe
    from methylation_predictor.models import (
        FeatureFusionArchitectureVariantModel, FeatureFusionLocusCLSModel, is_architecture_variant,
    )

    if not torch.cuda.is_available():
        print("[arch-suite] CUDA not available"); return 1
    device = torch.device("cuda")
    failures = 0
    for arm in arms:
        try:
            recipe = load_rna_recipe(REPO_ROOT / arm.recipe)
            lc = recipe.raw.get("locus_cls", {})
            variant = is_architecture_variant(recipe.model)
            cls = FeatureFusionArchitectureVariantModel if variant else FeatureFusionLocusCLSModel
            model = cls(
                25_017, 1536, recipe.model,
                trunk_hidden_dim=lc.get("trunk_hidden_dim", 256), bottleneck_dim=lc.get("bottleneck_dim", 64),
                trunk_dropout=lc.get("trunk_dropout", 0.1), use_mean_branch=lc.get("use_mean_branch", True),
                use_fusion_product=lc.get("use_fusion_product", False),
                fusion_init_std=lc.get("fusion_init_std", 0.01),
            ).to(device).train()
            rna = torch.randn(samples, 25_017, device=device)
            emb = torch.randn(loci, 1536, device=device)
            target = torch.rand(samples, loci, device=device)
            target[0, :3] = float("nan")
            kwargs = (
                {"cpg_positions": torch.arange(loci, device=device) * 91}
                if getattr(model, "requires_cpg_positions", False) else {}
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(rna, emb, **kwargs)
                mask = torch.isfinite(target)
                safe_target = torch.where(mask, target, torch.zeros_like(target))
                beta_mse = masked_mean((out["beta"] - safe_target) ** 2, mask)
                pearson, _ = locus_correlation_loss(out["beta"], safe_target, mask, recipe.loss)
                nll = beta_nll_term(out, safe_target, mask, recipe.loss)
                loss = (
                    recipe.loss.beta_mse_weight * beta_mse
                    + recipe.loss.locus_pearson_weight * pearson
                    + recipe.loss.beta_nll_weight * nll
                )
                if lc.get("use_mean_branch", True) and out["mu_logit"] is not None:
                    loss = loss + 0.15 * out["mu_logit"].pow(2).mean()  # stand-in aux target, gradient-flow check only
                if out["residual_logit"] is not None:
                    loss = loss + 0.0 * out["residual_logit"].sum()  # exercised at weight 0 in the reference recipe
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
            print(f"{arm.name:<42} loss={float(loss.detach()):>8.4f} peak={peak:>5.2f}GB "
                  f"grads={len(grads):>3} {diagnostics or ''}")
            del model, out, loss
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"{arm.name:<42} FAILED  {type(exc).__name__}: {exc}")
            torch.cuda.empty_cache()
    print()
    print(f"{len(arms)} arm(s), {failures} failure(s) "
          f"(blocks were {samples}x{loci}; real array blocks are 640x640, WGBS 32x20480)")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", help="comma-separated arm names (default: all)")
    ap.add_argument("--stages", help=f"comma-separated stages; known: {', '.join(STAGES)}")
    ap.add_argument("--shard", help="i/n -- run this machine's slice of the selected arms")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--data-root", help="overrides METHYL_DATA_ROOT for a machine with a different mount")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--max-wait-hours", type=float, default=24.0)
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true", help="CPU preflight only: build every arm, print sizes")
    ap.add_argument("--cuda-smoke", action="store_true",
                    help="one small forward+backward per arm on the GPU under real bf16 autocast")
    ap.add_argument("--smoke-samples", type=int, default=8)
    ap.add_argument("--smoke-loci", type=int, default=256)
    ap.add_argument("--print-commands", action="store_true")
    ap.add_argument("--refresh-evaluation", action="store_true",
                    help="rerun evaluation for completed units and log all official views")
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
            run_dir = _run_dir(paths, arm, seed)
            print(shlex.join(_train_command(arm, seed, paths)))
            print(shlex.join(_eval_command(arm, seed, paths, run_dir)))
        return 0

    host = socket.gethostname()
    log_dir = REPO_ROOT / "logs" / "arch_suite"
    log_dir.mkdir(parents=True, exist_ok=True)
    state_path = log_dir / f"queue_state_{host}.json"
    state: dict[str, dict] = json.loads(state_path.read_text()) if state_path.is_file() else {}

    print(f"[arch-suite] host={host} gpu={args.gpu} engine={ENGINE} arms={len(arms)} runs={len(units)}", flush=True)
    for arm, seed in units:
        unit = run_id(arm, seed)
        run_dir = _run_dir(paths, arm, seed)
        if _is_complete(run_dir) and not args.refresh_evaluation:
            print(f"[arch-suite] {unit}: already complete, skipping", flush=True)
            state[unit] = {**state.get(unit, {}), "status": "already_complete", "run_dir": str(run_dir)}
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            continue
        refresh_only = _is_complete(run_dir) and args.refresh_evaluation
        resume_train = run_dir.is_dir() and not refresh_only
        if resume_train and not (run_dir / "checkpoints" / "last.pt").is_file():
            print(f"[arch-suite] {unit}: incomplete run has no checkpoints/last.pt; skipping", flush=True)
            state[unit] = {**state.get(unit, {}), "status": "blocked_incomplete_run_dir",
                           "run_dir": str(run_dir), "host": host}
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            continue

        if not _wait_for_gpu(args.gpu, args.min_free_gb, args.max_wait_hours, args.poll_seconds):
            print(f"[arch-suite] {unit}: GPU never freed up within --max-wait-hours; stopping", flush=True)
            state[unit] = {"status": "gpu_unavailable", "host": host}
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            return 2

        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONPATH": str(REPO_ROOT / "src")}
        started = datetime.now(timezone.utc)
        state[unit] = {
            "status": "training" if not refresh_only else "evaluating", "host": host, "platform": platform.platform(), "gpu": args.gpu,
            "arm": arm.name, "stage": arm.stage, "seed": seed, "recipe": arm.recipe,
            "run_dir": str(run_dir), "started_at_utc": started.isoformat(),
        }
        state_path.write_text(json.dumps(state, indent=2) + "\n")

        if not refresh_only:
            train_log = log_dir / f"{unit}.train.log"
            print(f"[arch-suite] {unit}: {'resuming' if resume_train else 'training'} -> {train_log}", flush=True)
            train_cmd = _train_command(arm, seed, paths, resume=resume_train)
            with train_log.open("a") as handle:
                handle.write(f"\n=== {started.isoformat()} {host} :: {shlex.join(train_cmd)}\n")
                handle.flush()
                train_result = subprocess.run(
                    train_cmd, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=REPO_ROOT,
                )
            if train_result.returncode != 0:
                state[unit].update({"status": "train_failed", "train_returncode": train_result.returncode})
                state_path.write_text(json.dumps(state, indent=2) + "\n")
                print(f"[arch-suite] {unit}: training FAILED (rc={train_result.returncode}), see {train_log}", flush=True)
                continue

        eval_log = log_dir / f"{unit}.eval.log"
        print(f"[arch-suite] {unit}: evaluating -> {eval_log}", flush=True)
        (run_dir / "evaluation" / "chr1").mkdir(parents=True, exist_ok=True)
        with eval_log.open("a") as handle:
            eval_cmd = _eval_command(arm, seed, paths, run_dir)
            handle.write(f"\n=== {datetime.now(timezone.utc).isoformat()} {host} :: {shlex.join(eval_cmd)}\n")
            handle.flush()
            eval_result = subprocess.run(eval_cmd, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=REPO_ROOT)

        finished = datetime.now(timezone.utc)
        state[unit].update({
            "status": "completed" if eval_result.returncode == 0 else "eval_failed",
            "eval_returncode": eval_result.returncode,
            "finished_at_utc": finished.isoformat(),
            "wall_seconds": (finished - started).total_seconds(),
        })
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        print(f"[arch-suite] {unit}: {state[unit]['status']} in "
              f"{state[unit]['wall_seconds'] / 3600:.2f} h", flush=True)

    print(f"[arch-suite] shard finished; state -> {state_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
