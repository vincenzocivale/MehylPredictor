#!/usr/bin/env python3
"""Launch the G0-G4 regulatory-fusion ladder through the existing trainer,
one variant after another on one GPU, from scratch, fixed epochs (default
20 -- reduced from the protocol's original 40 after G0's own partial run
showed val MAS-PCC peaking ~epoch 12 and val MSE rising past ~epoch 8), no
early stopping. See docs/RNA_METHYLATION.md's regulatory-fusion ladder
section and FUNCTIONAL_FUSION_LADDER_REPORT.md (the F0-F6 predecessor).

Meant to be launched as a single persistent background process that
survives SSH disconnection, e.g.:

  nohup python scripts/experiments/run_regulatory_fusion_ladder.py \\
      --variants g0,g1,g2,g3,g4 --epochs 20 --seed 17 \\
      --output-root ... --canonical-root ... --feature-cache ... \\
      --rna-cache ... --registry ... --prepared-root ... \\
      --cpg-targets-dir ... \\
      > logs/regulatory_fusion_ladder.log 2>&1 &

Runs strictly sequentially (never two variants on one GPU at once). If a run
fails, its traceback is saved and the sequence stops -- it does not silently
continue to the next variant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ATLAS = "/dune/DATASETS/MethylPredictionData/derived/ntv3_functional_peak_atlas_chr1_all_sources"
ANNOTATION = "/dune/DATASETS/MethylPredictionData/derived/ntv3_probe_targets/chr1_annotation_features_all_sources"
VARIANT_LABEL = {
    "g0": "g0_control", "g1": "g1_grouped", "g2": "g2_regulatory_head",
    "g3": "g3_gated_residual", "g4": "g4_lowrank_fusion",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", default="g0,g1,g2,g3,g4")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true", help="resume an interrupted run of the SAME variant only")
    p.add_argument("--output-root", required=True)
    p.add_argument("--canonical-root", required=True)
    p.add_argument("--feature-cache", required=True)
    p.add_argument("--rna-cache", required=True)
    p.add_argument("--registry", required=True)
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--cpg-targets-dir", required=True)
    p.add_argument("--functional-atlas", default=ATLAS)
    p.add_argument("--annotation-cache", default=ANNOTATION)
    p.add_argument("--log-dir", default=str(ROOT / "logs" / "regulatory_fusion_ladder"))
    args = p.parse_args()

    variants = [x.strip().lower() for x in args.variants.split(",") if x.strip()]
    allowed = set(VARIANT_LABEL)
    if not variants or not set(variants) <= allowed:
        p.error(f"--variants accepts a subset of {sorted(allowed)}")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    base = ROOT / "configs/models/functional_fusion/base.yaml"
    provenance = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "base_recipe_sha256": sha256(base),
        "functional_atlas": args.functional_atlas,
        "annotation_cache": args.annotation_cache,
        "seed": args.seed, "epochs": args.epochs,
        "development_split_seed": 17,
        "variants": variants,
    }
    print(json.dumps(provenance, indent=2), flush=True)
    (log_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    expected_plan_hash: str | None = None
    for variant in variants:
        recipe = ROOT / f"configs/models/functional_fusion/{variant}.yaml"
        run_id = f"regulatory-fusion-{variant}-ladder-seed{args.seed}-e{args.epochs}"
        run_dir = Path(args.output_root) / "runs" / "locus_cls_joint" / "chr1" / run_id
        if run_dir.exists() and not args.resume:
            raise FileExistsError(f"refusing to overwrite {run_dir}; use --resume for an interrupted run of {variant}")
        cmd = [
            sys.executable, str(ROOT / "scripts/train.py"), "--model", "rna_methylation",
            "--scope", "chr1", "--engine", "matched_chr1_shared_backbone", "--mode", "development",
            "--recipe", str(recipe), "--output-root", args.output_root, "--run-id", run_id,
            "--epochs", str(args.epochs), "--seed", str(args.seed),
            # early-stop-patience deliberately not passed: early stopping stays disabled.
            "--development-split-seed", "17", "--canonical-root", args.canonical_root,
            "--feature-cache", args.feature_cache, "--rna-cache", args.rna_cache,
            "--registry", args.registry, "--prepared-root", args.prepared_root,
            "--cpg-targets-dir", args.cpg_targets_dir, "--functional-atlas", args.functional_atlas,
            "--annotation-cache", args.annotation_cache, "--functional-only",
        ]
        if args.resume:
            cmd.append("--resume")
        print(" ".join(cmd), flush=True)
        variant_log = log_dir / f"{variant}.log"
        if args.dry_run:
            continue
        with open(variant_log, "a") as log_fh:
            log_fh.write(f"\n=== launch: {' '.join(cmd)} ===\n")
            log_fh.flush()
            try:
                subprocess.run(cmd, cwd=ROOT, check=True, stdout=log_fh, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as exc:
                traceback_path = log_dir / f"{variant}.FAILED.traceback.txt"
                traceback_path.write_text(
                    f"variant={variant} run_id={run_id} returncode={exc.returncode}\n"
                    f"cmd={' '.join(cmd)}\n\nSee {variant_log} for full stdout/stderr.\n"
                    + traceback.format_exc()
                )
                print(f"[regulatory-fusion-ladder] {variant} FAILED (rc={exc.returncode}); "
                      f"traceback saved to {traceback_path}; stopping sequence.", file=sys.stderr, flush=True)
                raise SystemExit(1) from exc
        summary_path = run_dir / "training/summary.json"
        if not summary_path.exists():
            print(f"[regulatory-fusion-ladder] {variant} produced no summary.json at {summary_path}; stopping.",
                  file=sys.stderr, flush=True)
            raise SystemExit(1)
        summary = json.loads(summary_path.read_text())
        plan_hash = summary.get("epoch1_plan_sha256")
        if expected_plan_hash is None:
            expected_plan_hash = plan_hash
        elif plan_hash != expected_plan_hash:
            print(f"[regulatory-fusion-ladder] epoch-1 source plan differs: {variant}={plan_hash}, "
                  f"expected {expected_plan_hash}; stopping.", file=sys.stderr, flush=True)
            raise SystemExit(1)
        print(f"[regulatory-fusion-ladder] {variant} done: run_dir={run_dir} plan_hash={plan_hash}", flush=True)

    print("[regulatory-fusion-ladder] all requested variants completed.", flush=True)


if __name__ == "__main__":
    main()
