#!/usr/bin/env python3
"""Launch isolated functional-only ladder runs through the existing trainer."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ATLAS = "/home/vcivale/dune_data/derived/ntv3_functional_peak_atlas_chr1_all_sources"
ANNOTATION = "/home/vcivale/dune_data/derived/ntv3_probe_targets/chr1_annotation_features_all_sources"

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", default="f0,f1,f2,f3,f4,f5")
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--seed", type=int, default=17)
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--resume", action="store_true")
    p.add_argument("--output-root", required=True); p.add_argument("--canonical-root", required=True)
    p.add_argument("--feature-cache", required=True); p.add_argument("--rna-cache", required=True)
    p.add_argument("--registry", required=True); p.add_argument("--prepared-root", required=True)
    p.add_argument("--cpg-targets-dir", required=True)
    p.add_argument("--functional-atlas", default=ATLAS); p.add_argument("--annotation-cache", default=ANNOTATION)
    args = p.parse_args()
    variants = [x.strip().lower() for x in args.variants.split(",") if x.strip()]
    allowed = {f"f{i}" for i in range(6)}
    if not variants or not set(variants) <= allowed:
        p.error("--variants accepts f0,...,f5; F6 is deliberately not automatic")
    base = ROOT / "configs/models/functional_fusion/base.yaml"
    print(json.dumps({"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                      "base_recipe_sha256": sha256(base), "functional_atlas": args.functional_atlas,
                      "annotation_cache": args.annotation_cache}, indent=2))
    for variant in variants:
        recipe = ROOT / f"configs/models/functional_fusion/{variant}.yaml"
        run_id = f"functional-fusion-{variant}-screen-seed{args.seed}-e{args.epochs}"
        run_dir = Path(args.output_root) / "runs" / "locus_cls_joint" / "chr1" / run_id
        if run_dir.exists() and not args.resume:
            raise FileExistsError(f"refusing to overwrite {run_dir}; use --resume for an interrupted run")
        cmd = [sys.executable, str(ROOT / "scripts/train.py"), "--model", "rna_methylation",
               "--scope", "chr1", "--engine", "matched_chr1_shared_backbone", "--mode", "development",
               "--recipe", str(recipe), "--output-root", args.output_root, "--run-id", run_id,
               "--epochs", str(args.epochs), "--seed", str(args.seed), "--early-stop-patience", "8",
               "--development-split-seed", "17", "--canonical-root", args.canonical_root,
               "--feature-cache", args.feature_cache, "--rna-cache", args.rna_cache,
               "--registry", args.registry, "--prepared-root", args.prepared_root,
               "--cpg-targets-dir", args.cpg_targets_dir, "--functional-atlas", args.functional_atlas,
               "--annotation-cache", args.annotation_cache, "--functional-only"]
        if args.resume: cmd.append("--resume")
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)
            summary = json.loads((run_dir / "training/summary.json").read_text())
            plan_hash = summary["epoch1_plan_sha256"]
            if "expected_plan_hash" not in locals():
                expected_plan_hash = plan_hash
            elif plan_hash != expected_plan_hash:
                raise RuntimeError(f"epoch-1 source plan differs: {variant}={plan_hash}, expected {expected_plan_hash}")

if __name__ == "__main__": main()
