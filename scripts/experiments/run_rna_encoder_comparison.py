#!/usr/bin/env python3
"""Run RNA-encoder comparators (gene-pathway, frozen BulkFormer, MethylProphet
BottleneckMLP) sequentially.

Default: chr1, development mode on the same fixed split seed=17 used for
architecture selection. Use --mode final only after the architecture is
frozen; final mode also runs scripts/evaluate.py on all three official views
for the chosen scope. --scope chr123 reuses the same recipes and the same
(scope-independent) canonical-RNA/BulkFormer sample caches, but trains
through --engine shared_backbone against the chr123 compact cache (see
scripts/prepare_chr123_compact.py) instead of the chr1-matched cache.

enc_bottleneck_mlp already has a confirmed_official chr1 number produced via
the separate arch_suite.py/run_arch_suite.py queue runner (see
results/reference/ours/02_rna_encoder_comparison.yaml) -- it's included here
only so chr123 can reuse this harness instead of extending that chr1-only
suite.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
ARMS = {
    "gene_pathway": {
        "recipe": "configs/models/rna_encoder_comparison/enc_gene_pathway.yaml",
        "cache": "canonical",
        "membership": REPO_ROOT / "resources/pathways/survpath_combine_tcga.npz",
    },
    "bulkformer_147m": {
        "recipe": "configs/models/rna_encoder_comparison/enc_frozen_bulkformer_147m.yaml",
        "cache": "bulkformer",
        "membership": None,
    },
    "enc_bottleneck_mlp": {
        "recipe": "configs/models/arch_shared_backbone/enc_bottleneck_mlp.yaml",
        "cache": "canonical",
        "membership": None,
    },
}


def resolve_root(value: str | None) -> Path:
    if value: return Path(value).resolve()
    if os.environ.get("METHYL_DATA_ROOT"): return Path(os.environ["METHYL_DATA_ROOT"]).resolve()
    local = REPO_ROOT / "MethylPredictionData"
    if local.is_dir(): return local.resolve()
    return Path("/dune/DATASETS/MethylPredictionData")


SCOPE_ENGINE = {"chr1": "matched_chr1_shared_backbone", "chr123": "shared_backbone"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default="gene_pathway,bulkformer_147m")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--data-root")
    ap.add_argument("--scope", choices=sorted(SCOPE_ENGINE), default="chr1")
    ap.add_argument("--mode", choices=["development", "final"], default="development")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--early-stop-patience", type=int, default=6)
    args = ap.parse_args()

    selected = [x.strip() for x in args.arms.split(",") if x.strip()]
    unknown = [x for x in selected if x not in ARMS]
    if unknown: raise SystemExit(f"unknown arms {unknown}; known={list(ARMS)}")

    root = resolve_root(args.data_root); os.chdir(REPO_ROOT)
    canonical = root / "datasets/methylprophet_repro_v1"
    chr1_prepared = root / "derived/methylprophet_table5_tcga_chr1"
    engine = SCOPE_ENGINE[args.scope]
    if args.scope == "chr1":
        prepared = chr1_prepared; feature_cache = prepared / "features"
    else:
        prepared = root / "derived/methylprophet_compact_chr123"
        feature_cache = root / "derived/rna_feature_cache/chr123"
        if not prepared.is_dir() or not feature_cache.is_dir():
            print(f"[rna-encoder] missing chr123 cache ({prepared} / {feature_cache})\nBuild it first:\n  python scripts/prepare_chr123_compact.py --canonical-root {canonical} --output {prepared}")
            return 2
    # The canonical (non-frozen) RNA z-score cache is per-sample, not per-CpG-scope,
    # so it's built once under the chr1 prep and reused unchanged for chr123 training
    # (same pattern as docs/PAPER_ROADMAP.md's chr123-ref run); BulkFormer's cache is
    # equally scope-independent.
    canonical_rna = chr1_prepared / "rna"
    bulkformer = root / "derived/bulkformer_embeddings/tcga_147m"
    registry = canonical / "cpg/registries/array_cpg_map.parquet"
    cpg_targets = root / f"derived/cpg_statistics/{args.scope}"; output_root = root / "experiments"
    log_dir = REPO_ROOT / "logs/rna_encoder_comparison_2026_09"; log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONPATH": str(REPO_ROOT / "src")}

    for arm in selected:
        spec = ARMS[arm]
        if spec["membership"] is not None and not spec["membership"].is_file():
            print(f"[rna-encoder] missing {spec['membership']}\nRun:\n  {sys.executable} scripts/prepare_gene_pathway_membership.py --canonical-root {canonical} --output {spec['membership']}")
            return 2
        rna_cache = canonical_rna if spec["cache"] == "canonical" else bulkformer
        if not (rna_cache / "rna_sample_idx.npy").is_file() or not (rna_cache / "rna_zscore.f16.npy").is_file():
            if arm == "bulkformer_147m":
                print(f"[rna-encoder] missing BulkFormer cache {rna_cache}\nPrepare it first, e.g.:\n  conda run -n bulkformer python scripts/prepare_bulkformer_embeddings.py --canonical-root {canonical} --output {rna_cache} --resume")
            else: print(f"[rna-encoder] missing RNA cache {rna_cache}")
            return 2

        run_id = f"rnaenc-{arm}-{args.mode}-seed{args.seed}"
        run_dir = output_root / "runs/locus_cls_joint" / args.scope / run_id
        summary = run_dir / "training/summary.json"; best = run_dir / "checkpoints/best.pt"
        eval_out = run_dir / "evaluation" / args.scope / "metrics.json"
        complete = summary.is_file() and best.is_file() and (args.mode == "development" or eval_out.is_file())
        if complete:
            print(f"[rna-encoder] {arm}: already complete, skipping", flush=True); continue
        resume = run_dir.is_dir() and (run_dir / "checkpoints/last.pt").is_file() and not summary.is_file()
        # A crash before the first checkpoint (e.g. an error on the very first
        # training step) leaves an empty run dir that neither "already
        # complete" nor "resume" covers: RunStore.create() refuses to reopen
        # it without resume=True, but the trainer itself refuses --resume
        # without an actual checkpoints/last.pt. Clear it so the next attempt
        # starts clean instead of failing on both checks forever.
        if run_dir.is_dir() and not resume and not summary.is_file():
            shutil.rmtree(run_dir)

        cmd = [
            sys.executable, "scripts/train.py", "--model", "rna_methylation", "--scope", args.scope,
            "--engine", engine, "--mode", args.mode,
            "--recipe", spec["recipe"], "--seed", str(args.seed),
            "--canonical-root", str(canonical), "--prepared-root", str(prepared),
            "--feature-cache", str(feature_cache), "--rna-cache", str(rna_cache),
            "--registry", str(registry), "--cpg-targets-dir", str(cpg_targets),
            "--output-root", str(output_root), "--run-id", run_id,
        ]
        if args.mode == "development":
            cmd += ["--early-stop-patience", str(args.early_stop_patience)]
        if resume: cmd.append("--resume")
        if not summary.is_file():
            log = log_dir / f"{args.scope}-{run_id}.train.log"; print(f"[rna-encoder] {arm}: {'resuming' if resume else 'training'} -> {log}", flush=True)
            with log.open("a") as h:
                h.write("\n=== " + shlex.join(cmd) + "\n"); h.flush()
                rc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=h, stderr=subprocess.STDOUT).returncode
            if rc:
                print(f"[rna-encoder] {arm}: FAILED rc={rc}; sequence stopped", flush=True); return rc

        if args.mode == "final":
            eval_out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, "scripts/evaluate.py", "--model", "rna_methylation",
                "--engine", engine, "--checkpoint", str(best),
                "--eval-scope", args.scope, "--output", str(eval_out), "--recipe", spec["recipe"],
                "--canonical-root", str(canonical), "--prepared-root", str(prepared),
                "--feature-cache", str(feature_cache), "--rna-cache", str(rna_cache),
                "--registry", str(registry), "--cpg-targets-dir", str(cpg_targets),
            ]
            log = log_dir / f"{args.scope}-{run_id}.eval.log"; print(f"[rna-encoder] {arm}: official evaluation -> {log}", flush=True)
            with log.open("a") as h:
                h.write("\n=== " + shlex.join(cmd) + "\n"); h.flush()
                rc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=h, stderr=subprocess.STDOUT).returncode
            if rc:
                print(f"[rna-encoder] {arm}: eval FAILED rc={rc}; sequence stopped", flush=True); return rc

    subprocess.run([sys.executable, "scripts/experiments/collect_rna_encoder_comparison.py", "--data-root", str(root), "--scope", args.scope, "--mode", args.mode, "--seed", str(args.seed)], cwd=REPO_ROOT, check=False)
    print("[rna-encoder] sequence complete", flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
