#!/usr/bin/env python3
"""Run the functional-matched RNA encoder comparison on official chr1.

All arms use the same functional CpG encoder, J0 locus-conditioned retrieval,
mean-proxy task, loss, optimizer, batching, and official data protocol. Only
the upstream patient RNA encoder changes.

Use --print-commands to audit commands without touching GPUs or requiring the
derived caches to exist on the current machine.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from functional_rna_encoder_suite import ARMS, SEEDS  # noqa: E402


def resolve_root(value: str | None) -> Path:
    if value:
        return Path(value).resolve()
    if os.environ.get("METHYL_DATA_ROOT"):
        return Path(os.environ["METHYL_DATA_ROOT"]).resolve()
    local = REPO_ROOT / "MethylPredictionData"
    if local.is_dir():
        return local.resolve()
    return Path("/dune/DATASETS/MethylPredictionData")


def paths_for(root: Path) -> dict[str, Path]:
    canonical = root / "datasets" / "methylprophet_repro_v1"
    prepared = root / "derived" / "methylprophet_table5_tcga_chr1"
    return {
        "canonical": canonical,
        "prepared": prepared,
        "prior_cache": prepared / "features",
        "canonical_rna": prepared / "rna",
        "bulkformer_147m": (
            root / "derived" / "bulkformer_embeddings" / "tcga_147m"
        ),
        "bulkrnabert": (
            root / "derived" / "bulkrnabert_embeddings" / "tcga"
        ),
        "registry": (
            canonical / "cpg" / "registries" / "array_cpg_map.parquet"
        ),
        "cpg_targets": root / "derived" / "cpg_statistics" / "chr1",
        "functional_atlas": (
            root
            / "derived"
            / "ntv3_functional_peak_atlas_chr1_all_sources"
        ),
        "annotation_cache": (
            root
            / "derived"
            / "ntv3_probe_targets"
            / "chr1_annotation_features_all_sources"
        ),
        "output_root": root / "experiments",
    }


def cache_for(spec, paths: dict[str, Path]) -> Path:
    if spec.cache_kind == "canonical":
        return paths["canonical_rna"]
    if spec.cache_kind == "bulkformer_147m":
        return paths["bulkformer_147m"]
    if spec.cache_kind == "bulkrnabert":
        return paths["bulkrnabert"]
    raise ValueError(f"unknown cache_kind {spec.cache_kind!r}")


def select_arms(raw: str | None):
    by_name = {arm.name: arm for arm in ARMS}
    if not raw:
        return ARMS
    names = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [x for x in names if x not in by_name]
    if unknown:
        raise SystemExit(
            f"unknown arm(s): {unknown}; known={list(by_name)}"
        )
    return tuple(by_name[x] for x in names)


def parse_seeds(raw: str | None) -> tuple[int, ...]:
    if not raw:
        return SEEDS
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def run_dir(paths: dict[str, Path], arm, seed: int) -> Path:
    return (
        paths["output_root"]
        / "runs"
        / "locus_cls_joint"
        / "chr1"
        / arm.run_id(seed)
    )


def eval_file(rd: Path) -> Path:
    return rd / "evaluation" / "chr1" / "metrics.json"


def train_cmd(
    arm,
    seed: int,
    mode: str,
    paths: dict[str, Path],
    *,
    resume: bool,
    early_stop_patience: int,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--model",
        "rna_methylation",
        "--scope",
        "chr1",
        "--engine",
        "matched_chr1_shared_backbone",
        "--mode",
        mode,
        "--recipe",
        arm.recipe,
        "--seed",
        str(seed),
        "--canonical-root",
        str(paths["canonical"]),
        "--prepared-root",
        str(paths["prepared"]),
        "--prior-cache",
        str(paths["prior_cache"]),
        "--rna-cache",
        str(cache_for(arm, paths)),
        "--registry",
        str(paths["registry"]),
        "--cpg-targets-dir",
        str(paths["cpg_targets"]),
        "--functional-atlas",
        str(paths["functional_atlas"]),
        "--annotation-cache",
        str(paths["annotation_cache"]),
        "--functional-only",
        "--output-root",
        str(paths["output_root"]),
        "--run-id",
        arm.run_id(seed),
    ]
    if mode == "development":
        cmd += ["--early-stop-patience", str(early_stop_patience)]
    if resume:
        cmd.append("--resume")
    return cmd


def evaluate_cmd(
    arm,
    seed: int,
    paths: dict[str, Path],
) -> list[str]:
    rd = run_dir(paths, arm, seed)
    return [
        sys.executable,
        "scripts/evaluate.py",
        "--model",
        "rna_methylation",
        "--engine",
        "matched_chr1_shared_backbone",
        "--checkpoint",
        str(rd / "checkpoints" / "best.pt"),
        "--eval-scope",
        "chr1",
        "--output",
        str(eval_file(rd)),
        "--recipe",
        arm.recipe,
        "--canonical-root",
        str(paths["canonical"]),
        "--prepared-root",
        str(paths["prepared"]),
        "--prior-cache",
        str(paths["prior_cache"]),
        "--rna-cache",
        str(cache_for(arm, paths)),
        "--registry",
        str(paths["registry"]),
        "--cpg-targets-dir",
        str(paths["cpg_targets"]),
        "--functional-atlas",
        str(paths["functional_atlas"]),
        "--annotation-cache",
        str(paths["annotation_cache"]),
        "--functional-only",
    ]


def validate_inputs(
    arm,
    paths: dict[str, Path],
) -> None:
    required = [
        paths["prepared"],
        paths["prior_cache"],
        paths["cpg_targets"],
        paths["functional_atlas"],
        paths["annotation_cache"],
        cache_for(arm, paths),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "missing required functional RNA-comparison input(s):\n  "
            + "\n  ".join(missing)
        )

    if arm.name == "gene_pathway":
        membership = (
            REPO_ROOT
            / "resources"
            / "pathways"
            / "survpath_combine_tcga.npz"
        )
        if not membership.is_file():
            raise FileNotFoundError(
                f"missing {membership}; run "
                "scripts/prepare_gene_pathway_membership.py first"
            )

    cache = cache_for(arm, paths)
    for name in ("rna_sample_idx.npy", "rna_zscore.f16.npy"):
        if not (cache / name).is_file():
            raise FileNotFoundError(
                f"RNA cache {cache} is missing {name}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", help="comma-separated arm names; default all")
    ap.add_argument(
        "--seeds",
        help="comma-separated seeds; default 17,29,43",
    )
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--data-root")
    ap.add_argument(
        "--mode",
        choices=["development", "final"],
        default="development",
    )
    ap.add_argument("--early-stop-patience", type=int, default=6)
    ap.add_argument("--print-commands", action="store_true")
    ap.add_argument("--refresh-evaluation", action="store_true")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    paths = paths_for(resolve_root(args.data_root))
    arms = select_arms(args.arms)
    seeds = parse_seeds(args.seeds)

    log_dir = REPO_ROOT / "logs" / "functional_rna_encoder_comparison"
    log_dir.mkdir(parents=True, exist_ok=True)

    for arm in arms:
        for seed in seeds:
            rd = run_dir(paths, arm, seed)
            summary = rd / "training" / "summary.json"
            best = rd / "checkpoints" / "best.pt"
            ev = eval_file(rd)

            train_done = summary.is_file() and best.is_file()
            eval_done = ev.is_file()
            resume = (
                rd.is_dir()
                and (rd / "checkpoints" / "last.pt").is_file()
                and not train_done
            )

            commands: list[tuple[str, list[str]]] = []
            if not train_done:
                commands.append(
                    (
                        "train",
                        train_cmd(
                            arm,
                            seed,
                            args.mode,
                            paths,
                            resume=resume,
                            early_stop_patience=args.early_stop_patience,
                        ),
                    )
                )
            if (
                args.mode == "final"
                and (not eval_done or args.refresh_evaluation)
            ):
                commands.append(
                    ("eval", evaluate_cmd(arm, seed, paths))
                )

            if args.print_commands:
                print(f"# {arm.name} seed={seed} mode={args.mode}")
                for _, cmd in commands:
                    print(shlex.join(cmd))
                continue

            validate_inputs(arm, paths)

            if rd.is_dir() and not resume and not train_done:
                shutil.rmtree(rd)

            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(args.gpu),
                "PYTHONPATH": str(REPO_ROOT / "src"),
            }

            if not commands:
                print(
                    f"[functional-rnaenc] {arm.name} seed={seed}: "
                    "complete, skipping",
                    flush=True,
                )
                continue

            for phase, cmd in commands:
                log = log_dir / f"{arm.run_id(seed)}.{phase}.log"
                print(
                    f"[functional-rnaenc] {arm.name} seed={seed}: "
                    f"{phase} -> {log}",
                    flush=True,
                )
                with log.open("a") as handle:
                    handle.write("\n=== " + shlex.join(cmd) + "\n")
                    handle.flush()
                    rc = subprocess.run(
                        cmd,
                        cwd=REPO_ROOT,
                        env=env,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    ).returncode
                if rc != 0:
                    print(
                        f"[functional-rnaenc] {arm.name}: "
                        f"{phase} FAILED rc={rc}",
                        flush=True,
                    )
                    return rc

    collect = [
        sys.executable,
        "scripts/experiments/collect_rna_encoder_comparison.py",
        "--data-root",
        str(resolve_root(args.data_root)),
        "--mode",
        args.mode,
    ]
    if len(seeds) == 1:
        collect += ["--seed", str(seeds[0])]
    subprocess.run(collect, cwd=REPO_ROOT, check=False)

    print("[functional-rnaenc] sequence complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
