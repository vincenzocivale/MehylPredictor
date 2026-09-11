#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


ROOT = Path.cwd()
EXPECTED_BRANCH = "refactor/repo-v2-2026-09"


def require_branch() -> None:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], text=True
    ).strip()
    if branch != EXPECTED_BRANCH:
        raise SystemExit(
            f"expected branch {EXPECTED_BRANCH!r}, got {branch!r}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    require_branch()

    if args.dry_run:
        print(
            '''Phase 3b3b will:
  ~ replace the historical shared-backbone RNA-encoder runner with the
    functional-matched J0 runner
  ~ replace its collector with a functional-only collector
  ~ use the functional_rna_encoder_suite.py registry created in phase 3b3a
  ~ support canonical RNA, BulkFormer, and BulkRNABert caches
  ~ pass functional atlas + annotation cache + --functional-only to train/eval
  ~ restrict the migrated study to chr1
  ~ add --print-commands for auditable dry execution
  + add runner/registry regression tests
  ~ write results to a fresh functional_rna_encoder_comparison_2026_09 ledger
  ~ document the migration

Historical shared-backbone result YAML files and old recipes remain untouched
for provenance; only the executable runner/collector are cut over.'''
        )
        return 0

    runner = ROOT / "scripts/experiments/run_rna_encoder_comparison.py"
    collector = ROOT / "scripts/experiments/collect_rna_encoder_comparison.py"
    test_file = ROOT / "tests/test_functional_rna_encoder_suite.py"
    if test_file.exists():
        raise SystemExit(f"{test_file}: already exists; refusing to overwrite")

    runner.write_text(r'''#!/usr/bin/env python3
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
        "feature_cache": prepared / "features",
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
        "--feature-cache",
        str(paths["feature_cache"]),
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
        "--feature-cache",
        str(paths["feature_cache"]),
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
        paths["feature_cache"],
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
''')

    collector.write_text(r'''#!/usr/bin/env python3
"""Collect functional-matched RNA encoder runs into a fresh result ledger."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = (
    REPO_ROOT
    / "results"
    / "reference"
    / "appendix"
    / "ablations"
    / "functional_rna_encoder_comparison_2026_09"
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from functional_rna_encoder_suite import ARMS, SEEDS, STUDY  # noqa: E402


def resolve_root(value: str | None) -> Path:
    if value:
        return Path(value).resolve()
    if os.environ.get("METHYL_DATA_ROOT"):
        return Path(os.environ["METHYL_DATA_ROOT"]).resolve()
    local = REPO_ROOT / "MethylPredictionData"
    if local.is_dir():
        return local.resolve()
    return Path("/dune/DATASETS/MethylPredictionData")


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def collect_one(base: Path, arm, seed: int, mode: str):
    rd = base / arm.run_id(seed)
    summary = load_json(rd / "training" / "summary.json")
    if not summary:
        return None

    record = {
        "arm": arm.name,
        "recipe": arm.recipe,
        "cache_kind": arm.cache_kind,
        "mode": mode,
        "seed": seed,
        "run_id": rd.name,
        "run_dir": str(rd),
        "best_epoch": summary.get("best_epoch"),
        "epochs_run": summary.get("epochs_run"),
    }

    if mode == "development":
        history = load_json(rd / "training" / "history.json") or []
        best_epoch = summary.get("best_epoch")
        row = next(
            (x for x in history if x.get("epoch") == best_epoch),
            None,
        ) or {}
        record["views"] = row.get("development")
        record["headline_mas_pcc"] = summary.get(
            "best_inner_double_ood_mas_pcc"
        )
        record["headline_mse"] = None
    else:
        ev = load_json(rd / "evaluation" / "chr1" / "metrics.json")
        if not ev:
            return None
        record["views"] = ev.get("views")
        metrics = ev.get("metrics") or {}
        record["headline_mas_pcc"] = metrics.get("mas_pcc")
        record["headline_mse"] = metrics.get("mse")

    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root")
    ap.add_argument(
        "--mode",
        choices=["development", "final"],
        default="development",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="collect one seed only; default collects 17,29,43",
    )
    args = ap.parse_args()

    root = resolve_root(args.data_root)
    base = (
        root
        / "experiments"
        / "runs"
        / "locus_cls_joint"
        / "chr1"
    )
    seeds = (args.seed,) if args.seed is not None else SEEDS

    rows = []
    for arm in ARMS:
        for seed in seeds:
            rec = collect_one(base, arm, seed, args.mode)
            if rec is not None:
                rows.append(rec)

    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "study": STUDY,
        "protocol": {
            "scope": "TCGA chr1",
            "mode": args.mode,
            "seeds": list(seeds),
            "fixed_harness": (
                "same functional CpG encoder, J0 retrieval, mean proxy, "
                "loss, optimizer and batching; only RNA encoder changes"
            ),
            "reference_recipe": "configs/models/main.yaml",
        },
        "runs": rows,
    }

    yaml_path = RESULTS / f"summary_{args.mode}.yaml"
    yaml_path.write_text(
        yaml.safe_dump(payload, sort_keys=False)
    )

    lines = [
        f"# Functional RNA encoder comparison — chr1 {args.mode}",
        "",
        "| encoder | seed | best epoch | headline MAS-PCC | headline MSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        mas = row.get("headline_mas_pcc")
        mse = row.get("headline_mse")
        mas_text = "—" if mas is None else f"{mas:.4f}"
        mse_text = "—" if mse is None else f"{mse:.5f}"
        lines.append(
            f"| `{row['arm']}` | {row['seed']} | "
            f"{row.get('best_epoch')} | {mas_text} | {mse_text} |"
        )

    md_path = RESULTS / f"summary_{args.mode}.md"
    md_path.write_text("\n".join(lines) + "\n")

    print(f"[collect] {len(rows)} run(s) -> {RESULTS}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
''')

    test_file.write_text(r'''from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


suite = _load(
    "functional_rna_encoder_suite",
    "scripts/experiments/functional_rna_encoder_suite.py",
)
runner = _load(
    "run_rna_encoder_comparison",
    "scripts/experiments/run_rna_encoder_comparison.py",
)


def test_registry_has_five_fresh_functional_arms():
    assert [arm.name for arm in suite.ARMS] == [
        "ours_program_tokens",
        "bottleneck_mlp",
        "gene_pathway",
        "bulkformer_147m",
        "bulkrnabert",
    ]
    assert len({arm.run_id(17) for arm in suite.ARMS}) == 5
    assert all(
        arm.run_id(17).startswith("functional-rnaenc-")
        for arm in suite.ARMS
    )


def test_runner_cache_mapping_is_explicit(tmp_path):
    paths = {
        "canonical_rna": tmp_path / "canonical",
        "bulkformer_147m": tmp_path / "bulkformer",
        "bulkrnabert": tmp_path / "bulkrnabert",
    }
    by_name = {arm.name: arm for arm in suite.ARMS}

    assert runner.cache_for(
        by_name["ours_program_tokens"], paths
    ) == paths["canonical_rna"]
    assert runner.cache_for(
        by_name["bottleneck_mlp"], paths
    ) == paths["canonical_rna"]
    assert runner.cache_for(
        by_name["gene_pathway"], paths
    ) == paths["canonical_rna"]
    assert runner.cache_for(
        by_name["bulkformer_147m"], paths
    ) == paths["bulkformer_147m"]
    assert runner.cache_for(
        by_name["bulkrnabert"], paths
    ) == paths["bulkrnabert"]


def test_generated_training_command_is_functional_only(tmp_path):
    paths = {
        "canonical": tmp_path / "canonical",
        "prepared": tmp_path / "prepared",
        "feature_cache": tmp_path / "features",
        "canonical_rna": tmp_path / "rna",
        "bulkformer_147m": tmp_path / "bulkformer",
        "bulkrnabert": tmp_path / "bulkrnabert",
        "registry": tmp_path / "registry.parquet",
        "cpg_targets": tmp_path / "targets",
        "functional_atlas": tmp_path / "functional",
        "annotation_cache": tmp_path / "annotations",
        "output_root": tmp_path / "experiments",
    }
    arm = suite.ARMS[0]
    cmd = runner.train_cmd(
        arm,
        17,
        "final",
        paths,
        resume=False,
        early_stop_patience=6,
    )

    assert "--functional-only" in cmd
    assert cmd[cmd.index("--functional-atlas") + 1] == str(
        paths["functional_atlas"]
    )
    assert cmd[cmd.index("--annotation-cache") + 1] == str(
        paths["annotation_cache"]
    )
    assert cmd[cmd.index("--recipe") + 1] == arm.recipe
    assert cmd[cmd.index("--run-id") + 1] == arm.run_id(17)


def test_generated_eval_command_uses_same_functional_inputs(tmp_path):
    paths = {
        "canonical": tmp_path / "canonical",
        "prepared": tmp_path / "prepared",
        "feature_cache": tmp_path / "features",
        "canonical_rna": tmp_path / "rna",
        "bulkformer_147m": tmp_path / "bulkformer",
        "bulkrnabert": tmp_path / "bulkrnabert",
        "registry": tmp_path / "registry.parquet",
        "cpg_targets": tmp_path / "targets",
        "functional_atlas": tmp_path / "functional",
        "annotation_cache": tmp_path / "annotations",
        "output_root": tmp_path / "experiments",
    }
    arm = suite.ARMS[3]
    cmd = runner.evaluate_cmd(arm, 29, paths)

    assert "--functional-only" in cmd
    assert cmd[cmd.index("--rna-cache") + 1] == str(
        paths["bulkformer_147m"]
    )
    assert cmd[cmd.index("--functional-atlas") + 1] == str(
        paths["functional_atlas"]
    )
''')

    scope = ROOT / "docs/REPOSITORY_SCOPE.md"
    with scope.open("a") as handle:
        handle.write(
            '''
## Phase 3b3b: functional RNA-encoder execution harness

The executable RNA-encoder runner and collector now target only the
functional-matched comparison introduced in phase 3b3a.

The migrated runner:

- is chr1-only;
- passes the frozen functional atlas and annotation cache explicitly;
- always uses `--functional-only`;
- selects canonical RNA, BulkFormer, or BulkRNABert sample caches explicitly;
- uses fresh `functional-rnaenc-*` run IDs;
- supports `--print-commands` for command-level audit before GPU execution.

The collector writes to a new
`functional_rna_encoder_comparison_2026_09` result directory and never mixes
historical shared-backbone metrics with the new functional comparison.

Historical result ledgers and historical recipes remain untouched for
provenance; their old executable runner logic has been retired.
'''
        )

    print("Phase 3b3b applied.")
    print()
    print("Focused tests:")
    print(
        "  pytest -q tests/test_functional_rna_encoder_suite.py "
        "tests/test_functional_rna_encoder_comparison.py "
        "tests/test_functional_baselines.py"
    )
    print()
    print("Then full suite:")
    print("  pytest -q")
    print()
    print("Audit generated commands without requiring caches:")
    print(
        "  python scripts/experiments/run_rna_encoder_comparison.py "
        "--arms ours_program_tokens,bottleneck_mlp,gene_pathway "
        "--seeds 17 --mode final --print-commands"
    )
    print()
    print("Inspect:")
    print("  git diff --stat")
    print("  git diff")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
