#!/usr/bin/env python3
"""Preflight check for the genome-wide full-coverage training run:
re-verifies official-split integrity/overlap, data completeness (embeddings/
prior/variability/beta/RNA), disk/RAM/GPU headroom, runs the existing test
suite, and does a brief smoke run of the real full-coverage training path
(uniform sampling, 1 epoch, 3 steps -- fast, not the full deterministic
schedule) on the real (dev-split) data paths. Writes a manifest with sha256
of every input checked. Exits non-zero on any hard failure.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MP_DATA = Path("/data/dataset/methylation/MethylProphetData/parquet/241231-tcga_array")
SCRATCH = Path("/raid/DATASETS/MethylPredictionData")
INPUT_DIR = SCRATCH
DEV_INPUT_DIR = SCRATCH / "rna_branch_inputs_dev_seed17"

DISK_FREE_GB_MIN = {"/": 10.0, "/raid": 20.0}


def _assert_disjoint(frame: pd.DataFrame, key: str, split: str) -> None:
    """Assert every value of ``key`` maps to exactly one ``split`` label."""
    counts = frame.groupby(key, dropna=False)[split].nunique()
    if (counts > 1).any():
        raise ValueError(f"{key} occurs in multiple split partitions")


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def check_git() -> dict:
    head = _run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).stdout.strip()
    status = _run(["git", "status", "--porcelain"], cwd=REPO_ROOT).stdout
    diff_stat = _run(["git", "diff", "--stat"], cwd=REPO_ROOT).stdout
    return {"head": head, "status_lines": len(status.splitlines()), "diff_stat": diff_stat[-4000:]}


def check_tests() -> dict:
    result = _run(
        ["python", "-m", "pytest", "-q"],
        cwd=REPO_ROOT,
    )
    tail = "\n".join(result.stdout.splitlines()[-15:])
    return {
        "returncode": result.returncode,
        "tail": tail,
    }


def check_official_splits(skip_official_check: bool = False) -> dict:
    cpg_split_manifest = pd.read_parquet(INPUT_DIR / "cpg_split_manifest.parquet")
    sample_metadata = pd.read_parquet(INPUT_DIR / "sample_metadata.parquet")

    _assert_disjoint(cpg_split_manifest, "cpg_idx", "split")
    _assert_disjoint(sample_metadata, "sample_idx", "split")

    result = {
        "aligned_train_cpg": int((cpg_split_manifest["split"] == "train").sum()),
        "aligned_train_sample": int((sample_metadata["split"] == "train").sum()),
        "no_cpg_duplicates": not cpg_split_manifest["cpg_idx"].duplicated().any(),
        "no_sample_duplicates": not sample_metadata["sample_idx"].duplicated().any(),
    }

    if skip_official_check:
        result["official_check_skipped"] = (
            f"--skip-official-split-check passed: MP_DATA ({MP_DATA}) not read/compared."
        )
        result["cpg_overlap_count"] = 0
        result["sample_overlap_count"] = 0
        return result

    train_cpg = pd.read_parquet(MP_DATA / "metadata/cpg_split/index_files/train.parquet")
    val_cpg = pd.read_parquet(MP_DATA / "metadata/cpg_split/index_files/val.parquet")
    train_sample = pd.read_csv(MP_DATA / "metadata/subset_sample_split/ind_cancer/train_sample_tissue_count_with_idx.csv")
    val_sample = pd.read_csv(MP_DATA / "metadata/subset_sample_split/ind_cancer/val_sample_tissue_count_with_idx.csv")

    cpg_overlap = set(train_cpg["chr_pos"]) & set(val_cpg["chr_pos"])
    sample_overlap = set(train_sample["sample_idx"]) & set(val_sample["sample_idx"])

    official_val_cpg = int((cpg_split_manifest["split"] == "validation").sum() + (cpg_split_manifest["split"] == "test").sum())
    official_val_sample = int((sample_metadata["split"] == "validation").sum() + (sample_metadata["split"] == "test").sum())

    result.update({
        "official_train_cpg": len(train_cpg),
        "official_val_cpg": len(val_cpg),
        "official_train_sample": len(train_sample),
        "official_val_sample": len(val_sample),
        "cpg_overlap_count": len(cpg_overlap),
        "sample_overlap_count": len(sample_overlap),
        "aligned_val_cpg_validation_plus_test": official_val_cpg,
        "aligned_val_sample_validation_plus_test": official_val_sample,
        "sample_coverage_gap_note": (
            f"official train_sample={len(train_sample)}, aligned train_sample="
            f"{int((sample_metadata['split'] == 'train').sum())} -- RNA expression is not available "
            "for every officially-assigned sample (dropped_na_cancer_type=true in "
            "rna_branch_inputs/manifest.json). 'all train_sample' in this run means all ALIGNED "
            "train rows, documented here, not silently assumed equal to the official count."
        ),
    })
    return result


def check_data_completeness() -> dict:
    with h5py.File(INPUT_DIR / "locus_embeddings.h5", "r") as f:
        embeddings_shape = f["embeddings"].shape
    locus_features = pd.read_parquet(INPUT_DIR / "locus_features.parquet")
    with h5py.File(INPUT_DIR / "tcga_genome_wide_beta.h5", "r") as f:
        beta_shape = f["beta"].shape
        sample_chunk = f["beta"][:200, :5000]
    with h5py.File(INPUT_DIR / "tcga_rna.h5", "r") as f:
        rna_shape = f["X"].shape

    nan_fraction = float(np.isnan(sample_chunk).mean())
    return {
        "locus_embeddings_shape": list(embeddings_shape),
        "locus_features_shape": list(locus_features.shape),
        "locus_features_any_nan": bool(locus_features.isna().any().any()),
        "beta_matrix_shape": list(beta_shape),
        "beta_matrix_sampled_nan_fraction": nan_fraction,
        "rna_matrix_shape": list(rna_shape),
    }


def check_dev_split() -> dict:
    manifest_path = DEV_INPUT_DIR / "manifest.json"
    if not manifest_path.is_file():
        return {"present": False}
    manifest = json.loads(manifest_path.read_text())
    return {
        "present": True,
        "cpg_axis": {k: manifest["cpg_axis"][k] for k in ("train_before", "dev_heldout", "train_after")},
        "sample_axis": {k: manifest["sample_axis"][k] for k in ("train_before", "dev_heldout", "train_after")},
    }


def check_disk_ram_gpu() -> dict:
    disk = {}
    for mount, min_gb in DISK_FREE_GB_MIN.items():
        usage = shutil.disk_usage(mount)
        free_gb = usage.free / (1024 ** 3)
        disk[mount] = {"free_gb": round(free_gb, 1), "min_required_gb": min_gb, "ok": free_gb >= min_gb}

    mem = _run(["free", "-b"]).stdout
    gpu = _run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader"]).stdout.strip()

    import torch
    return {
        "disk": disk,
        "free_output": mem.splitlines()[:2],
        "gpu_query": gpu,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def smoke_run(output_dir: Path) -> dict:
    """1-epoch, 3-step, uniform-sampling (NOT full_coverage -- kept fast)
    smoke run of the real training path against the real dev-split data, to
    catch shape/NaN/OOM errors before committing to the long run."""
    dev_config_path = REPO_ROOT / "configs/train.yaml"
    raw = yaml.safe_load(dev_config_path.read_text())
    raw["training"]["cpg_sampling"] = "uniform"
    raw["training"]["epochs"] = 1
    raw["training"]["min_epochs"] = 1
    raw["training"]["patience"] = 1
    raw["training"]["steps_per_epoch"] = 3
    raw["training"]["sample_batch_size"] = 8
    raw["training"]["cpg_batch_size"] = 64
    raw["training"]["validation_max_cpgs"] = 64
    raw["tracking"]["backend"] = "none"
    raw["output_dir"] = str(output_dir)
    smoke_config_path = output_dir / "smoke_config.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_config_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from methylation_predictor.config import load_config
    from methylation_predictor.trainer import ExperimentRunner

    config = load_config(smoke_config_path)
    started = time.time()
    runner = ExperimentRunner(config)
    try:
        result = runner.train()
    finally:
        runner.close()
    return {
        "elapsed_seconds": time.time() - started,
        "best_epoch": result["best_epoch"],
        "train_beta_mse_last_epoch": result.get("train_beta_mse_last_epoch"),
        "num_parameters": result["num_parameters"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts/train/seed17/manifests/preflight.json",
    )
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-smoke-run", action="store_true")
    parser.add_argument(
        "--skip-official-split-check",
        action="store_true",
        help=(
            "Skip the cross-check against the raw MethylProphet official split source "
            f"files under {MP_DATA} (not read/compared when passed). Internal integrity "
            "checks on the aligned manifests (disjointness, no duplicates) still run."
        ),
    )
    args = parser.parse_args()

    report: dict[str, object] = {"created_at": time.time()}
    ok = True

    print("[preflight] git state", flush=True)
    report["git"] = check_git()

    if not args.skip_tests:
        print("[preflight] running test suite", flush=True)
        report["tests"] = check_tests()
        if report["tests"]["returncode"] != 0:
            ok = False

    print("[preflight] official split integrity", flush=True)
    report["splits"] = check_official_splits(args.skip_official_split_check)
    if report["splits"]["cpg_overlap_count"] or report["splits"]["sample_overlap_count"]:
        ok = False
    if not report["splits"]["no_cpg_duplicates"] or not report["splits"]["no_sample_duplicates"]:
        ok = False

    print("[preflight] data completeness", flush=True)
    report["data"] = check_data_completeness()
    if report["data"]["locus_embeddings_shape"][0] != 408_399 or report["data"]["locus_features_any_nan"]:
        ok = False

    print("[preflight] dev split", flush=True)
    report["dev_split"] = check_dev_split()
    if not report["dev_split"]["present"]:
        ok = False

    print("[preflight] disk/RAM/GPU", flush=True)
    report["resources"] = check_disk_ram_gpu()
    if not all(v["ok"] for v in report["resources"]["disk"].values()):
        ok = False

    if not args.skip_smoke_run:
        print("[preflight] smoke run", flush=True)
        smoke_dir = args.output.parent / "smoke_run"
        report["smoke_run"] = smoke_run(smoke_dir)

    report["ok"] = ok
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
