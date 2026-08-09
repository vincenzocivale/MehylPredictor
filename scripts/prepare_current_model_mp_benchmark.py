#!/usr/bin/env python3
"""Prepare an exact MethylProphet-compatible chr1 benchmark for RNA2DNAmModel.

This is an adapter-only preparation step. It does not alter the canonical TCGA
bundle, the model architecture, or the trainer. It creates small sample/CpG
manifests that let the existing ExperimentRunner consume the exact released
Array chr1 protocol with the canonical 25,017-gene RNA matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# Match tcga_canonical.__init__: pyarrow must load before h5py on this server.
import pyarrow as _pyarrow  # noqa: F401
import h5py
import numpy as np
import pandas as pd
import yaml

EXPECTED_FINITE = {
    "train_cpg_x_val_sample": 30_574_946,
    "val_cpg_x_train_sample": 55_155_121,
    "val_cpg_x_val_sample": 6_129_547,
}


def _decode(x) -> str:
    return x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)


def _sha_ids(values: np.ndarray) -> str:
    arr = np.sort(np.asarray(values, dtype=np.int64))
    return hashlib.sha256(("\n".join(map(str, arr.tolist())) + "\n").encode()).hexdigest()


def _nested_sample_split(ids: np.ndarray, strata: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fit: list[np.ndarray] = []
    dev: list[np.ndarray] = []
    for label in sorted(np.unique(strata).tolist()):
        group = np.asarray(ids[strata == label], dtype=np.int64)
        order = rng.permutation(len(group))
        if len(group) <= 1:
            n_dev = 0
        else:
            n_dev = max(1, int(round(frac * len(group))))
            n_dev = min(n_dev, len(group) - 1)
        dev.append(group[order[:n_dev]])
        fit.append(group[order[n_dev:]])
    return np.sort(np.concatenate(fit)), np.sort(np.concatenate(dev) if dev else np.empty(0, np.int64))


def _nested_cpg_split(ids: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(ids, dtype=np.int64)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    n_dev = max(1, int(round(frac * len(ids))))
    n_dev = min(n_dev, len(ids) - 1)
    return np.sort(ids[order[n_dev:]]), np.sort(ids[order[:n_dev]])


def _read_protocol(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = root / "protocols" / "tcga_array_chr1"
    train_s = np.load(d / "array_train_sample_idx.npy").astype(np.int64)
    val_s = np.load(d / "array_val_sample_idx.npy").astype(np.int64)
    train_c = np.load(d / "array_train_cpg_idx.npy").astype(np.int64)
    val_c = np.load(d / "array_val_cpg_idx.npy").astype(np.int64)
    assert len(train_s) == 8260 and len(val_s) == 918
    assert len(train_c) == 33885 and len(val_c) == 6742
    assert not np.intersect1d(train_s, val_s).size
    assert not np.intersect1d(train_c, val_c).size
    return train_s, val_s, train_c, val_c


def _rna_strata(root: Path, needed: np.ndarray) -> dict[int, str]:
    with h5py.File(root / "tcga_rna_official_full.h5", "r") as f:
        sample_idx = np.asarray(f["sample_idx"][...], dtype=np.int64)
        if "tissue_name" in f:
            labels = np.asarray([_decode(x) for x in f["tissue_name"][...]], dtype=object)
        elif "tissue_idx" in f:
            labels = np.asarray([f"tissue_{int(x)}" for x in f["tissue_idx"][...]], dtype=object)
        else:
            labels = np.full(len(sample_idx), "UNKNOWN", dtype=object)
    lookup = {int(s): (str(l) if str(l) not in {"", "nan", "None"} else "UNKNOWN") for s, l in zip(sample_idx, labels)}
    missing = [int(s) for s in needed if int(s) not in lookup]
    if missing:
        raise RuntimeError(f"canonical RNA missing {len(missing)} protocol samples; examples={missing[:10]}")
    return lookup


def _feature_audit(cpgs: np.ndarray, embeddings_path: Path, features_path: Path, out_dir: Path) -> dict:
    with h5py.File(embeddings_path, "r") as f:
        if "cpg_idx" not in f or "embeddings" not in f:
            raise RuntimeError(f"{embeddings_path} must contain cpg_idx and embeddings")
        emb_ids = np.asarray(f["cpg_idx"][...]).astype(str)
        emb_shape = tuple(f["embeddings"].shape)
    emb_set = set(emb_ids.tolist())
    required_str = np.asarray(cpgs).astype(str)
    missing_emb = np.asarray([int(x) for x in required_str if x not in emb_set], dtype=np.int64)

    required_cols = ["cpg_idx", "pred_ntv3_prior", "pred_log_var_between", "pred_log_var_within"]
    feat = pd.read_parquet(features_path, columns=required_cols)
    feat["cpg_idx"] = feat["cpg_idx"].astype(str)
    if feat["cpg_idx"].duplicated().any():
        raise RuntimeError("locus_features contains duplicate cpg_idx")
    feat = feat.set_index("cpg_idx")
    missing_feat = np.asarray([int(x) for x in required_str if x not in feat.index], dtype=np.int64)
    present = [x for x in required_str if x in feat.index]
    values = feat.loc[present, required_cols[1:]].to_numpy(np.float64) if present else np.empty((0, 3))
    nonfinite_rows = int((~np.isfinite(values)).any(axis=1).sum()) if len(values) else 0

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "missing_embedding_cpg_idx.npy", missing_emb)
    np.save(out_dir / "missing_feature_cpg_idx.npy", missing_feat)
    audit = {
        "required_cpgs": int(len(cpgs)),
        "required_cpg_sha256": _sha_ids(cpgs),
        "embedding_shape": list(emb_shape),
        "embedding_missing": int(len(missing_emb)),
        "feature_missing": int(len(missing_feat)),
        "feature_nonfinite_rows": nonfinite_rows,
        "status": "PASS" if len(missing_emb) == 0 and len(missing_feat) == 0 and nonfinite_rows == 0 else "NTV3_FEATURES_REQUIRED",
    }
    (out_dir / "feature_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def _write_manifests(root: Path, out: Path, seed: int, dev_fraction: float) -> dict:
    train_s, val_s, train_c, val_c = _read_protocol(root)
    label_lookup = _rna_strata(root, np.concatenate([train_s, val_s]))
    train_labels = np.asarray([label_lookup[int(x)] for x in train_s], dtype=object)
    fit_s, dev_s = _nested_sample_split(train_s, train_labels, dev_fraction, seed)
    fit_c, dev_c = _nested_cpg_split(train_c, dev_fraction, seed)

    manifest_dir = out / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    def sample_frame(split_map: list[tuple[np.ndarray, str]]) -> pd.DataFrame:
        rows = []
        for ids, split in split_map:
            rows.extend({"sample_idx": int(x), "cancer_type": label_lookup[int(x)], "split": split} for x in ids)
        return pd.DataFrame(rows)

    sample_frame([(fit_s, "train"), (dev_s, "dev_heldout"), (val_s, "official_val")]).to_parquet(
        manifest_dir / "sample_metadata_dev.parquet", index=False
    )
    sample_frame([(train_s, "train"), (val_s, "validation")]).to_parquet(
        manifest_dir / "sample_metadata_final.parquet", index=False
    )

    pd.DataFrame({
        "cpg_idx": np.concatenate([fit_c, dev_c, val_c]),
        "split": np.concatenate([
            np.full(len(fit_c), "train", dtype=object),
            np.full(len(dev_c), "dev_heldout", dtype=object),
            np.full(len(val_c), "official_val", dtype=object),
        ]),
    }).to_parquet(manifest_dir / "cpg_splits_dev.parquet", index=False)
    pd.DataFrame({
        "cpg_idx": np.concatenate([train_c, val_c]),
        "split": np.concatenate([
            np.full(len(train_c), "train", dtype=object),
            np.full(len(val_c), "validation", dtype=object),
        ]),
    }).to_parquet(manifest_dir / "cpg_splits_final.parquet", index=False)

    summary = {
        "official": {"train_samples": len(train_s), "val_samples": len(val_s), "train_cpgs": len(train_c), "val_cpgs": len(val_c)},
        "nested_dev": {"fit_samples": len(fit_s), "dev_samples": len(dev_s), "fit_cpgs": len(fit_c), "dev_cpgs": len(dev_c)},
        "sha256": {
            "official_train_samples": _sha_ids(train_s), "official_val_samples": _sha_ids(val_s),
            "official_train_cpgs": _sha_ids(train_c), "official_val_cpgs": _sha_ids(val_c),
            "dev_fit_samples": _sha_ids(fit_s), "dev_heldout_samples": _sha_ids(dev_s),
            "dev_fit_cpgs": _sha_ids(fit_c), "dev_heldout_cpgs": _sha_ids(dev_c),
        },
    }
    (manifest_dir / "split_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _make_dev_config(base_config: Path, canonical_root: Path, out: Path, embeddings: Path, features: Path, seed: int) -> Path:
    raw = yaml.safe_load(base_config.read_text())
    m = out / "manifests"
    raw["run_name"] = f"current-model-mp-array-chr1-dev-seed{seed}"
    raw["output_dir"] = str(out / "development")
    raw["data"]["rna"] = {
        "path": str(canonical_root / "tcga_rna_official_full.h5"), "values_key": "X", "row_ids_key": "sample_idx", "col_ids_key": "gene_ids"
    }
    raw["data"]["methylation"] = {
        "path": str(canonical_root / "methylation/tcga_array_official_full.h5"), "values_key": "beta", "row_ids_key": "sample_idx", "col_ids_key": "cpg_idx"
    }
    raw["data"]["locus_embeddings"] = {
        "path": str(embeddings), "values_key": "embeddings", "row_ids_key": "cpg_idx"
    }
    raw["data"]["locus_features"] = {"path": str(features), "id_column": "cpg_idx"}
    raw["data"]["sample_metadata"] = {"path": str(m / "sample_metadata_dev.parquet"), "id_column": "sample_idx"}
    raw["data"]["cpg_splits"] = {"path": str(m / "cpg_splits_dev.parquet"), "id_column": "cpg_idx"}
    raw["data"]["allow_partial_overlap"] = False
    raw["data"]["rna_control"] = "real"
    raw["data"]["standardize_rna"] = True
    raw["data"]["rna_transform"] = "zscore"

    # Architecture and objective remain exactly those in configs/train.yaml.
    tr = raw["training"]
    tr["seed"] = seed
    tr["train_sample_split"] = "train"
    tr["train_cpg_split"] = "train"
    tr["validation_sample_split"] = "dev_heldout"
    tr["validation_cpg_split"] = "dev_heldout"
    tr["cpg_sampling"] = "full_coverage"
    tr["validation_max_cpgs"] = 1024
    tr["checkpoint_selection"] = "best"
    raw["evaluation"]["panels"] = {}

    # Avoid network stalls during an unattended benchmark; all metrics/checkpoints stay local.
    raw.setdefault("tracking", {})["backend"] = "none"
    raw["tracking"]["name"] = raw["run_name"]
    raw["tracking"]["log_checkpoint"] = False

    path = out / "dev_config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def prepare(args: argparse.Namespace) -> None:
    root = Path(args.canonical_root).resolve()
    out = Path(args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    split_summary = _write_manifests(root, out, args.seed, args.dev_fraction)
    _, _, train_c, val_c = _read_protocol(root)
    audit = _feature_audit(np.concatenate([train_c, val_c]), Path(args.locus_embeddings), Path(args.locus_features), out / "manifests")

    # Validate the canonical finite counts using the already-tested protocol layer.
    from methylation_predictor.tcga_canonical import TCGACanonicalBundle, load_protocol
    with TCGACanonicalBundle.from_root(root) as bundle:
        protocol = load_protocol("tcga_array_chr1", bundle)
        finite = protocol.evaluation_finite_counts()
    if finite != EXPECTED_FINITE:
        raise RuntimeError(f"official finite-count regression failed: {finite} != {EXPECTED_FINITE}")

    report = {"feature_audit": audit, "split_audit": split_summary, "finite_counts": finite}
    (out / "manifests" / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    if audit["status"] != "PASS":
        print(json.dumps(report, indent=2))
        raise SystemExit(42)
    dev_config = _make_dev_config(Path(args.base_config), root, out, Path(args.locus_embeddings), Path(args.locus_features), args.seed)
    print(json.dumps({"status": "PASS", "dev_config": str(dev_config), **report}, indent=2))


def render_final(args: argparse.Namespace) -> None:
    out = Path(args.output_root).resolve()
    raw = yaml.safe_load(Path(args.dev_config).read_text())
    raw["run_name"] = f"current-model-mp-array-chr1-final-seed{args.seed}"
    raw["output_dir"] = str(out / "final_refit")
    raw["data"]["sample_metadata"]["path"] = str(out / "manifests/sample_metadata_final.parquet")
    raw["data"]["cpg_splits"]["path"] = str(out / "manifests/cpg_splits_final.parquet")
    tr = raw["training"]
    tr["epochs"] = int(args.best_epoch)
    tr["min_epochs"] = int(args.best_epoch)
    tr["patience"] = int(args.best_epoch)
    tr["checkpoint_selection"] = "final"
    tr["train_sample_split"] = "train"
    tr["train_cpg_split"] = "train"
    tr["validation_sample_split"] = "train"
    tr["validation_cpg_split"] = "train"
    tr["validation_every"] = 1_000_000
    raw["evaluation"]["panels"] = {}
    raw.setdefault("tracking", {})["backend"] = "none"
    raw["tracking"]["name"] = raw["run_name"]
    final = out / "final_config.yaml"
    final.write_text(yaml.safe_dump(raw, sort_keys=False))
    print(json.dumps({"final_config": str(final), "best_epoch": int(args.best_epoch)}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--canonical-root", default=os.environ.get("TCGA_CANONICAL_ROOT", "/raid/DATASETS/MethylPredictionData/methylprophet_official/official_training_data"))
    a.add_argument("--locus-embeddings", default="/raid/DATASETS/MethylPredictionData/locus_embeddings.h5")
    a.add_argument("--locus-features", default="/raid/DATASETS/MethylPredictionData/locus_features.parquet")
    a.add_argument("--base-config", default="configs/train.yaml")
    a.add_argument("--output-root", required=True)
    a.add_argument("--seed", type=int, default=17)
    a.add_argument("--dev-fraction", type=float, default=0.10)
    a.set_defaults(func=prepare)

    r = sub.add_parser("render-final")
    r.add_argument("--dev-config", required=True)
    r.add_argument("--output-root", required=True)
    r.add_argument("--best-epoch", type=int, required=True)
    r.add_argument("--seed", type=int, default=17)
    r.set_defaults(func=render_final)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
