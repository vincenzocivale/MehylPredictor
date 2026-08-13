#!/usr/bin/env python3
"""Prepare leakage-safe exact Array-chr1 architecture-ablation runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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


def _read_protocol(root: Path):
    d = root / "protocols" / "tcga_array_chr1"
    train_s = np.load(d / "array_train_sample_idx.npy").astype(np.int64)
    val_s = np.load(d / "array_val_sample_idx.npy").astype(np.int64)
    train_c = np.load(d / "array_train_cpg_idx.npy").astype(np.int64)
    val_c = np.load(d / "array_val_cpg_idx.npy").astype(np.int64)
    if (len(train_s), len(val_s), len(train_c), len(val_c)) != (8260, 918, 33885, 6742):
        raise RuntimeError(
            "unexpected exact Array-chr1 split counts: "
            f"{len(train_s)}/{len(val_s)} samples, {len(train_c)}/{len(val_c)} CpGs"
        )
    if np.intersect1d(train_s, val_s).size or np.intersect1d(train_c, val_c).size:
        raise RuntimeError("official Array-chr1 train/validation split overlaps")
    return train_s, val_s, train_c, val_c


def _nested_sample_split(ids: np.ndarray, strata: np.ndarray, frac: float, seed: int):
    rng = np.random.default_rng(seed)
    fit, dev = [], []
    for label in sorted(np.unique(strata).tolist()):
        group = np.asarray(ids[strata == label], dtype=np.int64)
        order = rng.permutation(len(group))
        n_dev = 0 if len(group) <= 1 else min(len(group) - 1, max(1, int(round(frac * len(group)))))
        dev.append(group[order[:n_dev]])
        fit.append(group[order[n_dev:]])
    return np.sort(np.concatenate(fit)), np.sort(np.concatenate(dev))


def _nested_cpg_split(ids: np.ndarray, frac: float, seed: int):
    ids = np.asarray(ids, dtype=np.int64)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    n_dev = min(len(ids) - 1, max(1, int(round(frac * len(ids)))))
    return np.sort(ids[order[n_dev:]]), np.sort(ids[order[:n_dev]])


def _rna_labels(root: Path, needed: np.ndarray) -> dict[int, str]:
    path = root / "rna" / "tcga_rna_official_full.h5"
    with h5py.File(path, "r") as h:
        ids = np.asarray(h["sample_idx"][...], dtype=np.int64)
        if "tissue_name" in h:
            labels = np.asarray([_decode(x) for x in h["tissue_name"][...]], dtype=object)
        elif "tissue_idx" in h:
            labels = np.asarray([f"tissue_{int(x)}" for x in h["tissue_idx"][...]], dtype=object)
        else:
            labels = np.full(len(ids), "UNKNOWN", dtype=object)
    lookup = {int(i): (str(x) if str(x) not in {"", "nan", "None"} else "UNKNOWN") for i, x in zip(ids, labels)}
    missing = [int(i) for i in needed if int(i) not in lookup]
    if missing:
        raise RuntimeError(f"RNA missing {len(missing)} protocol samples; examples={missing[:10]}")
    return lookup


def _write_manifests(root: Path, out: Path, seed: int, dev_fraction: float) -> dict:
    train_s, val_s, train_c, val_c = _read_protocol(root)
    labels = _rna_labels(root, np.concatenate([train_s, val_s]))
    train_labels = np.asarray([labels[int(x)] for x in train_s], dtype=object)
    fit_s, dev_s = _nested_sample_split(train_s, train_labels, dev_fraction, seed)
    fit_c, dev_c = _nested_cpg_split(train_c, dev_fraction, seed)

    m = out / "manifests"
    m.mkdir(parents=True, exist_ok=True)

    def sample_frame(parts):
        rows = []
        for ids, split in parts:
            rows.extend({"sample_idx": int(x), "cancer_type": labels[int(x)], "split": split} for x in ids)
        return pd.DataFrame(rows)

    sample_frame([(fit_s, "train"), (dev_s, "dev_heldout"), (val_s, "official_val")]).to_parquet(
        m / "sample_metadata_dev.parquet", index=False
    )
    sample_frame([(train_s, "train"), (val_s, "validation")]).to_parquet(
        m / "sample_metadata_final.parquet", index=False
    )
    pd.DataFrame({
        "cpg_idx": np.concatenate([fit_c, dev_c, val_c]),
        "split": np.concatenate([
            np.full(len(fit_c), "train", object),
            np.full(len(dev_c), "dev_heldout", object),
            np.full(len(val_c), "official_val", object),
        ]),
    }).to_parquet(m / "cpg_splits_dev.parquet", index=False)
    pd.DataFrame({
        "cpg_idx": np.concatenate([train_c, val_c]),
        "split": np.concatenate([
            np.full(len(train_c), "train", object),
            np.full(len(val_c), "validation", object),
        ]),
    }).to_parquet(m / "cpg_splits_final.parquet", index=False)

    result = {
        "official": {
            "train_samples": len(train_s), "val_samples": len(val_s),
            "train_cpgs": len(train_c), "val_cpgs": len(val_c),
        },
        "nested_dev": {
            "fit_samples": len(fit_s), "dev_samples": len(dev_s),
            "fit_cpgs": len(fit_c), "dev_cpgs": len(dev_c),
        },
        "sha256": {
            "official_train_samples": _sha_ids(train_s),
            "official_val_samples": _sha_ids(val_s),
            "official_train_cpgs": _sha_ids(train_c),
            "official_val_cpgs": _sha_ids(val_c),
            "dev_fit_samples": _sha_ids(fit_s),
            "dev_heldout_samples": _sha_ids(dev_s),
            "dev_fit_cpgs": _sha_ids(fit_c),
            "dev_heldout_cpgs": _sha_ids(dev_c),
        },
    }
    (m / "split_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _feature_audit(root: Path, embeddings: Path, features: Path, out: Path) -> dict:
    _, _, train_c, val_c = _read_protocol(root)
    required = np.unique(np.concatenate([train_c, val_c])).astype(np.int64)

    with h5py.File(embeddings, "r") as h:
        if "cpg_idx" not in h or "embeddings" not in h:
            raise RuntimeError(f"{embeddings} must contain cpg_idx and embeddings")
        emb_ids = np.asarray(h["cpg_idx"][...], dtype=np.int64)
        emb_shape = tuple(h["embeddings"].shape)
    missing_emb = required[~np.isin(required, emb_ids)]

    cols = ["cpg_idx", "pred_ntv3_prior", "pred_log_var_between", "pred_log_var_within"]
    frame = pd.read_parquet(features, columns=cols)
    if frame["cpg_idx"].duplicated().any():
        raise RuntimeError("locus_features contains duplicate cpg_idx")
    frame["cpg_idx"] = frame["cpg_idx"].astype(np.int64)
    frame = frame.set_index("cpg_idx")
    missing_feat = required[~np.isin(required, frame.index.to_numpy(np.int64))]
    present = required[np.isin(required, frame.index.to_numpy(np.int64))]
    values = frame.loc[present, cols[1:]].to_numpy(np.float64)
    nonfinite = int((~np.isfinite(values)).any(axis=1).sum())

    result = {
        "required_cpgs": len(required),
        "embedding_shape": list(emb_shape),
        "embedding_missing": len(missing_emb),
        "feature_missing": len(missing_feat),
        "feature_nonfinite_rows": nonfinite,
        "status": "PASS" if not len(missing_emb) and not len(missing_feat) and not nonfinite else "FAIL",
    }
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "missing_embedding_cpg_idx.npy", missing_emb)
    np.save(out / "missing_feature_cpg_idx.npy", missing_feat)
    (out / "feature_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _make_dev_config(base_config: Path, root: Path, out: Path, embeddings: Path, features: Path, seed: int) -> Path:
    raw = yaml.safe_load(base_config.read_text())
    m = out / "manifests"
    raw["run_name"] = f"architecture-ablation-array-chr1-dev-seed{seed}"
    raw["output_dir"] = str(out / "development")
    raw["data"]["rna"] = {
        "path": str(root / "rna" / "tcga_rna_official_full.h5"),
        "values_key": "X", "row_ids_key": "sample_idx", "col_ids_key": "gene_ids",
    }
    raw["data"]["methylation"] = {
        "path": str(root / "methylation" / "tcga_array_official_full.h5"),
        "values_key": "beta", "row_ids_key": "sample_idx", "col_ids_key": "cpg_idx",
    }
    raw["data"]["locus_embeddings"] = {
        "path": str(embeddings), "values_key": "embeddings", "row_ids_key": "cpg_idx",
    }
    raw["data"]["locus_features"] = {"path": str(features), "id_column": "cpg_idx"}
    raw["data"]["sample_metadata"] = {"path": str(m / "sample_metadata_dev.parquet"), "id_column": "sample_idx"}
    raw["data"]["cpg_splits"] = {"path": str(m / "cpg_splits_dev.parquet"), "id_column": "cpg_idx"}
    raw["data"]["allow_partial_overlap"] = False
    raw["data"]["rna_control"] = "real"
    raw["data"]["standardize_rna"] = True
    raw["data"]["rna_transform"] = "zscore"

    tr = raw["training"]
    tr["seed"] = seed
    tr["train_sample_split"] = "train"
    tr["train_cpg_split"] = "train"
    tr["validation_sample_split"] = "dev_heldout"
    tr["validation_cpg_split"] = "dev_heldout"
    tr["cpg_sampling"] = "full_coverage"
    tr["checkpoint_selection"] = "best"
    # Do not overwrite validation_max_cpgs: the ablation renderer deliberately
    # sets it to null so MAS-PCC sees the complete nested-development CpG set.

    raw["evaluation"]["panels"] = {}
    raw["tracking"] = {"backend": "none", "log_checkpoint": False}
    path = out / "dev_config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def prepare(args) -> None:
    root = Path(args.canonical_root).resolve()
    out = Path(args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    split = _write_manifests(root, out, args.seed, args.dev_fraction)
    features = _feature_audit(root, Path(args.locus_embeddings), Path(args.locus_features), out / "manifests")

    from methylation_predictor.tcga_canonical import TCGACanonicalBundle, load_protocol
    with TCGACanonicalBundle.from_root(root) as bundle:
        finite = load_protocol("tcga_array_chr1", bundle).evaluation_finite_counts()
    if finite != EXPECTED_FINITE:
        raise RuntimeError(f"official finite-count regression failed: {finite} != {EXPECTED_FINITE}")
    if features["status"] != "PASS":
        raise RuntimeError(f"Array-chr1 feature contract failed: {features}")

    dev = _make_dev_config(
        Path(args.base_config), root, out,
        Path(args.locus_embeddings), Path(args.locus_features), args.seed,
    )
    result = {"status": "PASS", "dev_config": str(dev), "split_audit": split, "feature_audit": features, "finite_counts": finite}
    (out / "manifests" / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def render_final(args) -> None:
    out = Path(args.output_root).resolve()
    raw = yaml.safe_load(Path(args.dev_config).read_text())
    raw["run_name"] = f"architecture-ablation-array-chr1-final-seed{args.seed}"
    raw["output_dir"] = str(out / "final_refit")
    raw["data"]["sample_metadata"]["path"] = str(out / "manifests" / "sample_metadata_final.parquet")
    raw["data"]["cpg_splits"]["path"] = str(out / "manifests" / "cpg_splits_final.parquet")
    tr = raw["training"]
    tr["epochs"] = int(args.best_epoch)
    tr["min_epochs"] = int(args.best_epoch)
    tr["patience"] = int(args.best_epoch)
    tr["checkpoint_selection"] = "final"
    tr["train_sample_split"] = "train"
    tr["train_cpg_split"] = "train"
    # The trainer always runs one validation pass on the final epoch
    # (`epoch == config.training.epochs`) regardless of validation_every --
    # pointing it at the full train x train cross product (~280M pairs) costs
    # ~30-40 min per variant for a diagnostic value nobody consumes: the
    # actual MethylProphet comparison (evaluate_current_model_vs_methylprophet.py)
    # only ever evaluates the three held-out val-based views. Point this
    # mandatory pass at the much smaller official held-out split instead, so
    # it stays cheap (~same order as one training epoch) while still giving a
    # non-trivial sanity check.
    tr["validation_sample_split"] = "validation"
    tr["validation_cpg_split"] = "validation"
    tr["validation_every"] = 1_000_000
    raw["evaluation"]["panels"] = {}
    raw["tracking"] = {"backend": "none", "log_checkpoint": False}
    path = out / "final_config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    print(json.dumps({"final_config": str(path), "best_epoch": int(args.best_epoch)}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("prepare")
    q.add_argument("--canonical-root", required=True)
    q.add_argument("--locus-embeddings", required=True)
    q.add_argument("--locus-features", required=True)
    q.add_argument("--base-config", required=True)
    q.add_argument("--output-root", required=True)
    q.add_argument("--seed", type=int, default=17)
    q.add_argument("--dev-fraction", type=float, default=0.10)
    q.set_defaults(func=prepare)

    q = sub.add_parser("render-final")
    q.add_argument("--dev-config", required=True)
    q.add_argument("--output-root", required=True)
    q.add_argument("--best-epoch", type=int, required=True)
    q.add_argument("--seed", type=int, default=17)
    q.set_defaults(func=render_final)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
