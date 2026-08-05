#!/usr/bin/env python3
"""Train and evaluate static-only predictors of a CpG training prior.

The input is deliberately one row per CpG.  It must already contain a target
computed only from the samples assigned to the upstream training partition;
this program never derives a target from test-sample methylation values.

Example:
  python -m methylation_predictor.genomic_encoder.cli static-baselines \
    --input artifacts/diagnostics/methylprophet/export/cpg_train_prior.parquet \
    --output-dir artifacts/genomic_encoder/static_prior/random --split random \
    --numeric gc_content,cpg_density,tss_distance,seq_0,seq_1 \
    --categorical cgi_category,genomic_region,chromosome
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


EPS = 1e-6


def _csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _hash_u64(value: pd.Series, seed: int) -> np.ndarray:
    """Stable hash independent of parquet order and Python hash randomisation."""
    return np.array(
        [int.from_bytes(hashlib.blake2b(f"{seed}:{x}".encode(), digest_size=8).digest(), "little") for x in value],
        dtype=np.uint64,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _partition(frame: pd.DataFrame, split: str, cpg_id: str, chromosome: str, position: str,
               block_size: int, seed: int) -> pd.Series:
    if split == "fixed":
        # Caller already attached the split column (e.g. MethylProphet's real
        # train_cpg/val_cpg boundary via build_genome_wide_targets.py) --
        # reuse it verbatim instead of re-deriving one by hashing.
        if "split" not in frame.columns:
            raise ValueError("split='fixed' requires an existing 'split' column in the input")
        if not frame["split"].isin(["train", "validation", "test"]).all():
            raise ValueError("split column must contain only train/validation/test")
        return frame["split"]
    if split == "random":
        keys = frame[cpg_id].astype(str)
    else:
        if frame[[chromosome, position]].isna().any().any():
            raise ValueError("block split requires non-null chromosome and position")
        block = (pd.to_numeric(frame[position]) // block_size).astype(np.int64)
        keys = frame[chromosome].astype(str) + ":" + block.astype(str)
    bucket = _hash_u64(keys, seed) % np.uint64(100)
    # Fixed 80/10/10 allocation.  Entire blocks use the same hash key.
    return pd.Series(np.where(bucket < 80, "train", np.where(bucket < 90, "validation", "test")), index=frame.index)


def _target(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "beta":
        return y
    clipped = np.clip(y, EPS, 1 - EPS)
    return np.log(clipped / (1 - clipped))


def _inverse_target(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "beta":
        return np.clip(y, 0.0, 1.0)
    return 1 / (1 + np.exp(-np.clip(y, -35, 35)))


def _safe_corr(fn, y: np.ndarray, p: np.ndarray) -> float | None:
    if len(y) < 2 or np.isclose(np.std(y), 0) or np.isclose(np.std(p), 0):
        return None
    value = float(fn(y, p)[0])
    return value if np.isfinite(value) else None


def _calibration(y: np.ndarray, p: np.ndarray) -> list[dict]:
    # Quantile bins make the diagnostic equally informative at both beta extremes.
    q = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
    if len(q) < 2:
        return []
    bin_id = np.clip(np.digitize(p, q[1:-1], right=True), 0, len(q) - 2)
    return [
        {"bin": int(b), "n": int((bin_id == b).sum()), "mean_prediction": float(p[bin_id == b].mean()),
         "mean_target": float(y[bin_id == b].mean())}
        for b in range(len(q) - 1) if (bin_id == b).any()
    ]


def _metrics(y: np.ndarray, p: np.ndarray, frame: pd.DataFrame, stratify: str | None) -> dict:
    result = {
        "n_cpg": int(len(y)), "mse": float(mean_squared_error(y, p)), "mae": float(mean_absolute_error(y, p)),
        "pearson": _safe_corr(pearsonr, y, p), "spearman": _safe_corr(spearmanr, y, p),
        "calibration": _calibration(y, p),
    }
    if stratify and stratify in frame:
        result["by_" + stratify] = {
            str(key): {"n_cpg": int(len(part)), "mse": float(mean_squared_error(y[part.index], p[part.index])),
                       "mae": float(mean_absolute_error(y[part.index], p[part.index]))}
            for key, part in frame.groupby(stratify, dropna=False, sort=True)
        }
    # The NTv3 protocol also requires genomic-region error.  Preserve the
    # existing configurable primary stratifier and add this independent audit
    # slice whenever it is available.
    if "genomic_region" in frame and stratify != "genomic_region":
        result["by_genomic_region"] = {
            str(key): {"n_cpg": int(len(part)), "mse": float(mean_squared_error(y[part.index], p[part.index])),
                       "mae": float(mean_absolute_error(y[part.index], p[part.index]))}
            for key, part in frame.groupby("genomic_region", dropna=False, sort=True)
        }
    # Requested diagnostics that must never be fed back as model features.
    for column, name in (("mean_train", "by_target_decile"), ("within_cpg_variance", "by_within_cpg_variance_tertile")):
        if column in frame and frame[column].nunique() > 1:
            bins = pd.qcut(frame[column], q=10 if column == "mean_train" else 3, duplicates="drop")
            result[name] = {
                str(key): {"n_cpg": int(len(part)), "mse": float(mean_squared_error(y[part.index], p[part.index])),
                           "mae": float(mean_absolute_error(y[part.index], p[part.index]))}
                for key, part in frame.assign(_bin=bins).groupby("_bin", observed=True, sort=True)
            }
    return result


def _kmer_frame(sequences: pd.Series, k: int) -> pd.DataFrame:
    alphabet = [""]
    for _ in range(k):
        alphabet = [prefix + base for prefix in alphabet for base in "ACGT"]
    index = {word: j for j, word in enumerate(alphabet)}
    values = np.zeros((len(sequences), len(alphabet)), dtype=np.float32)
    for row, sequence in enumerate(sequences.fillna("")):
        sequence = str(sequence).upper()
        n = 0
        for offset in range(max(0, len(sequence) - k + 1)):
            word = sequence[offset:offset + k]
            if word in index:
                values[row, index[word]] += 1; n += 1
        if n:
            values[row] /= n
    return pd.DataFrame(values, columns=[f"k{k}_{word}" for word in alphabet])


def _group_baseline(train: pd.DataFrame, other: pd.DataFrame, target: str, columns: list[str]) -> np.ndarray:
    """Nested training-only means, falling back to the global training mean."""
    global_mean = float(train[target].mean())
    if not columns:
        return np.full(len(other), global_mean)
    lookup = train.groupby(columns, dropna=False)[target].mean().rename("prediction").reset_index()
    pred = other[columns].merge(lookup, how="left", on=columns, sort=False)["prediction"].fillna(global_mean)
    return pred.to_numpy(float)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Parquet/CSV, exactly one row per CpG")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target", default="mean_train")
    p.add_argument("--cpg-id", default="cpg_idx")
    p.add_argument("--chromosome", default="chromosome")
    p.add_argument("--position", default="position")
    p.add_argument("--numeric", default="", help="Comma-separated static numeric features")
    p.add_argument("--categorical", default="", help="Comma-separated static categorical features")
    p.add_argument("--sequence", help="Optional local DNA-sequence column; enables k-mer sequence models")
    p.add_argument("--kmer", type=int, default=3)
    p.add_argument("--split", choices=("random", "block", "fixed"), required=True)
    p.add_argument("--block-size", type=int, default=5_000_000)
    p.add_argument("--seed", type=int, default=20260728)
    p.add_argument("--target-space", choices=("beta", "logit"), default="logit")
    p.add_argument("--stratify", default="cgi_category")
    p.add_argument("--mlp-max-iter", type=int, default=300)
    args = p.parse_args()

    path = Path(args.input)
    df = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    numeric, categorical = _csv(args.numeric), _csv(args.categorical)
    required = [args.cpg_id, args.target, *numeric, *categorical]
    if args.split == "block": required += [args.chromosome, args.position]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Input lacks required columns: {missing}")
    if df[args.cpg_id].duplicated().any():
        raise ValueError("Input must have exactly one row per CpG; duplicate cpg_id values found")
    y = pd.to_numeric(df[args.target], errors="coerce")
    valid = y.notna() & np.isfinite(y) & y.between(0, 1)
    dropped = int((~valid).sum())
    df = df.loc[valid].reset_index(drop=True); df[args.target] = y.loc[valid].to_numpy()
    kmer_features: list[str] = []
    if args.sequence:
        if args.sequence not in df:
            raise ValueError(f"Input lacks sequence column {args.sequence!r}")
        kmer_data = _kmer_frame(df[args.sequence], args.kmer)
        df = pd.concat([df, kmer_data], axis=1); kmer_features = list(kmer_data.columns)
    df["split"] = _partition(df, args.split, args.cpg_id, args.chromosome, args.position, args.block_size, args.seed)
    train, validation, test = (df[df.split == x].copy() for x in ("train", "validation", "test"))
    if min(map(len, (train, validation, test))) == 0:
        raise ValueError("At least one split is empty; change --seed or --block-size")
    if args.split == "block":
        blocks = df[args.chromosome].astype(str) + ":" + (pd.to_numeric(df[args.position]) // args.block_size).astype(str)
        if (pd.DataFrame({"block": blocks, "split": df.split}).groupby("block").split.nunique() > 1).any():
            raise AssertionError("block leakage detected")

    features = numeric + categorical
    if not features:
        raise ValueError("Specify at least one --numeric or --categorical feature")
    def prep(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
        transformations = []
        if numeric_columns:
            transformations.append(("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_columns))
        if categorical_columns:
            transformations.append(("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_columns))
        return ColumnTransformer(transformations, remainder="drop")
    target_train = _target(train[args.target].to_numpy(float), args.target_space)
    models = {
        "ridge": (Pipeline([("features", prep(numeric, categorical)), ("model", Ridge(alpha=10.0))]), features),
        "tabular_mlp": (Pipeline([("features", prep(numeric, categorical)), ("model", MLPRegressor(hidden_layer_sizes=(128, 64),
            early_stopping=True, validation_fraction=0.1, max_iter=args.mlp_max_iter, random_state=args.seed))]),
            features),
    }
    if kmer_features:
        models["sequence_kmer_mlp"] = (Pipeline([("features", prep(kmer_features, [])), ("model", MLPRegressor(
            hidden_layer_sizes=(128, 64), early_stopping=True, validation_fraction=0.1, max_iter=args.mlp_max_iter, random_state=args.seed))]), kmer_features)
        models["sequence_plus_tabular_mlp"] = (Pipeline([("features", prep(numeric + kmer_features, categorical)), ("model", MLPRegressor(
            hidden_layer_sizes=(128, 64), early_stopping=True, validation_fraction=0.1, max_iter=args.mlp_max_iter, random_state=args.seed))]), features + kmer_features)
    predictions: dict[str, np.ndarray] = {
        "global_mean": np.full(len(test), float(train[args.target].mean())),
        "cgi_category_mean": _group_baseline(train, test, args.target, [args.stratify] if args.stratify in df else []),
        "chromosome_mean": _group_baseline(train, test, args.target, [args.chromosome] if args.chromosome in df else []),
        "genomic_region_mean": _group_baseline(train, test, args.target, ["genomic_region"] if "genomic_region" in df else []),
    }
    validation_predictions: dict[str, np.ndarray] = {
        "global_mean": np.full(len(validation), float(train[args.target].mean())),
        "cgi_category_mean": _group_baseline(train, validation, args.target, [args.stratify] if args.stratify in df else []),
        "chromosome_mean": _group_baseline(train, validation, args.target, [args.chromosome] if args.chromosome in df else []),
        "genomic_region_mean": _group_baseline(train, validation, args.target, ["genomic_region"] if "genomic_region" in df else []),
    }
    for name, (model, model_features) in models.items():
        model.fit(train[model_features], target_train)
        predictions[name] = _inverse_target(model.predict(test[model_features]), args.target_space)
        validation_predictions[name] = _inverse_target(model.predict(validation[model_features]), args.target_space)
    truth = test[args.target].to_numpy(float)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "command": " ".join(sys.argv),
        "source": {"path": str(path), "sha256": _file_sha256(path)},
        "design": {"unit": "one row per CpG", "target": args.target, "target_space": args.target_space,
                   "split": args.split, "block_size": args.block_size if args.split == "block" else None,
                   "seed": args.seed, "numeric_features": numeric, "categorical_features": categorical,
                   "sequence": args.sequence, "kmer": args.kmer if args.sequence else None},
        "rows": {"input": int(len(df) + dropped), "dropped_invalid_target": dropped, "train": int(len(train)),
                 "validation_unused_for_model_selection": int(len(validation)), "test": int(len(test))},
        "test": {name: _metrics(truth, pred, test.reset_index(drop=True), args.stratify) for name, pred in predictions.items()},
    }
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    test_out = test[[args.cpg_id, args.target, "split"]].copy()
    for name, pred in predictions.items(): test_out["pred_" + name] = pred
    test_out.to_parquet(out / "test_predictions.parquet", index=False)
    validation_out = validation[[args.cpg_id, args.target, "split"]].copy()
    for name, pred in validation_predictions.items(): validation_out["pred_" + name] = pred
    pd.concat([validation_out, test_out], ignore_index=True).to_parquet(out / "locus_predictions.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output_dir": str(out), "models": {k: v["mse"] for k, v in report["test"].items()}}, indent=2))


if __name__ == "__main__":
    main()
