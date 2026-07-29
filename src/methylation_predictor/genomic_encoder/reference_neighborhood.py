#!/usr/bin/env python3
"""Experiment 3: leakage-safe reference-neighborhood priors for new CpGs.

The input has one row per CpG and a training-sample-only ``mean_train``.
For every target, reference values are restricted to CpGs in the locus training
partition.  The target itself is never a reference.  ``interpolation`` allows
nearby training loci; ``block_ood`` additionally removes all references within
``--exclusion-radius`` bp of every validation/test target.

This deliberately predicts a locus-level prior, not a patient's methylation.
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
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import torch
from torch import nn

from methylation_predictor.genomic_encoder.static_prior import (_csv, _file_sha256, _inverse_target,
                                           _metrics, _partition, _target, _kmer_frame)


def _stable_seed(value: str, seed: int) -> int:
    return int.from_bytes(hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest(), "little") % (2 ** 32)


def _nearest_indices(reference_pos: np.ndarray, target_pos: int, max_neighbors: int,
                     exclusion_radius: int, target_row: int | None) -> np.ndarray:
    """Return at most K references on each side, ordered left then right."""
    where = np.searchsorted(reference_pos, target_pos)
    left, right = [], []
    cursor = where - 1
    while cursor >= 0 and len(left) < max_neighbors:
        if abs(int(reference_pos[cursor]) - target_pos) > exclusion_radius and cursor != target_row:
            left.append(cursor)
        cursor -= 1
    cursor = where
    while cursor < len(reference_pos) and len(right) < max_neighbors:
        if abs(int(reference_pos[cursor]) - target_pos) > exclusion_radius and cursor != target_row:
            right.append(cursor)
        cursor += 1
    return np.asarray(left[::-1] + right, dtype=np.int64)


def build_neighborhood_features(frame: pd.DataFrame, reference: pd.DataFrame, *, target: str,
                                chromosome: str, position: str, cgi: str | None,
                                variance: str | None, count: str | None, max_neighbors: int,
                                windows: tuple[int, ...], exclusion_radius: int,
                                exclude_self: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Build fixed-width neighbor and multiscale features from allowed references.

    ``reference`` contains training CpGs only.  It is intentionally passed as a
    separate frame so callers cannot accidentally use validation/test values.
    The returned baseline is an exponential distance-weighted reference mean.
    """
    if reference.empty:
        raise ValueError("reference set is empty")
    features, feature_index = [], []
    baseline = pd.Series(np.nan, index=frame.index, dtype=float)
    columns = ["neighbor_mean", "neighbor_sd", "neighbor_log_distance", "neighbor_direction",
               "neighbor_cgi_match", "neighbor_log_n"]
    for side in ("left", "right"):
        columns.extend([f"{side}_{j}_{name}" for j in range(max_neighbors) for name in
                        ("mean", "sd", "log_distance", "cgi_match", "log_n")])
    for w in windows:
        columns.extend([f"w{w}_{name}" for name in ("mean", "median", "sd", "min", "max", "n", "nearest_log_distance")])

    for chrom, targets in frame.groupby(chromosome, sort=False):
        refs = reference[reference[chromosome] == chrom].sort_values(position).copy()
        if refs.empty:
            features.extend([np.full(len(columns), np.nan)] * len(targets)); feature_index.extend(targets.index); continue
        ref_pos = refs[position].to_numpy(np.int64)
        ref_mean = refs[target].to_numpy(float)
        ref_sd = (np.sqrt(np.maximum(refs[variance].to_numpy(float), 0)) if variance and variance in refs
                  else np.full(len(refs), np.nan))
        ref_n = (refs[count].to_numpy(float) if count and count in refs else np.full(len(refs), np.nan))
        ref_ids = refs.index.to_numpy()
        for row_id, row in targets.iterrows():
            p = int(row[position]); self_row = row_id if exclude_self else None
            # Index is a label in the original frame, not necessarily an offset.
            eligible = np.ones(len(refs), dtype=bool)
            if exclude_self:
                eligible &= ref_ids != self_row
            eligible &= np.abs(ref_pos - p) > exclusion_radius
            dist_all = np.abs(ref_pos - p)
            # Nearest tokens are not capped by the largest aggregate window:
            # a sparse genomic region must expose that its anchors are distant.
            eligible_ix = np.flatnonzero(eligible)
            ordered = eligible_ix[np.argsort(dist_all[eligible_ix], kind="stable")]
            chosen_left = [x for x in ordered if ref_pos[x] < p][:max_neighbors]
            chosen_right = [x for x in ordered if ref_pos[x] >= p][:max_neighbors]
            chosen_left.sort(key=lambda x: ref_pos[x]); chosen_right.sort(key=lambda x: ref_pos[x])
            vector: list[float] = []
            # Aggregate scalar neighbor descriptors (all selected values).
            chosen = np.asarray(chosen_left + chosen_right, dtype=int)
            if len(chosen):
                d = dist_all[chosen].astype(float)
                # The common exp(-min(d)/tau) factor cancels in a normalized
                # weighted mean; subtracting it prevents all weights underflowing
                # for a block-held-out target many Mb from its references.
                weight = np.exp(-(d - d.min()) / 10_000.0)
                baseline.loc[row_id] = np.average(ref_mean[chosen], weights=weight)
                chosen_sd, chosen_n = ref_sd[chosen], ref_n[chosen]
                vector.extend([float(np.mean(ref_mean[chosen])), float(np.nanmean(chosen_sd)) if np.isfinite(chosen_sd).any() else np.nan,
                               float(np.mean(np.log1p(d))), float(np.mean(np.sign(ref_pos[chosen] - p))),
                               float(np.mean(refs[cgi].iloc[chosen].astype(str).to_numpy() == str(row[cgi]))) if cgi else np.nan,
                               float(np.nanmean(np.log1p(chosen_n))) if np.isfinite(chosen_n).any() else np.nan])
            else:
                vector.extend([np.nan] * 6)
            for selected in (chosen_left, chosen_right):
                for j in range(max_neighbors):
                    if j < len(selected):
                        ix = selected[j]; d = float(dist_all[ix])
                        vector.extend([ref_mean[ix], ref_sd[ix], np.log1p(d),
                                       float(refs[cgi].iloc[ix] == row[cgi]) if cgi else np.nan, np.log1p(ref_n[ix])])
                    else:
                        vector.extend([np.nan] * 5)
            for w in windows:
                ix = np.flatnonzero(eligible & (dist_all <= w))
                if len(ix):
                    vals = ref_mean[ix]
                    vector.extend([np.mean(vals), np.median(vals), np.std(vals), np.min(vals), np.max(vals), len(ix),
                                   np.log1p(dist_all[ix].min())])
                else:
                    vector.extend([np.nan] * 7)
            features.append(np.asarray(vector, dtype=float)); feature_index.append(row_id)
    result = pd.DataFrame(np.vstack(features), index=feature_index, columns=columns).reindex(frame.index)
    return result, baseline


class _NeighborAttention(nn.Module):
    """Target-query attention over fixed left/right reference CpG tokens."""
    def __init__(self, static_dim: int, token_dim: int, hidden: int = 48):
        super().__init__()
        self.query = nn.Sequential(nn.Linear(static_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.key = nn.Sequential(nn.Linear(token_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.value = nn.Sequential(nn.Linear(token_dim, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(static_dim + hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, static: torch.Tensor, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        q, k, v = self.query(static), self.key(tokens), self.value(tokens)
        score = (k * q[:, None]).sum(-1) / np.sqrt(k.shape[-1])
        score = score.masked_fill(~valid, -1e9)
        # No reference token is a valid, explicit state rather than an NaN.
        weight = torch.softmax(score, dim=1) * valid
        weight = weight / weight.sum(1, keepdim=True).clamp_min(1e-8)
        return self.head(torch.cat([static, (weight[..., None] * v).sum(1)], dim=1)).squeeze(1)


def _attention_predictions(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, *, static_numeric: list[str],
                           static_categorical: list[str], neighborhood: pd.DataFrame, max_neighbors: int,
                           target: str, target_space: str, seed: int, epochs: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit target-query attention without exposing target methylation as a token."""
    token_names = [f"{side}_{j}_{name}" for side in ("left", "right") for j in range(max_neighbors)
                   for name in ("mean", "sd", "log_distance", "cgi_match", "log_n")]
    static_fields = static_numeric + static_categorical
    transforms = []
    if static_numeric:
        transforms.append(("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                                              ("scale", StandardScaler())]), static_numeric))
    if static_categorical:
        transforms.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), static_categorical))
    prep = ColumnTransformer(transforms)
    static_train = prep.fit_transform(train[static_fields]); static_val = prep.transform(validation[static_fields]); static_test = prep.transform(test[static_fields])
    def dense(x): return x.toarray() if hasattr(x, "toarray") else x
    combined = pd.concat([train[token_names], validation[token_names], test[token_names]])
    med = combined.iloc[:len(train)].median().fillna(0.0)
    def tokens(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raw = frame[token_names].to_numpy(float).reshape(len(frame), 2 * max_neighbors, 5)
        valid = np.isfinite(raw[..., 0])
        filled = np.where(np.isfinite(raw), raw, np.tile(med.to_numpy().reshape(1, 2 * max_neighbors, 5), (len(frame), 1, 1)))
        return filled.astype(np.float32), valid
    tok_train, mask_train = tokens(train); tok_val, mask_val = tokens(validation); tok_test, mask_test = tokens(test)
    torch.manual_seed(seed); torch.set_num_threads(1)
    model = _NeighborAttention(dense(static_train).shape[1], 5)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x = torch.tensor(dense(static_train), dtype=torch.float32); t = torch.tensor(tok_train); m = torch.tensor(mask_train)
    y = torch.tensor(_target(train[target].to_numpy(float), target_space), dtype=torch.float32)
    model.train()
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for ix in np.array_split(rng.permutation(len(train)), max(1, len(train) // 256)):
            optimiser.zero_grad(); loss = nn.functional.mse_loss(model(x[ix], t[ix], m[ix]), y[ix]); loss.backward(); optimiser.step()
    def predict(static, tok, mask):
        model.eval()
        with torch.no_grad(): return _inverse_target(model(torch.tensor(dense(static), dtype=torch.float32), torch.tensor(tok), torch.tensor(mask)).numpy(), target_space)
    return predict(static_val, tok_val, mask_val), predict(static_test, tok_test, mask_test)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--target", default="mean_train"); p.add_argument("--cpg-id", default="cpg_idx")
    p.add_argument("--chromosome", default="chromosome"); p.add_argument("--position", default="position")
    p.add_argument("--numeric", default=""); p.add_argument("--categorical", default="")
    p.add_argument("--sequence"); p.add_argument("--kmer", type=int, default=3)
    p.add_argument("--cgi", default="cgi_category"); p.add_argument("--variance", default="within_cpg_variance")
    p.add_argument("--count", default="n_train")
    p.add_argument("--split", choices=("random", "block"), required=True); p.add_argument("--block-size", type=int, default=5_000_000)
    p.add_argument("--scenario", choices=("interpolation", "block_ood"), required=True)
    p.add_argument("--exclusion-radius", type=int, default=10_000, help="Required for block_ood; bp")
    p.add_argument("--neighbors-per-side", type=int, default=8); p.add_argument("--windows", default="1000,10000,100000")
    p.add_argument("--seed", type=int, default=20260728); p.add_argument("--target-space", choices=("beta", "logit"), default="logit")
    p.add_argument("--mlp-max-iter", type=int, default=300); p.add_argument("--attention-epochs", type=int, default=200); args = p.parse_args()
    if args.scenario == "block_ood" and args.exclusion_radius <= 0:
        raise ValueError("block_ood requires a positive --exclusion-radius")
    path = Path(args.input); df = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    required = [args.cpg_id, args.target, args.chromosome, args.position]
    missing = sorted(set(required) - set(df));
    if missing: raise ValueError(f"Input lacks required columns: {missing}")
    if df[args.cpg_id].duplicated().any(): raise ValueError("Input must have exactly one row per CpG")
    y = pd.to_numeric(df[args.target], errors="coerce"); valid = y.notna() & np.isfinite(y) & y.between(0, 1)
    dropped = int((~valid).sum()); df = df.loc[valid].reset_index(drop=True); df[args.target] = y.loc[valid].to_numpy()
    numeric, categorical = _csv(args.numeric), _csv(args.categorical)
    for col in numeric + categorical:
        if col not in df: raise ValueError(f"Input lacks requested feature {col!r}")
    kmer = []
    if args.sequence:
        if args.sequence not in df: raise ValueError(f"Input lacks sequence column {args.sequence!r}")
        km = _kmer_frame(df[args.sequence], args.kmer); df = pd.concat([df, km], axis=1); kmer = list(km)
    df["split"] = _partition(df, args.split, args.cpg_id, args.chromosome, args.position, args.block_size, args.seed)
    train, validation, test = (df[df.split == name].copy() for name in ("train", "validation", "test"))
    if min(len(train), len(validation), len(test)) == 0: raise ValueError("At least one split is empty")
    windows = tuple(int(x) for x in _csv(args.windows))
    radius = args.exclusion_radius if args.scenario == "block_ood" else 0
    # Training loci are references for every partition; each own training target is excluded by ID.
    all_features, all_baseline = build_neighborhood_features(df, train, target=args.target, chromosome=args.chromosome,
        position=args.position, cgi=args.cgi if args.cgi in df else None, variance=args.variance if args.variance in df else None,
        count=args.count if args.count in df else None, max_neighbors=args.neighbors_per_side, windows=windows,
        exclusion_radius=radius)
    df = pd.concat([df, all_features], axis=1); df["distance_weighted_baseline"] = all_baseline
    train, validation, test = (df[df.split == name].copy() for name in ("train", "validation", "test"))
    # A reference-free region has no distance-weighted estimate.  Its baseline
    # falls back to the training global mean; the MLP receives explicit missing
    # feature indicators and can distinguish this case.
    fallback_mean = float(train[args.target].mean())
    baseline_missing = {"validation": int(validation.distance_weighted_baseline.isna().sum()),
                        "test": int(test.distance_weighted_baseline.isna().sum())}
    test.loc[:, "distance_weighted_baseline"] = test.distance_weighted_baseline.fillna(fallback_mean)
    validation.loc[:, "distance_weighted_baseline"] = validation.distance_weighted_baseline.fillna(fallback_mean)
    neighborhood = list(all_features.columns); static = numeric + kmer
    variants = {"A_sequence": (static, []), "B_sequence_annotations": (static, categorical),
                "C_neighborhood": (neighborhood, []), "D_sequence_neighborhood": (static + neighborhood, []),
                "E_sequence_annotations_neighborhood": (static + neighborhood, categorical)}
    y_train = _target(train[args.target].to_numpy(float), args.target_space)
    predictions, validation_predictions = {"distance_weighted": test.distance_weighted_baseline.to_numpy()}, {"distance_weighted": validation.distance_weighted_baseline.to_numpy()}
    for name, (number_fields, category_fields) in variants.items():
        if not number_fields and not category_fields: continue
        transforms = []
        if number_fields:
            transforms.append(("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                                                      ("scale", StandardScaler())]), number_fields))
        if category_fields:
            transforms.append(("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), category_fields))
        fields = number_fields + category_fields
        model = Pipeline([("features", ColumnTransformer(transforms)),
                          ("model", MLPRegressor(hidden_layer_sizes=(128, 64), early_stopping=True, validation_fraction=.1,
                                                   max_iter=args.mlp_max_iter, random_state=_stable_seed(name, args.seed)))])
        model.fit(train[fields], y_train)
        predictions[name] = _inverse_target(model.predict(test[fields]), args.target_space)
        validation_predictions[name] = _inverse_target(model.predict(validation[fields]), args.target_space)
    if static:
        att_val, att_test = _attention_predictions(train, validation, test, static_numeric=static,
            static_categorical=categorical, neighborhood=all_features, max_neighbors=args.neighbors_per_side,
            target=args.target, target_space=args.target_space, seed=_stable_seed("attention", args.seed), epochs=args.attention_epochs)
        predictions["F_attention_sequence_annotations_neighborhood"] = att_test
        validation_predictions["F_attention_sequence_annotations_neighborhood"] = att_val
    truth = test[args.target].to_numpy(float)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "command": " ".join(sys.argv),
              "source": {"path": str(path), "sha256": _file_sha256(path)},
              "design": {"target": args.target, "target_is_training_samples_only": True, "split": args.split,
                         "scenario": args.scenario, "exclusion_radius_bp": radius, "neighbors_per_side": args.neighbors_per_side,
                         "windows_bp": windows, "reference_partition": "train", "target_excluded_from_reference": True,
                         "target_space": args.target_space, "neighborhood_nan_handling": "median imputation plus missingness indicator",
                         "attention_epochs": args.attention_epochs,
                         "variants": {k: v for k, v in variants.items() if v}},
              "rows": {"input": len(df) + dropped, "dropped_invalid_target": dropped, "train": len(train), "validation": len(validation), "test": len(test),
                       "distance_weighted_global_mean_fallback": baseline_missing,
                       "neighborhood_nan_cells": {"train": int(train[neighborhood].isna().sum().sum()),
                                                   "validation": int(validation[neighborhood].isna().sum().sum()),
                                                   "test": int(test[neighborhood].isna().sum().sum())}},
              "test": {name: _metrics(truth, pred, test.reset_index(drop=True), args.cgi if args.cgi in df else None) for name, pred in predictions.items()}}
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cols = [args.cpg_id, args.target, "split", "distance_weighted_baseline"]
    test_out, val_out = test[cols].copy(), validation[cols].copy()
    for name, pred in predictions.items(): test_out["pred_" + name] = pred
    for name, pred in validation_predictions.items(): val_out["pred_" + name] = pred
    test_out.to_parquet(out / "test_predictions.parquet", index=False)
    pd.concat([val_out, test_out], ignore_index=True).to_parquet(out / "locus_predictions.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output_dir": str(out), "mse": {k: v["mse"] for k, v in report["test"].items()}}, indent=2))


if __name__ == "__main__": main()
