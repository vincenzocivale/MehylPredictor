#!/usr/bin/env python3
"""CPU probe of locus representations on the existing inner genomic split.

Only official training CpGs/samples are read. A response basis is learned on
inner training CpGs and training samples; ridge predicts its coefficients at
unseen CpGs. This is a seen-sample representation diagnostic, NOT an RNA model
or an official/double-OOD benchmark. All preprocessing fits on inner train.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from methylation_predictor.storage import FunctionalLocusCache, SortedIndex
from methylation_predictor.rna_training.splits import blocked_cpg_split, stratified_sample_split

DATA = Path("/dune/DATASETS/MethylPredictionData")


def digest(values):
    return hashlib.sha256(np.asarray(values, np.int64).tobytes()).hexdigest()


def response_basis(beta, train, rank, seed):
    """Missing values are imputed by that locus's observed sample mean.

    Validation rows are projected onto a fixed training basis for diagnostics
    only; they cannot influence the basis or the fitted predictor.
    """
    mean = np.nanmean(beta, axis=1)
    if not np.isfinite(mean).all():
        raise ValueError("a locus has no observations in the selected samples")
    residual = np.where(np.isfinite(beta), beta - mean[:, None], 0).astype(np.float32)
    _, _, basis = randomized_svd(residual[train], n_components=rank, random_state=seed)
    targets = np.column_stack([mean, residual @ basis.T])
    return targets, basis


def score_response(truth, prediction):
    valid = np.isfinite(truth)
    count = valid.sum(1)
    mean = np.nansum(truth, axis=1) / np.maximum(count, 1)
    pred_mean = np.where(valid, prediction, 0).sum(1) / np.maximum(count, 1)
    a = np.where(valid, truth - mean[:, None], 0)
    b = np.where(valid, prediction - pred_mean[:, None], 0)
    denom = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    ok = (count > 1) & (denom > 1e-12)
    corr = (a[ok] * b[ok]).sum(1) / denom[ok]
    mean_sse = np.square(mean - pred_mean).sum()
    return {
        "response_median_pcc": float(np.median(corr)),
        "valid_correlation_loci": int(ok.sum()),
        "mean_r2": float(1 - mean_sse / np.square(mean - mean.mean()).sum()),
        "mse": float(np.square(truth[valid] - prediction[valid]).mean()),
    }


def learned_f0(cache, ids, checkpoint):
    import torch
    torch.set_num_threads(4)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["mode"] != "development":
        raise ValueError("learned representation must come from a development checkpoint")
    state = payload["model_state"]
    bag = torch.nn.EmbeddingBag(4165, 256, mode="mean", include_last_offset=True)
    dense = torch.nn.Sequential(torch.nn.LayerNorm(23), torch.nn.Linear(23, 256), torch.nn.GELU())
    norm = torch.nn.LayerNorm(256)
    for module, prefix in [(bag, "track_embedding."), (dense, "dense_encoder."), (norm, "locus_norm.")]:
        module.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
    chunks = []
    with torch.no_grad():
        for start in range(0, len(ids), 256):
            batch = cache.get(ids[start:start + 256])
            encoded = bag(torch.from_numpy(batch["track_indices"]), torch.from_numpy(batch["offsets"]))
            chunks.append(norm(encoded + dense(torch.from_numpy(batch["dense"]))).numpy())
    return np.concatenate(chunks)


def multiscale_context(cache, ids):
    """Unlabelled CpG-sampled neighborhood breadth, NOT continuous BigWig signal.

    Uses the full immutable genomic feature atlas, with no methylation/RNA
    labels and no trainable transformation. Report the sampling limitation.
    """
    frame = pd.read_csv(cache.functional_atlas_root / "cpg_coordinates.tsv.gz", sep="\t")
    if frame["chr"].nunique() != 1:
        raise ValueError("this diagnostic currently requires a single-chromosome atlas")
    coordinate_rows = SortedIndex(frame.cpg_idx.to_numpy()).positions_of(cache.ids)
    positions = frame.position.to_numpy()[coordinate_rows]
    order = np.argsort(positions)
    positions_sorted = positions[order]
    breadth = np.asarray(cache.breadth[order], dtype=np.float64)
    prefix = np.vstack([np.zeros((1, breadth.shape[1])), np.cumsum(breadth, axis=0)])
    query = positions[cache.index.positions_of(ids)]
    features = []
    for radius in [128, 512, 2048, 8192, 16384]:
        left = np.searchsorted(positions_sorted, query - radius, side="left")
        right = np.searchsorted(positions_sorted, query + radius, side="right")
        count = right - left
        features.append((prefix[right] - prefix[left]) / np.maximum(count[:, None], 1))
        features.append(np.log1p(count[:, None]))
    return np.column_stack(features).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--alphas", default="10,100,1000,10000")
    parser.add_argument("--arms", default="ntv3_post,raw_binary,mean_normalized,expanded_annotations,learned_f0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    data = args.data_root
    prepared = data / "derived/methylprophet_table5_tcga_chr1"
    canonical = data / "datasets/methylprophet_repro_v1"
    proto = prepared / "table5_protocol"
    official_cpgs = np.load(proto / "array_train_cpg_idx.npy")
    official_samples = np.load(proto / "array_train_sample_idx.npy")
    train_ids, val_ids = blocked_cpg_split(
        registry=canonical / "cpg/registries/array_cpg_map.parquet", cpg_ids=official_cpgs,
        val_fraction=0.1, seed=17,
    )
    sample_train, _ = stratified_sample_split(
        canonical_root=canonical, sample_ids=official_samples, val_fraction=0.1, seed=17,
    )
    rng = np.random.default_rng(args.seed)
    samples = np.sort(rng.choice(sample_train, size=min(args.samples, len(sample_train)), replace=False))
    ids = np.concatenate([train_ids, val_ids])
    train = np.arange(len(train_ids))
    val = np.arange(len(train_ids), len(ids))
    # Selecting columns first is bounded (~150 MB per 1,024 CpGs) and never
    # includes official validation CpG labels in an in-memory matrix.
    beta = np.empty((len(ids), len(samples)), np.float32)
    with h5py.File(prepared / "methylation/array_table5_chr1.h5", "r") as handle:
        sample_rows = SortedIndex(handle["sample_idx"][:]).positions_of(samples)
        cpg_rows = SortedIndex(handle["cpg_idx"][:]).positions_of(ids)
        for start in range(0, len(ids), 1024):
            rows = cpg_rows[start:start + 1024]
            order = np.argsort(rows)
            block = handle["beta"][:, rows[order]][sample_rows]
            beta[start:start + len(rows)] = block[:, np.argsort(order)].T
    print(f"targets: train_loci={len(train)} val_loci={len(val)} samples={len(samples)}", flush=True)
    targets, basis = response_basis(beta, train, args.rank, args.seed)
    np.savez(args.output / "axes.npz", train_cpg_idx=train_ids, val_cpg_idx=val_ids, sample_idx=samples)
    report = {
        "protocol": "inner_unseen_cpg_seen_sample_response_probe",
        "not_official_or_double_ood": True,
        "training_cpg_sha256": digest(train_ids), "validation_cpg_sha256": digest(val_ids),
        "sample_sha256": digest(samples), "n_samples": len(samples), "rank": args.rank,
        "response_basis_fit": "inner_train_cpgs_and_inner_train_samples_only",
        "oracle_projection": score_response(beta[val], targets[val, :1] + targets[val, 1:] @ basis),
        "results": [],
    }
    cache = FunctionalLocusCache(
        data / "derived/ntv3_functional_peak_atlas_chr1_all_sources",
        data / "derived/ntv3_probe_targets/chr1_annotation_features_all_sources",
    )
    batch = cache.get(ids)
    binary = sparse.csr_matrix((np.ones(len(batch["track_indices"]), np.float32),
                               batch["track_indices"], batch["offsets"]), shape=(len(ids), 4165)).toarray()
    for arm in args.arms.split(","):
        print(f"probe {arm}", flush=True)
        if arm == "ntv3_post":
            root = prepared / "features"
            rows = SortedIndex(np.load(root / "cpg_idx.npy")).positions_of(ids)
            features = np.asarray(np.load(root / "embeddings.f16.npy", mmap_mode="r")[rows], np.float32)
        elif arm == "learned_f0":
            checkpoint = data / "experiments/runs/locus_cls_joint/chr1/functional-fusion-f0-screen-seed17-e30/checkpoints/best.pt"
            features = learned_f0(cache, ids, checkpoint)
            report["learned_f0_checkpoint"] = str(checkpoint)
        elif arm in {"raw_binary", "mean_normalized", "expanded_annotations",
                     "raw_context", "raw_kmer", "raw_context_kmer",
                     "raw_random256", "raw_pca256", "raw_pca1536"}:
            tracks = binary
            dense = batch["dense"]
            if arm == "mean_normalized":
                tracks = binary / np.maximum(binary.sum(1, keepdims=True), 1)
            elif arm == "expanded_annotations":
                expanded = np.load(cache.annotation_cache_root / "annotation_expanded.f32.npy", mmap_mode="r")
                dense = np.column_stack([expanded[cache.index.positions_of(ids)], dense[:, 18:]])
            features = np.column_stack([tracks, dense])
            if "context" in arm:
                features = np.column_stack([features, multiscale_context(cache, ids)])
                report["context_semantics"] = "CpG-sampled breadth at 128/512/2048/8192/16384 bp; not BigWig"
            if "kmer" in arm:
                kmer_root = data / "derived/rna_feature_cache/chr1_kmer5_32768"
                rows = SortedIndex(np.load(kmer_root / "cpg_idx.npy")).positions_of(ids)
                kmer = np.asarray(np.load(kmer_root / "embeddings.f16.npy", mmap_mode="r")[rows], np.float32)
                features = np.column_stack([features, kmer])
                report["kmer_semantics"] = "hg38 5-mer frequencies over 32768 bp, no foundation-model features or priors"
            if "random" in arm or "pca" in arm:
                raw_scaler = StandardScaler().fit(features[train])
                normalized = raw_scaler.transform(features)
                if "random" in arm:
                    projection = np.random.default_rng(args.seed).normal(
                        scale=features.shape[1] ** -0.5, size=(features.shape[1], 256),
                    ).astype(np.float32)
                    features = normalized @ projection
                else:
                    width = int(arm.split("pca")[1])
                    pca = PCA(n_components=width, svd_solver="randomized", iterated_power=3,
                              random_state=args.seed).fit(normalized[train])
                    features = pca.transform(normalized)
                    report.setdefault("pca_explained_variance", {})[arm] = float(pca.explained_variance_ratio_.sum())
        else:
            raise ValueError(f"unknown arm {arm}")
        scaler = StandardScaler().fit(features[train])
        xtrain = scaler.transform(features[train])
        xval = scaler.transform(features[val])
        for alpha in map(float, args.alphas.split(",")):
            probe = Ridge(alpha=alpha, solver="cholesky").fit(xtrain, targets[train])
            predicted = probe.predict(xval)
            metrics = score_response(beta[val], predicted[:, :1] + predicted[:, 1:] @ basis)
            result = {"arm": arm, "alpha": alpha, "dimensions": features.shape[1], **metrics}
            report["results"].append(result)
            print(json.dumps(result), flush=True)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"completed: {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
