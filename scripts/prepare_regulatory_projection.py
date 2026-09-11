#!/usr/bin/env python3
"""Initialize a regulatory projection from training-locus covariance, without NTv3."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from methylation_predictor.regulatory_embedding import StandardizedTrackEmbedding
from methylation_predictor.rna_training.splits import blocked_cpg_split
from methylation_predictor.storage import FunctionalLocusCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/dune/DATASETS/MethylPredictionData"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["development", "final"], default="development")
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = args.data_root
    ids = np.load(data / "derived/methylprophet_table5_tcga_chr1/table5_protocol/array_train_cpg_idx.npy")
    if args.mode == "development":
        ids, _ = blocked_cpg_split(
            registry=data / "datasets/methylprophet_repro_v1/cpg/registries/array_cpg_map.parquet",
            cpg_ids=ids, val_fraction=0.1, seed=17,
        )
    cache = FunctionalLocusCache(data / "derived/ntv3_functional_peak_atlas_chr1_all_sources",
                               data / "derived/ntv3_probe_targets/chr1_annotation_features_all_sources")
    encoder = StandardizedTrackEmbedding(4165, args.width)
    encoder.fit(cache, ids)
    batch = cache.get(ids)
    matrix = sparse.csr_matrix((np.ones(len(batch["track_indices"]), np.float32),
                                batch["track_indices"], batch["offsets"]), shape=(len(ids), 4165)).toarray()
    matrix -= encoder.frequency.numpy()
    matrix *= encoder.inverse_scale.numpy()
    print(f"fitting PCA on {matrix.shape}, mode={args.mode}, width={args.width}", flush=True)
    pca = PCA(n_components=args.width, svd_solver="randomized", iterated_power=3, random_state=17).fit(matrix)
    # Whiten the retained components: the RNA fusion receives comparable
    # input scales instead of being dominated by redundant common tracks.
    weights = (pca.components_.T / np.sqrt(pca.explained_variance_.clip(1e-6))).astype(np.float32)
    with torch.no_grad():
        encoder.weight.copy_(torch.from_numpy(weights))
    payload = {
        "schema_version": 1, "mode": args.mode, "width": args.width,
        "train_cpg_sha256": hashlib.sha256(np.sort(ids).astype(np.int64).tobytes()).hexdigest(),
        "train_cpg_count": len(ids), "state_dict": encoder.state_dict(),
        "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        "teacher": None, "uses_methylation_labels": False,
        "source_atlas": str(cache.functional_atlas_root.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    manifest = {k: v for k, v in payload.items() if k != "state_dict"}
    manifest["projection_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        main()
