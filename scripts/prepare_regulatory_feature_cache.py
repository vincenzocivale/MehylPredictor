#!/usr/bin/env python3
"""Build independent 1536D regulatory features for the unchanged RNA model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from methylation_predictor.regulatory_pca import RegulatoryPCA
from methylation_predictor.rna_training.splits import blocked_cpg_split
from methylation_predictor.storage import FunctionalLocusCache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("/dune/DATASETS/MethylPredictionData"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["development", "final"], default="development")
    p.add_argument("--chunk-size", type=int, default=4096)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.chunk_size <= 0:
        p.error("--chunk-size must be positive")
    data = args.data_root
    training_ids = np.load(data / "derived/methylprophet_table5_tcga_chr1/table5_protocol/array_train_cpg_idx.npy")
    if args.mode == "development":
        training_ids, _ = blocked_cpg_split(
            registry=data / "datasets/methylprophet_repro_v1/cpg/registries/array_cpg_map.parquet",
            cpg_ids=training_ids, val_fraction=0.1, seed=17,
        )
    cache = FunctionalLocusCache(data / "derived/ntv3_functional_peak_atlas_chr1_all_sources",
                               data / "derived/ntv3_probe_targets/chr1_annotation_features_all_sources")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    required_bytes = len(cache.ids) * (1536 * 2 + 16)
    if shutil.disk_usage(args.output.parent).free < required_bytes + 2 * 1024**3:
        raise RuntimeError("insufficient disk space for feature cache and 2 GiB margin")
    args.output.mkdir()
    (args.output / "manifest.json").write_text(json.dumps({
        "feature_type": "regulatory_pca", "status": "building",
    }) + "\n")
    print(f"Fitting 1536D PCA on {len(training_ids)} training loci", flush=True)
    projection = RegulatoryPCA.fit(cache, training_ids)
    projection.save(args.output / "projection.npz")
    np.save(args.output / "cpg_idx.npy", np.asarray(cache.ids))
    # The current RNA model never uses prior/sigma in prediction or training
    # loss. These satisfy the shared cache schema; their diagnostic baseline
    # is explicitly constant and contains no NTv3-derived information.
    np.save(args.output / "prior.npy", np.full(len(cache.ids), 0.5, np.float32))
    np.save(args.output / "sigma.npy", np.ones(len(cache.ids), np.float32))
    embeddings = np.lib.format.open_memmap(args.output / "embeddings.f16.npy", mode="w+",
                                          dtype=np.float16, shape=(len(cache.ids), 1536))
    content_hash = hashlib.sha256()
    for start in range(0, len(cache.ids), args.chunk_size):
        batch = cache.get(cache.ids[start:start + args.chunk_size])
        transformed = projection.transform(batch).astype(np.float16)
        if not np.isfinite(transformed).all():
            raise ValueError(f"non-finite projected features at row {start}")
        embeddings[start:start + len(transformed)] = transformed
        content_hash.update(transformed.tobytes())
        if start % (args.chunk_size * 25) == 0:
            print(f"Projected {start + len(transformed)}/{len(cache.ids)} loci", flush=True)
    embeddings.flush()
    manifest = {
        "schema_version": 1, "status": "complete", "feature_type": "regulatory_pca",
        "mode": args.mode, "rows": len(cache.ids), "embedding_dim": 1536,
        "train_cpg_sha256": hashlib.sha256(np.sort(training_ids).astype(np.int64).tobytes()).hexdigest(),
        "train_cpg_count": len(training_ids), "uses_methylation_labels": False, "teacher": None,
        "source_atlas": str(cache.functional_atlas_root.resolve()),
        "annotation_cache": str(cache.annotation_cache_root.resolve()),
        "explained_variance_ratio": projection.explained_variance_ratio,
        "prior_kind": "constant_0.5_for_diagnostic_only", "sigma_kind": "constant_1_unused",
        "projection_sha256": hashlib.sha256((args.output / "projection.npz").read_bytes()).hexdigest(),
        "embedding_content_sha256": content_hash.hexdigest(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
