#!/usr/bin/env python3
"""CLI for full E2-E4 experiments with NTv3 feature expansion.

Import pyarrow before the package on this host; h5py/torch may otherwise load an
older libstdc++ before pyarrow.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow  # noqa: F401
import h5py
import numpy as np
import pandas as pd
import yaml

from methylation_predictor.full_suite.cache import build_compact_source_cache, prepare_rna_cache
from methylation_predictor.full_suite.expansion import (
    extract_ntv3_worker,
    merge_ntv3_shards,
    prepare_missing_universe,
)
from methylation_predictor.full_suite.feature_store import prepare_base_cache
from methylation_predictor.full_suite.probe import fit_feature_extension_probe, infer_expanded_features
from methylation_predictor.full_suite.trainer import MixedSuiteRunner, _load_suite_protocol


def _print(value):
    print(json.dumps(value, indent=2, default=str), flush=True)


def cmd_prepare_base(a):
    _print(prepare_base_cache(a.embeddings, a.features, a.output, storage_dtype=a.storage_dtype))


def cmd_prepare_universe(a):
    _print(prepare_missing_universe(a.canonical_root, a.embeddings, a.output, protocol=a.protocol, shard_size=a.shard_size))


def cmd_extract(a):
    _print(extract_ntv3_worker(a.universe, a.fasta, a.output, rank=a.rank, world_size=a.world_size,
                               checkpoint=a.checkpoint, length=32768, batch_size=a.batch_size,
                               device=a.device, bf16=not a.no_bf16, storage_dtype=a.storage_dtype))


def cmd_merge(a):
    _print(merge_ntv3_shards(a.universe, a.shards, a.output, storage_dtype=a.storage_dtype))


def cmd_fit_probe(a):
    fit_ids = None
    if a.cpg_split:
        split = pd.read_parquet(a.cpg_split, columns=["cpg_idx", "split"])
        fit_ids = split.loc[~split.split.eq("train"), "cpg_idx"].to_numpy(np.int64)
        if len(fit_ids) != 81_493:
            raise ValueError(f"expected 81,493 official-heldout Array CpGs for feature-probe distillation, got {len(fit_ids)}")
    _print(fit_feature_extension_probe(a.base_cache, a.output, device=a.device,
                                       max_epochs=a.epochs, patience=a.patience,
                                       block_rows=a.block_rows, fit_cpg_ids=fit_ids))


def cmd_infer_features(a):
    _print(
        infer_expanded_features(
            a.embeddings, a.probe, a.output, device=a.device, batch_rows=a.batch_rows
        )
    )


def cmd_rna_cache(a):
    p = _load_suite_protocol(a.canonical_root, "tcga_mix_chr1")
    _print(prepare_rna_cache(a.canonical_root, a.output, train_sample_idx=p.array_train_sample_idx))


def cmd_compact_cache(a):
    result = {}
    for source in ("array", "epic"):
        result[source] = build_compact_source_cache(
            a.canonical_root, a.protocol, source, Path(a.output) / f"{source}_{a.protocol}.h5", row_chunk=a.row_chunk
        )
    _print(result)


def cmd_train(a):
    runner = MixedSuiteRunner(
        canonical_root=a.canonical_root,
        protocol_name=a.protocol,
        base_config=a.base_config,
        base_feature_cache=a.base_cache,
        expanded_embedding_dir=a.expanded_embeddings,
        expanded_feature_dir=a.expanded_features,
        rna_cache_dir=a.rna_cache,
        output_dir=a.output,
        source_policy=a.source_policy,
        holdout_policy=a.holdout_policy,
        genomewide_cpg_split=a.genomewide_cpg_split,
        array_cache=a.array_cache,
        epic_cache=a.epic_cache,
        seed=a.seed,
        mixed_steps_per_epoch=a.steps_per_epoch,
        dev_fraction=a.dev_fraction,
        block_rows={"array": a.array_rows, "epic": a.epic_rows, "wgbs": a.wgbs_rows},
        block_cpgs={"array": a.array_cpgs, "epic": a.epic_cpgs, "wgbs": a.wgbs_cpgs},
    )
    try:
        _print(runner.run())
    finally:
        runner.close()



def cmd_write_eval_adapter(a):
    root = Path(a.canonical_root)
    protocol = _load_suite_protocol(root, "tcga_mix_chr1")
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    manifests = out / "manifests"; manifests.mkdir(exist_ok=True)
    with h5py.File(root / "rna" / "tcga_rna_official_full.h5", "r") as h:
        ids = np.asarray(h["sample_idx"][...], np.int64)
        labels_raw = h["tissue_name"][...] if "tissue_name" in h else np.full(len(ids), b"UNKNOWN")
    labels = np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in labels_raw], dtype=object)
    lookup = {int(i): str(l) if str(l) else "UNKNOWN" for i, l in zip(ids, labels)}
    sample = pd.DataFrame({
        "sample_idx": np.concatenate([protocol.array_train_sample_idx, protocol.array_val_sample_idx]),
        "cancer_type": [lookup[int(x)] for x in np.concatenate([protocol.array_train_sample_idx, protocol.array_val_sample_idx])],
        "split": np.concatenate([np.full(len(protocol.array_train_sample_idx), "train", object), np.full(len(protocol.array_val_sample_idx), "validation", object)]),
    })
    sample.to_parquet(manifests / "sample_metadata.parquet", index=False)
    pd.DataFrame({
        "cpg_idx": np.concatenate([protocol.array_train_cpg_idx, protocol.array_val_cpg_idx]),
        "split": np.concatenate([np.full(len(protocol.array_train_cpg_idx), "train", object), np.full(len(protocol.array_val_cpg_idx), "validation", object)]),
    }).to_parquet(manifests / "cpg_splits.parquet", index=False)
    raw = yaml.safe_load(Path(a.base_config).read_text())
    raw["output_dir"] = str(out / "runner")
    raw["run_name"] = "mixed-e2-chr1-eval-adapter"
    raw["data"]["rna"] = {"path": str(root / "rna" / "tcga_rna_official_full.h5"), "values_key": "X", "row_ids_key": "sample_idx", "col_ids_key": "gene_ids"}
    raw["data"]["methylation"] = {"path": str(root / "methylation/tcga_array_official_full.h5"), "values_key": "beta", "row_ids_key": "sample_idx", "col_ids_key": "cpg_idx"}
    raw["data"]["locus_embeddings"] = {"path": a.embeddings, "values_key": "embeddings", "row_ids_key": "cpg_idx"}
    raw["data"]["locus_features"] = {"path": a.features, "id_column": "cpg_idx"}
    raw["data"]["sample_metadata"] = {"path": str(manifests / "sample_metadata.parquet"), "id_column": "sample_idx"}
    raw["data"]["cpg_splits"] = {"path": str(manifests / "cpg_splits.parquet"), "id_column": "cpg_idx"}
    raw["data"]["allow_partial_overlap"] = False
    raw["data"]["standardize_rna"] = True; raw["data"]["rna_transform"] = "zscore"
    raw["tracking"] = {"backend": "none", "log_checkpoint": False}
    path = out / "eval_config.yaml"; path.write_text(yaml.safe_dump(raw, sort_keys=False))
    _print({"config": str(path)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("prepare-base-cache")
    q.add_argument("--embeddings", required=True); q.add_argument("--features", required=True); q.add_argument("--output", required=True)
    q.add_argument("--storage-dtype", choices=("float16", "float32"), default="float32"); q.set_defaults(func=cmd_prepare_base)

    q = sub.add_parser("prepare-universe")
    q.add_argument("--canonical-root", required=True); q.add_argument("--embeddings", required=True); q.add_argument("--output", required=True)
    q.add_argument("--protocol", default="tcga_mix_chr123"); q.add_argument("--shard-size", type=int, default=25000); q.set_defaults(func=cmd_prepare_universe)

    q = sub.add_parser("extract-ntv3-worker")
    q.add_argument("--universe", required=True); q.add_argument("--fasta", required=True); q.add_argument("--output", required=True)
    q.add_argument("--rank", type=int, required=True); q.add_argument("--world-size", type=int, required=True)
    q.add_argument("--checkpoint", default="InstaDeepAI/NTv3_650M_post"); q.add_argument("--batch-size", type=int, default=4)
    q.add_argument("--device", default="cuda"); q.add_argument("--no-bf16", action="store_true")
    q.add_argument("--storage-dtype", choices=("float16", "float32"), default="float32"); q.set_defaults(func=cmd_extract)

    q = sub.add_parser("merge-ntv3")
    q.add_argument("--universe", required=True); q.add_argument("--shards", required=True); q.add_argument("--output", required=True)
    q.add_argument("--storage-dtype", choices=("float16", "float32"), default="float32"); q.set_defaults(func=cmd_merge)

    q = sub.add_parser("fit-feature-probe")
    q.add_argument("--base-cache", required=True); q.add_argument("--output", required=True); q.add_argument("--device", default="cuda")
    q.add_argument("--epochs", type=int, default=40); q.add_argument("--patience", type=int, default=6); q.add_argument("--block-rows", type=int, default=4096)
    q.add_argument("--cpg-split", help="genome-wide Array split manifest; if set, distil only validation+test (full-fit) feature outputs")
    q.set_defaults(func=cmd_fit_probe)

    q = sub.add_parser("infer-expanded-features")
    q.add_argument("--embeddings", required=True); q.add_argument("--probe", required=True); q.add_argument("--output", required=True)
    q.add_argument("--device", default="cuda")
    q.add_argument("--batch-rows", type=int, default=8192); q.set_defaults(func=cmd_infer_features)

    q = sub.add_parser("prepare-rna-cache")
    q.add_argument("--canonical-root", required=True); q.add_argument("--output", required=True); q.set_defaults(func=cmd_rna_cache)

    q = sub.add_parser("build-compact-cache")
    q.add_argument("--canonical-root", required=True); q.add_argument("--protocol", default="tcga_mix_chr123"); q.add_argument("--output", required=True)
    q.add_argument("--row-chunk", type=int, default=16); q.set_defaults(func=cmd_compact_cache)

    q = sub.add_parser("write-chr1-eval-adapter")
    q.add_argument("--canonical-root", required=True); q.add_argument("--base-config", default="configs/train.yaml")
    q.add_argument("--embeddings", required=True); q.add_argument("--features", required=True); q.add_argument("--output", required=True)
    q.set_defaults(func=cmd_write_eval_adapter)

    q = sub.add_parser("train")
    q.add_argument("--canonical-root", required=True); q.add_argument("--protocol", required=True, choices=("tcga_mix_chr1", "tcga_mix_chr123", "array_genomewide"))
    q.add_argument("--base-config", default="configs/train.yaml"); q.add_argument("--base-cache", required=True)
    q.add_argument("--expanded-embeddings"); q.add_argument("--expanded-features")
    q.add_argument("--rna-cache", required=True); q.add_argument("--output", required=True)
    q.add_argument("--source-policy", choices=("equal_source", "array_heavy", "proportional_to_measurements"), default="equal_source")
    q.add_argument("--holdout-policy", choices=("mp_matched", "strict_global"), default="mp_matched")
    q.add_argument("--genomewide-cpg-split"); q.add_argument("--array-cache"); q.add_argument("--epic-cache")
    q.add_argument("--seed", type=int, default=17); q.add_argument("--steps-per-epoch", type=int, default=128); q.add_argument("--dev-fraction", type=float, default=0.10)
    q.add_argument("--array-rows", type=int, default=128); q.add_argument("--array-cpgs", type=int, default=2048)
    q.add_argument("--epic-rows", type=int, default=128); q.add_argument("--epic-cpgs", type=int, default=4096)
    q.add_argument("--wgbs-rows", type=int, default=32); q.add_argument("--wgbs-cpgs", type=int, default=16384)
    q.set_defaults(func=cmd_train)

    args = p.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
