#!/usr/bin/env python3
"""Run CpGPT/MethylGPT on the official validation-CpG x training-sample view.

The released models use Illumina probe ids, while this repository's canonical
stores use genomic coordinates.  This runner builds a small, reproducible
probe-ordered parquet from the canonical HDF5 source, then delegates model
loading/inference to the already tested ``methylation-fm-benchmark`` adapters.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = Path("/data2/home/vcivale/projects/methylation/methylation-fm-benchmark")


def _norm_chrom(value: object) -> str:
    s = str(value)
    return s[3:] if s.lower().startswith("chr") else s


def _load_model_vocab(benchmark_root: Path) -> list[str]:
    vocab_path = benchmark_root / "models/MethylGPT/tutorials/get_embeddings/pretrained_models/medium/vocab.json"
    vocab = json.loads(vocab_path.read_text())
    return [token for token, _idx in sorted(vocab.items(), key=lambda item: item[1]) if not token.startswith("<")]


def _load_model_coordinates(benchmark_root: Path) -> pd.DataFrame:
    path = benchmark_root / "data/cpgpt/cpg_manifest.tsv"
    frame = pd.read_csv(path, sep="\t", usecols=["probe_id", "chrom", "position"])
    frame["chrom"] = frame["chrom"].map(_norm_chrom)
    frame["position"] = frame["position"].astype(np.int64)
    return frame.drop_duplicates("probe_id").set_index("probe_id")


def _scope_paths(scope: str) -> tuple[Path, Path, Path, str, str]:
    backup = Path("/data2/home/vcivale/projects/methylation/backup/MethylPredictionData")
    if scope == "encode":
        root = REPO / "local_methyl_data/datasets/methylprophet_encode_v1"
        return (
            root / "methylation/encode_wgbs_full.h5",
            root / "cpg/registries/encode_wgbs_cpg_map.parquet",
            root / "cpg/splits",
            "X",
            "encode",
        )
    root = backup / "datasets/methylprophet_repro_v1"
    protocol = root / "protocols" / f"tcga_mix_{scope}"
    return (
        root / "methylation/tcga_array_official_full.h5",
        root / "cpg/registries/array_cpg_map.parquet",
        protocol,
        "beta",
        "tcga",
    )


def _prepare(scope: str, out_dir: Path, benchmark_root: Path) -> tuple[Path, Path, Path]:
    h5_path, registry_path, split_dir, dataset_key, _ = _scope_paths(scope)
    out_dir.mkdir(parents=True, exist_ok=True)
    beta_path = out_dir / f"{scope}_model_input.parquet"
    mask_path = out_dir / f"{scope}_model_input_observed_mask.parquet"
    targets_path = out_dir / f"{scope}_validation_cpgs.txt"
    if beta_path.exists() and mask_path.exists() and targets_path.exists():
        return beta_path, mask_path, targets_path
    import h5py

    train_samples = np.load(split_dir / ("train_sample_idx.npy" if scope == "encode" else "array_train_sample_idx.npy"))
    train_cpgs = np.load(split_dir / ("train_cpg_idx.npy" if scope == "encode" else "array_train_cpg_idx.npy"))
    val_cpgs = np.load(split_dir / ("val_cpg_idx.npy" if scope == "encode" else "array_val_cpg_idx.npy"))
    registry = pd.read_parquet(registry_path, columns=["cpg_idx", "chr", "pos"])
    model_ids = _load_model_vocab(benchmark_root)
    coords = _load_model_coordinates(benchmark_root)
    # Do not materialise the full ENCODE train+validation index (27M rows) or
    # iterate it in Python. A vectorised coordinate join isolates the ~49k
    # released-model probes directly.
    model_coords = coords.reset_index().rename(columns={"chrom": "chrom_norm", "position": "pos0"})
    model_coords["pos0"] = model_coords["pos0"].astype(np.int64)
    registry["chrom_norm"] = registry["chr"].map(_norm_chrom)
    registry["pos0"] = registry["pos"].astype(np.int64)
    registry["pos0"] = registry["pos0"] - 1
    overlap = registry.merge(model_coords[["probe_id", "chrom_norm", "pos0"]], on=["chrom_norm", "pos0"], how="inner")
    source_coord_to_idx = dict(zip(overlap["probe_id"], overlap["cpg_idx"].astype(np.int64)))
    # The canonical registry stores 1-based positions; released FM manifests use 0-based.
    selected_ids = [p for p in model_ids if p in source_coord_to_idx]
    if not selected_ids:
        raise RuntimeError(f"No model probes overlap {scope}; coordinate convention or registry is wrong")
    source_idx_by_probe = {
        p: int(source_coord_to_idx[p])
        for p in selected_ids
    }
    val_set = set(val_cpgs.tolist())
    val_ids = [p for p in selected_ids if source_idx_by_probe[p] in val_set]
    if not val_ids:
        raise RuntimeError(f"No validation CpGs overlap model vocabulary for {scope}")

    sample_set = set(train_samples.tolist())
    with h5py.File(h5_path, "r") as h5:
        sample_axis = np.asarray(h5["sample_idx"])
        row_positions = [int(np.flatnonzero(sample_axis == s)[0]) for s in train_samples]
        cpg_axis = np.asarray(h5["cpg_idx"]) if "cpg_idx" in h5 else np.arange(h5[dataset_key].shape[1])
        col_by_idx = {int(v): i for i, v in enumerate(cpg_axis)}
        columns = [col_by_idx[source_idx_by_probe[p]] for p in selected_ids]
        row_order = np.argsort(row_positions)
        sorted_rows = np.asarray(row_positions, dtype=np.int64)[row_order]
        col_order = np.argsort(columns)
        sorted_cols = np.asarray(columns, dtype=np.int64)[col_order]
        # h5py cannot fancy-index two axes and ``dset[rows, :][:, cols]`` creates
        # a full (patients x all-CpGs) temporary for ENCODE. Read chromosome-
        # independent column blocks instead, keeping memory proportional to the
        # model vocabulary rather than to the 27M-CpG source.
        matrix_sorted = np.empty((len(sorted_rows), len(sorted_cols)), dtype=np.float32)
        for block_start in range(0, len(sorted_cols), 8192):
            block_end = min(block_start + 8192, len(sorted_cols))
            block_cols = sorted_cols[block_start:block_end]
            lo, hi = int(block_cols[0]), int(block_cols[-1]) + 1
            # Read in row slabs: this avoids both one-I/O-per-patient fancy
            # indexing and retaining a 300MB temporary for every column block.
            for row_start in range(0, len(sorted_rows), 256):
                row_end = min(row_start + 256, len(sorted_rows))
                rows = sorted_rows[row_start:row_end]
                block = np.asarray(h5[dataset_key][rows, lo:hi], dtype=np.float32)
                matrix_sorted[row_start:row_end, block_start:block_end] = block[:, block_cols - lo]
        matrix = matrix_sorted[np.argsort(row_order)][:, np.argsort(col_order)]
        names = np.asarray(h5["sample_name"])[row_positions].astype(str).tolist()
    # Keep the complete released vocabulary. Probes absent from this chromosome/scope
    # are filled with the training mean and marked unobserved; they are never targets.
    full = np.zeros((len(names), len(model_ids)), dtype=np.float32)
    observed = np.zeros_like(full, dtype=bool)
    full[:] = 0.5
    selected_pos = {p: i for i, p in enumerate(selected_ids)}
    for j, probe in enumerate(model_ids):
        if probe in selected_pos:
            full[:, j] = matrix[:, selected_pos[probe]]
            observed[:, j] = True
    # Replace occasional missing WGBS values with the train mean, retaining observed mask.
    bad = ~np.isfinite(full)
    full[bad] = 0.5
    pd.DataFrame({"id": names, "data": full.tolist()}).to_parquet(beta_path, index=False)
    pd.DataFrame({"id": names, "data": observed.tolist()}).to_parquet(mask_path, index=False)
    targets_path.write_text("\n".join(val_ids) + "\n")
    (out_dir / f"{scope}_preparation.json").write_text(json.dumps({
        "scope": scope, "n_training_samples": len(names), "n_model_vocab": len(model_ids),
        "n_overlapping_probes": len(selected_ids), "n_validation_targets": len(val_ids),
        "source": str(h5_path), "checkpoint_policy": "released pretrained checkpoint",
    }, indent=2))
    return beta_path, mask_path, targets_path


def _run_model(model: str, scope: str, paths: tuple[Path, Path, Path], out_dir: Path, benchmark_root: Path) -> None:
    beta_path, mask_path, targets_path = paths
    sys.path.insert(0, str(benchmark_root / "src"))
    # The adapters import the vendored model package only at inference time.
    # Keep it outside this repository's normal dependency set.
    sys.path.insert(0, str(benchmark_root / "models/MethylGPT"))
    sys.path.insert(0, str(benchmark_root / "models/CpGPT"))
    if model == "cpgpt":
        # CpGPT's generic ``common-target`` mode uses the target list as the
        # whole input universe.  That is appropriate for target-only scoring,
        # but wrong here: validation CpGs must be masked while training-observed
        # CpGs remain available as context.  Keep the target file as the mask
        # list and build the universe from all beta/cache-overlapping probes.
        import benchmark.inference.cpgpt as cpgpt_inference

        def _training_context_probe_set(**kwargs):
            beta_ids = [str(p) for p in kwargs["beta_probe_ids"]]
            cache = set(map(str, kwargs["cache_probe_ids"]))
            universe = [p for p in beta_ids if p in cache]
            if not universe:
                raise ValueError("No CpGs remain after intersecting beta matrix and embedding cache")
            n_before = len(universe)
            max_seq_len = int(kwargs["max_seq_len"])
            capped = n_before > max_seq_len
            if capped:
                rng = np.random.default_rng(int(kwargs.get("cap_seed") or 0))
                chosen = np.sort(rng.choice(n_before, size=max_seq_len, replace=False))
                universe = [universe[i] for i in chosen]
            target_ids = cpgpt_inference._load_text_list(kwargs["target_cpgs_file"])
            target_ids = [p for p in target_ids if p in set(universe)]
            masked = cpgpt_inference._select_masked_cpgs(
                target_ids, float(kwargs["mask_fraction"]), int(kwargs["mask_seed"])
            )
            return cpgpt_inference.EffectiveProbeSet(
                probe_ids=universe, masked_cpgs=masked, capped=capped,
                n_before_cap=n_before, n_after_cap=len(universe),
                mode="training-context-val-targets",
            )

        cpgpt_inference._build_effective_probe_set = _training_context_probe_set
        predict = cpgpt_inference.predict
        cfg = {
            # The local benchmark has a verified Small masked-recovery pair.
            # The large artifact uses a different input contract and is not
            # interchangeable with this adapter's forward path.
            "model": {"loader": "benchmark.inference.cpgpt_loader.load", "checkpoint": str(benchmark_root / "models/CpGPT/model/weights/small.ckpt"), "config": str(benchmark_root / "models/CpGPT/model/config/small.yaml"), "device": "cuda"},
            "data": {"beta_parquet": str(beta_path), "observed_mask": str(mask_path), "dna_embeddings_cache": str(benchmark_root / "data/cpgpt/dna_embeddings_cache.npz"), "probe_ids_file": str(benchmark_root / "data/methylgpt/probe_order_type3.txt")},
            "eval": {"target_cpgs_file": str(targets_path), "mask_fraction": 1.0, "mask_seed": 0, "cap_seed": 0, "batch_size": 8},
            "output": {"metadata_path": str(out_dir / f"{scope}_{model}_metadata.json")},
        }
    else:
        from benchmark.inference.methylgpt import predict
        model_dir = benchmark_root / "models/MethylGPT/tutorials/get_embeddings/pretrained_models/medium"
        cfg = {
            "input_parquet": str(beta_path), "probe_ids_path": str(benchmark_root / "models/MethylGPT/tutorials/pretraining/probe_ids_type3.csv"),
            "checkpoint": str(model_dir / "medium-best_model_epoch6.pt"), "args_json": str(model_dir / "args.json"),
            # The available released checkpoint in the local benchmark cache is
            # the 256-dim medium/large-labelled artifact; its args.json calls
            # this field layer_size, so pass the architecture explicitly to the
            # generic adapter.
            "model": {"d_model": 256, "nhead": 4, "nlayers": 6, "n_special": 3, "cls_gene_id": 1},
            "device": "cuda", "max_seq_len": 4096, "batch_size": 8,
            "observed_mask_path": str(mask_path), "eval": {"target_cpgs_file": str(targets_path), "mask_fraction": 1.0, "mask_seed": 0},
            "output_path": str(out_dir / f"{scope}_{model}_predictions.parquet"),
        }
    pred = predict(cfg)
    if model == "cpgpt":
        pred.to_parquet(out_dir / f"{scope}_{model}_predictions.parquet", index=False)
    from benchmark.cli import build_report_long
    report = build_report_long(pred, include_positive_controls=False)
    report.update({"scope": scope, "model": model, "view": "val_cpg_x_train_sample", "checkpoint_policy": "released pretrained checkpoint"})
    (out_dir / f"{scope}_{model}_report.json").write_text(json.dumps(report, indent=2, default=str))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scope", choices=["chr1", "chr123", "encode"], required=True)
    p.add_argument("--model", choices=["cpgpt", "methylgpt", "both"], default="both")
    p.add_argument("--output-root", default=str(REPO / "local_methyl_data/runs/foundation_models"))
    p.add_argument("--benchmark-root", default=str(DEFAULT_BENCHMARK))
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    out = Path(args.output_root) / args.scope
    paths = _prepare(args.scope, out / "input", Path(args.benchmark_root))
    if args.prepare_only:
        return
    models = ["cpgpt", "methylgpt"] if args.model == "both" else [args.model]
    for model in models:
        _run_model(model, args.scope, paths, out, Path(args.benchmark_root))


if __name__ == "__main__":
    main()
