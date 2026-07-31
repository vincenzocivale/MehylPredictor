#!/usr/bin/env python3
"""Create biological module perturbations and expression-matched random controls."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from methylation_predictor.rna_encoder_quality.config import load_config
from methylation_predictor.rna_encoder_quality.io import decode_ids, read_table, unique_index


def _aliases(value: object) -> set[str]:
    fields = {field.strip() for field in str(value).split(";") if field.strip()}
    fields.add(str(value).strip())
    result = set(fields)
    for field in fields:
        if field.upper().startswith("ENSG"):
            result.add(field.split(".", 1)[0])
    return {value for value in result if value}


def _unique_alias_map(gene_ids: np.ndarray) -> dict[str, int]:
    occurrences: dict[str, list[int]] = {}
    for index, gene in enumerate(gene_ids):
        for alias in _aliases(gene):
            occurrences.setdefault(alias, []).append(index)
    return {alias: rows[0] for alias, rows in occurrences.items() if len(set(rows)) == 1}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower()


def _train_mean_tpm(matrix: h5py.Dataset, train_rows: np.ndarray, chunk_size: int) -> np.ndarray:
    total = np.zeros(matrix.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, len(train_rows), chunk_size):
        rows = np.sort(train_rows[start:start + chunk_size])
        values = np.asarray(matrix[rows], dtype=np.float64)
        total += np.maximum(np.exp2(np.clip(values, 0.0, 30.0)) - 1.0, 0.0).sum(axis=0)
        count += len(values)
    return total / max(count, 1)


def _matched_random(module: np.ndarray, mean_tpm: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    quantiles = np.quantile(mean_tpm, np.linspace(0, 1, 11))
    bins = np.clip(np.digitize(mean_tpm, quantiles[1:-1]), 0, 9)
    module_set = set(module.tolist())
    available = set(range(len(mean_tpm))) - module_set
    chosen: list[int] = []
    for index in module:
        candidates = np.asarray(sorted(value for value in available if bins[value] == bins[index]), dtype=int)
        if not len(candidates):
            candidates = np.asarray(sorted(available), dtype=int)
        if not len(candidates):
            raise RuntimeError("not enough genes for a matched random control")
        pick = int(rng.choice(candidates))
        chosen.append(pick)
        available.remove(pick)
    return np.asarray(chosen, dtype=np.int64)


def _write_perturbed(
    source: h5py.Dataset,
    output: Path,
    values_key: str,
    row_key: str,
    col_key: str,
    row_ids: np.ndarray,
    col_ids: np.ndarray,
    gene_indices: np.ndarray,
    fold_change: float,
    chunk_size: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    strings = h5py.string_dtype("utf-8")
    with h5py.File(output, "w") as handle:
        matrix = handle.create_dataset(
            values_key,
            shape=source.shape,
            dtype="f4",
            chunks=(min(chunk_size, source.shape[0]), source.shape[1]),
        )
        handle.create_dataset(row_key, data=np.asarray(row_ids, dtype=object), dtype=strings)
        handle.create_dataset(col_key, data=np.asarray(col_ids, dtype=object), dtype=strings)
        for start in range(0, source.shape[0], chunk_size):
            stop = min(start + chunk_size, source.shape[0])
            values = np.asarray(source[start:stop], dtype=np.float64)
            tpm = np.maximum(np.exp2(np.clip(values, 0.0, 30.0)) - 1.0, 0.0)
            tpm[:, gene_indices] *= fold_change
            totals = tpm.sum(axis=1, keepdims=True)
            tpm = np.divide(tpm * 1_000_000.0, totals, out=np.zeros_like(tpm), where=totals > 0)
            matrix[start:stop] = np.log2(tpm + 1.0).astype(np.float32)


def _write_extractor_config(base: Path, output: Path, rna_path: Path) -> None:
    payload = yaml.safe_load(base.read_text())
    payload["data"]["rna"]["path"] = str(rna_path)
    output.write_text(yaml.safe_dump(payload, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-config", required=True)
    parser.add_argument("--extractor-base-config", required=True)
    parser.add_argument("--membership", required=True, help="TSV with module_id and gene_id")
    parser.add_argument("--modules", required=True, help="comma-separated module IDs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold-change", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--chunk-size", type=int, default=16)
    args = parser.parse_args()
    if args.fold_change <= 0 or np.isclose(args.fold_change, 1.0):
        raise ValueError("--fold-change must be positive and different from 1")
    config = load_config(args.quality_config)
    membership = pd.read_csv(args.membership, sep="\t")
    if not {"module_id", "gene_id"}.issubset(membership.columns):
        raise ValueError("membership requires module_id and gene_id columns")
    requested = [value.strip() for value in args.modules.split(",") if value.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_table(config.metadata.path)
    metadata_lookup = unique_index(metadata[config.metadata.id_column].astype(str), "metadata sample ID")
    with h5py.File(config.rna.path, "r") as handle:
        source = handle[config.rna.values_key]
        row_ids = decode_ids(handle[config.rna.row_ids_key][...])
        col_ids = decode_ids(handle[config.rna.col_ids_key][...])
        aliases = _unique_alias_map(col_ids)
        train_values = set(config.metadata.train_values)
        train_rows = []
        for row_index, sample_id in enumerate(row_ids):
            if str(sample_id) not in metadata_lookup:
                raise ValueError(f"RNA sample {sample_id!r} is absent from metadata")
            metadata_row = metadata.iloc[metadata_lookup[str(sample_id)]]
            if str(metadata_row[config.metadata.split_column]) in train_values:
                train_rows.append(row_index)
        train_rows_array = np.asarray(train_rows, dtype=np.int64)
        if not len(train_rows_array):
            raise ValueError("no train samples available for expression matching")
        mean_tpm = _train_mean_tpm(source, train_rows_array, args.chunk_size)
        report: dict[str, object] = {
            "fold_change": args.fold_change,
            "seed": args.seed,
            "modules": {},
        }
        for module_offset, module_name in enumerate(requested):
            genes = membership.loc[membership["module_id"].astype(str) == module_name, "gene_id"].astype(str)
            mapped = sorted({
                aliases[alias]
                for gene in genes
                for alias in _aliases(gene)
                if alias in aliases
            })
            if len(mapped) < 2:
                raise ValueError(f"module {module_name!r} maps to fewer than two unique RNA genes")
            module_indices = np.asarray(mapped, dtype=np.int64)
            random_indices = _matched_random(module_indices, mean_tpm, args.seed + module_offset)
            safe = _safe_name(module_name)
            module_path = output_dir / f"{safe}.h5"
            control_path = output_dir / f"{safe}_random_matched.h5"
            _write_perturbed(
                source, module_path, config.rna.values_key, config.rna.row_ids_key,
                config.rna.col_ids_key, row_ids, col_ids, module_indices, args.fold_change, args.chunk_size,
            )
            _write_perturbed(
                source, control_path, config.rna.values_key, config.rna.row_ids_key,
                config.rna.col_ids_key, row_ids, col_ids, random_indices, args.fold_change, args.chunk_size,
            )
            _write_extractor_config(
                Path(args.extractor_base_config), output_dir / f"extract_{safe}.yaml", module_path
            )
            _write_extractor_config(
                Path(args.extractor_base_config), output_dir / f"extract_{safe}_random_matched.yaml", control_path
            )
            report["modules"][module_name] = {
                "requested_genes": int(len(genes)),
                "matched_genes": int(len(module_indices)),
                "module_gene_ids": col_ids[module_indices].astype(str).tolist(),
                "random_gene_ids": col_ids[random_indices].astype(str).tolist(),
                "module_h5": str(module_path),
                "control_h5": str(control_path),
            }
    (output_dir / "module_perturbation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
