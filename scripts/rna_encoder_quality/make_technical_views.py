#!/usr/bin/env python3
"""Create deterministic technical perturbation views of a log2(TPM+1) RNA matrix."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml

from methylation_predictor.rna_encoder_quality.config import load_config
from methylation_predictor.rna_encoder_quality.io import decode_ids


def _write_view(
    source: h5py.Dataset,
    output: Path,
    values_key: str,
    row_key: str,
    col_key: str,
    row_ids: np.ndarray,
    col_ids: np.ndarray,
    seed: int,
    mode: str,
    magnitude: float,
    chunk_size: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    strings = h5py.string_dtype("utf-8")
    with h5py.File(output, "w") as handle:
        matrix = handle.create_dataset(values_key, shape=source.shape, dtype="f4", chunks=(min(chunk_size, source.shape[0]), source.shape[1]))
        handle.create_dataset(row_key, data=np.asarray(row_ids, dtype=object), dtype=strings)
        handle.create_dataset(col_key, data=np.asarray(col_ids, dtype=object), dtype=strings)
        for start in range(0, source.shape[0], chunk_size):
            stop = min(start + chunk_size, source.shape[0])
            log_values = np.asarray(source[start:stop], dtype=np.float64)
            tpm = np.maximum(np.exp2(np.clip(log_values, 0.0, 30.0)) - 1.0, 0.0)
            result = np.empty_like(tpm, dtype=np.float32)
            for local in range(len(tpm)):
                rng = np.random.default_rng(seed + start + local)
                vector = tpm[local]
                total = vector.sum()
                if total <= 0:
                    result[local] = 0.0
                    continue
                if mode == "multinomial":
                    reads = max(1, int(round(total * magnitude)))
                    perturbed = rng.multinomial(reads, vector / total).astype(np.float64)
                elif mode == "dropout":
                    keep = rng.random(len(vector)) >= magnitude
                    perturbed = vector * keep
                else:
                    raise ValueError(mode)
                perturbed_total = perturbed.sum()
                if perturbed_total > 0:
                    perturbed *= 1_000_000.0 / perturbed_total
                result[local] = np.log2(perturbed + 1.0).astype(np.float32)
            matrix[start:stop] = result


def _write_extractor_config(base: Path, output: Path, rna_path: Path) -> None:
    payload = yaml.safe_load(base.read_text())
    payload["data"]["rna"]["path"] = str(rna_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-config", required=True)
    parser.add_argument("--extractor-base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--library-fraction", type=float, default=0.5)
    parser.add_argument("--dropout-rate", type=float, default=0.05)
    parser.add_argument("--chunk-size", type=int, default=16)
    args = parser.parse_args()
    if not 0 < args.library_fraction <= 1:
        raise ValueError("--library-fraction must be in (0, 1]")
    if not 0 <= args.dropout_rate < 1:
        raise ValueError("--dropout-rate must be in [0, 1)")
    config = load_config(args.quality_config)
    output = Path(args.output_dir)
    with h5py.File(config.rna.path, "r") as handle:
        source = handle[config.rna.values_key]
        row_ids = decode_ids(handle[config.rna.row_ids_key][...])
        col_ids = decode_ids(handle[config.rna.col_ids_key][...])
        views = (
            ("multinomial_50pct", "multinomial", args.library_fraction),
            ("gene_dropout_5pct", "dropout", args.dropout_rate),
        )
        for name, mode, magnitude in views:
            h5_path = output / f"{name}.h5"
            _write_view(
                source, h5_path, config.rna.values_key, config.rna.row_ids_key,
                config.rna.col_ids_key, row_ids, col_ids, args.seed, mode, magnitude, args.chunk_size,
            )
            _write_extractor_config(
                Path(args.extractor_base_config), output / f"extract_{name}.yaml", h5_path
            )
            print(f"wrote {h5_path} and extract_{name}.yaml")


if __name__ == "__main__":
    main()
