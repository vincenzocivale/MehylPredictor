#!/usr/bin/env python3
"""Strictly align and merge frozen RNA encoder embeddings into one HDF5 file.

The quality-audit configuration accepts one embedding HDF5 with multiple named
representations.  This utility combines independently extracted BulkRNABert and
BulkFormer files without silently dropping or reordering samples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in np.asarray(values).tolist()
    ], dtype=object)


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("sources"), dict):
        raise ValueError("merge config must contain a 'sources' mapping")
    return payload


def _unique_index(values: np.ndarray, label: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, value in enumerate(values):
        key = str(value)
        if key in result:
            raise ValueError(f"duplicate {label}: {key}")
        result[key] = index
    return result


def _dataset_spec(raw: Any) -> tuple[list[str], str]:
    """Return input keys and transform for one merged representation.

    Backwards-compatible forms:
      output_name: input_dataset
      output_name: [dataset_a, dataset_b]
      output_name: {keys: [dataset_a, dataset_b], transform: concat}
    """
    if isinstance(raw, str):
        return [raw], "identity"
    if isinstance(raw, list):
        return [str(value) for value in raw], "concat"
    if isinstance(raw, dict):
        keys = raw.get("keys", raw.get("key"))
        if isinstance(keys, str):
            keys = [keys]
        if not isinstance(keys, list) or not keys:
            raise ValueError("dataset mapping requires non-empty 'key' or 'keys'")
        transform = str(raw.get("transform", "identity" if len(keys) == 1 else "concat"))
        if transform not in {"identity", "concat"}:
            raise ValueError(f"unsupported dataset transform {transform!r}")
        if transform == "identity" and len(keys) != 1:
            raise ValueError("identity transform requires exactly one input key")
        return [str(value) for value in keys], transform
    raise TypeError(f"invalid dataset specification: {raw!r}")


def _read_representation(
    handle: h5py.File,
    raw_spec: Any,
    rows: np.ndarray,
    expected_rows: int,
    label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    keys, transform = _dataset_spec(raw_spec)
    parts: list[np.ndarray] = []
    for key in keys:
        if key not in handle:
            raise KeyError(f"{handle.filename} lacks embedding dataset {key!r}")
        values = np.asarray(handle[key][...], dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != expected_rows:
            raise ValueError(f"invalid embedding shape for {label}/{key}: {values.shape}")
        parts.append(values[rows])
    values = parts[0] if transform == "identity" else np.concatenate(parts, axis=1)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite values in {label}")
    return values, {"input_keys": keys, "transform": transform, "shape": list(values.shape)}


def merge(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    config = _load_config(config_path)
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = config["sources"]

    canonical_ids: np.ndarray | None = None
    arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, Any] = {"config": str(config_path), "sources": {}}

    for source_name, raw_spec in sources.items():
        spec = dict(raw_spec)
        path = Path(spec["path"])
        row_key = str(spec.get("row_ids_key", "sample_idx"))
        datasets = dict(spec.get("datasets", {}))
        if not datasets:
            raise ValueError(f"source {source_name!r} has no datasets")
        with h5py.File(path, "r") as handle:
            if row_key not in handle:
                raise KeyError(f"{path} lacks sample ID dataset {row_key!r}")
            ids = _decode(handle[row_key][...])
            index = _unique_index(ids, f"{source_name} sample ID")
            if canonical_ids is None:
                canonical_ids = ids
                rows = np.arange(len(ids), dtype=np.int64)
            else:
                missing = [str(value) for value in canonical_ids if str(value) not in index]
                extra = [str(value) for value in ids if str(value) not in set(canonical_ids.astype(str))]
                if missing or extra:
                    raise ValueError(
                        f"sample sets differ for {source_name!r}: "
                        f"missing={missing[:5]} extra={extra[:5]}"
                    )
                rows = np.asarray([index[str(value)] for value in canonical_ids], dtype=np.int64)
            source_manifest = {"path": str(path), "row_ids_key": row_key, "datasets": {}}
            for output_name, dataset_spec in datasets.items():
                values, dataset_manifest = _read_representation(
                    handle,
                    dataset_spec,
                    rows,
                    len(ids),
                    f"{source_name}/{output_name}",
                )
                merged_name = f"{source_name}__{output_name}"
                if merged_name in arrays:
                    raise ValueError(f"duplicate merged representation {merged_name!r}")
                arrays[merged_name] = values
                source_manifest["datasets"][output_name] = {
                    **dataset_manifest,
                    "output_key": merged_name,
                }
            manifest["sources"][source_name] = source_manifest

    if canonical_ids is None:
        raise ValueError("no sources were configured")
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(output, "w") as handle:
        handle.create_dataset("sample_idx", data=np.asarray(canonical_ids.astype(str), dtype=object), dtype=string_dtype)
        for name, values in arrays.items():
            handle.create_dataset(name, data=values, compression="gzip", shuffle=True)
        handle.attrs["transcriptome_only"] = True
        handle.attrs["manifest_json"] = json.dumps(manifest, sort_keys=True)
    manifest["output"] = str(output)
    manifest["n_samples"] = int(len(canonical_ids))
    manifest["representations"] = {name: list(values.shape) for name, values in arrays.items()}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(merge(args.config))


if __name__ == "__main__":
    main()
