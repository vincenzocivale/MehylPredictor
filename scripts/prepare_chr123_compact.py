#!/usr/bin/env python3
"""Build protocol-ordered chr123 methylation caches for dense block slicing."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from methylation_predictor.scopes import scope_protocol  # noqa: E402
from methylation_predictor.tcga_canonical import TCGACanonicalBundle  # noqa: E402
from scripts.benchmark_methylprophet.prepare import TABLE5_PROTOCOL_NAME, build_exact_compact_cache  # noqa: E402


def digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, np.int64).tobytes()).hexdigest()


def file_digest(path: Path, chunk_bytes: int = 8 * 1024**2) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_wgbs_compact(root: Path, output: Path, cpg_ids: np.ndarray) -> dict[str, object]:
    """Compact WGBS columns while retaining all physical rows, including duplicate patients."""
    if output.is_file():
        with h5py.File(output, "r") as h:
            return {"status": "cached", "shape": list(h["beta"].shape)}
    with TCGACanonicalBundle.from_root(root) as bundle:
        source = bundle.sources["wgbs"]
        cols = source.cpg_positions(cpg_ids)
        sample_ids = np.asarray(source.sample_idx, np.int64)
        tmp = output.with_suffix(".tmp.h5")
        with h5py.File(tmp, "w") as h:
            ds = h.create_dataset("beta", shape=(source.n_rows, len(cpg_ids)), dtype="f4", chunks=(source.n_rows, 8192))
            for start in range(0, len(cpg_ids), 8192):
                stop = min(start + 8192, len(cpg_ids)); local = cols[start:stop]
                order = np.argsort(local); inverse = np.empty_like(order); inverse[order] = np.arange(len(order))
                ds[:, start:stop] = np.asarray(source.h5["beta"][:, local[order]], np.float32)[:, inverse]
                if start == 0 or stop == len(cpg_ids) or (start // 8192) % 100 == 0:
                    print(f"[wgbs-cache] CpGs {stop:,}/{len(cpg_ids):,}", flush=True)
            h.create_dataset("sample_idx", data=sample_ids, dtype="i8")
            h.create_dataset("measurement_idx", data=np.asarray(source.measurement_idx, np.int64), dtype="i8")
            h.create_dataset("cpg_idx", data=np.asarray(cpg_ids, np.int64), dtype="i8")
            h.attrs["source"] = "wgbs"; h.attrs["protocol"] = TABLE5_PROTOCOL_NAME
        os.replace(tmp, output)
    return {"status": "built", "shape": [int(len(sample_ids)), int(len(cpg_ids))]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root, output = Path(args.canonical_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with TCGACanonicalBundle.from_root(root) as bundle:
        protocol = scope_protocol("chr123", bundle, canonical_root=root)
        axes = {
            "array": (
                np.concatenate([protocol.array_train_sample_idx, protocol.array_val_sample_idx]),
                np.concatenate([protocol.array_train_cpg_idx, protocol.array_val_cpg_idx]),
            ),
            "epic": (np.asarray(bundle.sources["epic"].sample_idx, np.int64), protocol.auxiliary_cpg_idx["epic"]),
            "wgbs": (np.asarray(bundle.sources["wgbs"].sample_idx, np.int64), protocol.auxiliary_cpg_idx["wgbs"]),
        }
    results = {}
    for name, (samples, cpgs) in axes.items():
        results[name] = (
            build_wgbs_compact(root, output / "wgbs.h5", cpgs)
            if name == "wgbs"
            else build_exact_compact_cache(
                root, name, output / f"{name}.h5", sample_ids=samples, cpg_ids=cpgs, row_chunk=16,
            )
        )
        with h5py.File(output / f"{name}.h5", "r") as h:
            if not np.array_equal(h["sample_idx"][...], samples):
                raise AssertionError(f"{name} sample axis mismatch")
            if not np.array_equal(h["cpg_idx"][...], cpgs):
                raise AssertionError(f"{name} CpG axis mismatch")
    manifest = {
        "schema_version": 1, "scope": "chr123", "canonical_root": str(root.resolve()),
        "sources": {
            name: {
                **results[name],
                "bytes": (output / f"{name}.h5").stat().st_size,
                "file_sha256": file_digest(output / f"{name}.h5"),
                "samples_sha256": digest(samples),
                "cpgs_sha256": digest(cpgs),
            }
            for name, (samples, cpgs) in axes.items()
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
