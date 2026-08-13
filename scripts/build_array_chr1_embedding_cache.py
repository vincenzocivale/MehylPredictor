#!/usr/bin/env python3
"""Build a tiny exact Array-chr1 NTv3 embedding cache from the consolidated atlas."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np


EXPECTED_TRAIN = 33_885
EXPECTED_VAL = 6_742
EXPECTED_TOTAL = 40_627
EXPECTED_DIM = 1536


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--canonical-root", required=True)
    p.add_argument("--atlas", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--chunk-rows", type=int, default=2048)
    args = p.parse_args()

    root = Path(args.canonical_root)
    protocol = root / "protocols" / "tcga_array_chr1"
    train = np.load(protocol / "array_train_cpg_idx.npy").astype(np.int64)
    val = np.load(protocol / "array_val_cpg_idx.npy").astype(np.int64)
    if len(train) != EXPECTED_TRAIN or len(val) != EXPECTED_VAL:
        raise RuntimeError(f"unexpected Array-chr1 split: train={len(train)} val={len(val)}")
    required = np.unique(np.concatenate([train, val])).astype(np.int64)
    if len(required) != EXPECTED_TOTAL:
        raise RuntimeError(f"expected {EXPECTED_TOTAL} unique Array-chr1 CpGs, got {len(required)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        with h5py.File(output, "r") as h:
            ids = np.asarray(h["cpg_idx"][...], dtype=np.int64)
            shape = tuple(h["embeddings"].shape)
        if len(ids) != EXPECTED_TOTAL or shape != (EXPECTED_TOTAL, EXPECTED_DIM):
            raise RuntimeError(f"existing cache has unexpected contract: ids={len(ids)} shape={shape}")
        if not np.array_equal(np.sort(ids), required):
            raise RuntimeError("existing cache CpG ids do not match exact Array-chr1 protocol")
        print(f"cached: {output} shape={shape}")
        return

    atlas = Path(args.atlas)
    with h5py.File(atlas, "r") as h:
        if "cpg_idx" not in h or "embedding" not in h:
            raise RuntimeError(f"{atlas} must contain cpg_idx and embedding")
        atlas_ids = np.asarray(h["cpg_idx"][...], dtype=np.int64)
        emb = h["embedding"]
        if emb.ndim != 2 or emb.shape[1] != EXPECTED_DIM:
            raise RuntimeError(f"unexpected atlas embedding shape: {emb.shape}")

        order = np.argsort(atlas_ids, kind="mergesort")
        sorted_ids = atlas_ids[order]
        pos = np.searchsorted(sorted_ids, required)
        ok = pos < len(sorted_ids)
        if ok.any():
            ii = np.flatnonzero(ok)
            ok[ii] &= sorted_ids[pos[ii]] == required[ii]
        if not np.all(ok):
            missing = required[~ok]
            raise RuntimeError(f"atlas missing {len(missing)} Array-chr1 CpGs; examples={missing[:10].tolist()}")

        # HDF5 fancy indexing is most efficient/reliable in increasing physical-row order.
        rows = np.sort(order[pos]).astype(np.int64)
        selected_ids = atlas_ids[rows]
        if not np.array_equal(np.sort(selected_ids), required):
            raise RuntimeError("internal Array-chr1 atlas selection mismatch")

        tmp = output.with_suffix(output.suffix + ".tmp")
        with h5py.File(tmp, "w") as out:
            out.create_dataset("cpg_idx", data=selected_ids, dtype="i8")
            ds = out.create_dataset(
                "embeddings",
                shape=(len(rows), EXPECTED_DIM),
                dtype="f2",
                chunks=(min(args.chunk_rows, len(rows)), EXPECTED_DIM),
            )
            for start in range(0, len(rows), args.chunk_rows):
                stop = min(start + args.chunk_rows, len(rows))
                ds[start:stop] = np.asarray(emb[rows[start:stop], :], dtype=np.float16)
            out.attrs["source_atlas"] = str(atlas)
            out.attrs["protocol"] = "tcga_array_chr1"
            out.attrs["rows"] = EXPECTED_TOTAL
            out.attrs["embedding_dim"] = EXPECTED_DIM
            out.attrs["storage_dtype"] = "float16"
        os.replace(tmp, output)

    print(f"built: {output} shape=({EXPECTED_TOTAL}, {EXPECTED_DIM})")


if __name__ == "__main__":
    main()
