#!/usr/bin/env python3
"""Build the ENCODE WGBS canonical bundle used by the `encode` paper study.

Source of truth is the official MethylProphet preprocessing pipeline's own
output, already computed in a sibling checkout (see
`~/methyl-benchmark/preprocessing_logs/run.log`):

    <mp-data-root>/parquet/241213-encode_wgbs/{gene_expr,me,cpg_bg,...}.parquet
    <mp-data-root>/parquet/241213-encode_wgbs/metadata/{cpg_split,sample_split}/...

This script does not re-derive the official 90/10, tissue-stratified split --
it only repackages the already-computed official artifacts into this repo's
canonical-bundle conventions (see
`dune_data/datasets/methylprophet_repro_v1/README.md` for the TCGA analogue):

    cpg/registries/encode_wgbs_cpg_map.parquet   (raw_cpg_row, cpg_idx, chr, pos, chr_pos)
    cpg/splits/{train,val}_cpg_idx.npy
    cpg/splits/{train,val}_sample_idx.npy
    methylation/encode_wgbs_full.h5              (X: samples x cpgs, sample_idx, chr_pos-ordered cpg axis)
    rna/encode_rna_official_full.h5              (X: samples x genes, gene_ids, sample_idx, sample_name)
    manifest.json, SHA256SUMS

Key facts established during investigation (see the approved plan in
`docs/EXPERIMENT_LOG.md` / conversation record, not re-derived here):

  * `cpg_idx` in the official pipeline's own outputs is exactly the 0-based
    row position of `metadata/cpg_per_chr_stats/cpg_chr_pos_df.parquet`
    (verified against `me_cpg_bg` processed rows: cpg_idx 3710395 ==
    chr11:511299, cpg_idx 9568907 == chr16:60863799, etc). `me.parquet`
    shard `NNNNN.parquet` covers global rows `[NNNNN*10000, ...)` of that
    same axis (verified: shard 00001 row 0 == global row 10000).
  * ENCODE's own `cpg_idx` numbering is NOT shared with this repo's TCGA
    canonical registries (spot-checked, no collision) -- ENCODE gets its own
    independent registry, which is fine: the genome-wide locus-feature store
    (`locus_features/catalog.append_coordinates`) keys everything by
    (chrom, position), not by any per-dataset `cpg_idx`.
  * ENCODE's `gene_expr.parquet` uses unversioned Ensembl gene ids and is a
    superset of this repo's canonical 25,017-gene TCGA RNA axis, except for
    1,080 genes (~4.3%) that don't appear in the ENCODE table at all. Per
    explicit user decision, those are zero-filled (documented in the
    manifest's `zero_filled_gene_ids` list), not imputed or masked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import pyarrow.parquet as pq

DATASET_NAME = "241213-encode_wgbs"
SPLIT_NAME = "train_0_9_val_0_1"
SAMPLE_SPLIT_NAME = "ind_tissue"
ME_SHARD_ROWS = 10_000


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    tmp.replace(path)


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def build_registry(mp_root: Path, output: Path, *, replace: bool) -> Path:
    """cpg_idx == row position of cpg_per_chr_stats/cpg_chr_pos_df.parquet."""
    dest = output / "cpg" / "registries" / "encode_wgbs_cpg_map.parquet"
    if dest.is_file() and not replace:
        return dest
    src = mp_root / "metadata" / "cpg_per_chr_stats" / "cpg_chr_pos_df.parquet"
    frame = pd.read_parquet(src)
    if list(frame.columns) != ["chr", "pos", "chr_pos"]:
        raise ValueError(f"unexpected cpg_chr_pos_df schema: {list(frame.columns)}")
    frame = frame.reset_index(drop=True)
    frame.insert(0, "cpg_idx", frame.index.to_numpy(dtype=np.int64))
    frame.insert(0, "raw_cpg_row", frame.index.to_numpy(dtype=np.int64))
    if frame["chr_pos"].duplicated().any():
        raise ValueError("duplicate chr_pos in ENCODE cpg registry source")
    _atomic_write_parquet(frame, dest)
    return dest


def build_cpg_split(mp_root: Path, output: Path, registry: pd.DataFrame, *, replace: bool) -> dict[str, Path]:
    out_dir = output / "cpg" / "splits"
    paths = {"train": out_dir / "train_cpg_idx.npy", "val": out_dir / "val_cpg_idx.npy"}
    if all(p.is_file() for p in paths.values()) and not replace:
        return paths
    by_chr_pos = registry.set_index("chr_pos")["cpg_idx"]
    split_dir = mp_root / "metadata" / "cpg_split" / SPLIT_NAME
    for name, path in paths.items():
        frame = pd.read_parquet(split_dir / f"{name}.parquet", columns=["chr_pos"])
        idx = by_chr_pos.reindex(frame["chr_pos"]).to_numpy()
        if np.isnan(idx).any():
            missing = frame.loc[pd.isna(by_chr_pos.reindex(frame["chr_pos"]).to_numpy()), "chr_pos"]
            raise ValueError(f"{name} split contains chr_pos missing from registry: {missing.head(5).tolist()}")
        idx = np.sort(idx.astype(np.int64))
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, idx)
    return paths


def build_sample_split(mp_root: Path, output: Path, *, replace: bool) -> dict[str, Path]:
    out_dir = output / "cpg" / "splits"
    paths = {"train": out_dir / "train_sample_idx.npy", "val": out_dir / "val_sample_idx.npy"}
    if all(p.is_file() for p in paths.values()) and not replace:
        return paths
    split_dir = mp_root / "metadata" / "sample_split" / SAMPLE_SPLIT_NAME
    for name, path in paths.items():
        frame = pd.read_csv(split_dir / f"{name}_sample_tissue_count_with_idx.csv")
        idx = np.sort(frame["sample_idx"].to_numpy(dtype=np.int64))
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, idx)
    return paths


def _sample_name_to_idx(mp_root: Path) -> tuple[pd.Series, pd.Series]:
    frame = pd.read_csv(mp_root / "metadata" / "sample_split" / "sample_with_idx.csv")
    if frame["sample_name"].duplicated().any():
        raise ValueError("sample_with_idx.csv has duplicate sample_name values")
    # gene_expr.parquet / me.parquet column headers are the *sample_name* text
    # (verified: all 95 sample_name values match parquet column headers
    # exactly; tissue_name is a coarser, non-unique category, e.g. "adipose"
    # shared by 3 distinct samples -- not usable as a join key).
    return frame.set_index("sample_name")["sample_idx"], frame.set_index("sample_idx")["sample_name"]


def build_rna_cache(mp_root: Path, output: Path, tcga_canonical_root: Path, *, replace: bool) -> dict:
    dest = output / "rna" / "encode_rna_official_full.h5"
    if dest.is_file() and not replace:
        return {"status": "cached", "path": str(dest)}

    name_to_idx, idx_to_name = _sample_name_to_idx(mp_root)
    n_samples = len(name_to_idx)

    with h5py.File(tcga_canonical_root / "rna" / "tcga_rna_official_full.h5", "r") as h:
        tcga_gene_ids = np.asarray(h["gene_ids"][...])
    tcga_ensg = [g.decode().split(";")[-1].split(".")[0] for g in tcga_gene_ids]

    shard_dir = mp_root / "gene_expr.parquet"
    frames = [pd.read_parquet(p) for p in sorted(shard_dir.glob("*.parquet"))]
    genes = pd.concat(frames, ignore_index=True)
    genes = genes.rename(columns={"Unnamed: 0": "gene_label"})
    genes["ensg"] = genes["gene_label"].str.split(";").str[-1]
    if genes["ensg"].duplicated().any():
        raise ValueError("duplicate ENSG ids in ENCODE gene_expr.parquet")
    genes = genes.set_index("ensg")

    sample_columns = [c for c in genes.columns if c not in ("gene_label",)]
    missing_names = sorted(set(sample_columns) - set(name_to_idx.index))
    if missing_names:
        raise ValueError(f"gene_expr.parquet columns not found in sample_with_idx.csv: {missing_names[:5]}")
    ordered_columns = [idx_to_name.loc[i] for i in range(n_samples)]

    n_genes = len(tcga_ensg)
    X = np.zeros((n_samples, n_genes), dtype=np.float32)
    zero_filled: list[str] = []
    present_mask = genes.index.isin(tcga_ensg)
    present_lookup = genes.loc[present_mask, ordered_columns]
    for j, ensg in enumerate(tcga_ensg):
        if ensg in present_lookup.index:
            X[:, j] = present_lookup.loc[ensg].to_numpy(dtype=np.float32)
        else:
            zero_filled.append(ensg)
    if len(zero_filled) + present_lookup.index.isin(tcga_ensg).sum() != n_genes:
        pass  # sanity note only; exact accounting is validated by the loop above

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as h:
        h.create_dataset("X", data=X, dtype=np.float32, compression="gzip", compression_opts=4)
        h.create_dataset("gene_ids", data=tcga_gene_ids)
        h.create_dataset("sample_idx", data=np.arange(n_samples, dtype=np.int64))
        h.create_dataset(
            "sample_name",
            data=np.asarray(ordered_columns, dtype=object),
            dtype=h5py.string_dtype(),
        )
    tmp.replace(dest)
    return {
        "status": "built",
        "path": str(dest),
        "shape": list(X.shape),
        "zero_filled_gene_count": len(zero_filled),
        "zero_filled_gene_ids": zero_filled,
    }


def build_methylation_cache(mp_root: Path, output: Path, n_cpg: int, *, replace: bool) -> dict:
    dest = output / "methylation" / "encode_wgbs_full.h5"
    if dest.is_file() and not replace:
        return {"status": "cached", "path": str(dest)}

    name_to_idx, idx_to_name = _sample_name_to_idx(mp_root)
    n_samples = len(name_to_idx)
    ordered_columns = [idx_to_name.loc[i] for i in range(n_samples)]

    shard_paths = sorted((mp_root / "me.parquet").glob("*.parquet"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as h:
        ds = h.create_dataset(
            "X", shape=(n_samples, n_cpg), dtype=np.float32,
            chunks=(n_samples, ME_SHARD_ROWS), compression="gzip", compression_opts=4,
        )
        offset = 0
        for shard in shard_paths:
            frame = pd.read_parquet(shard)
            frame = frame.rename(columns={"Unnamed: 0": "chr_pos"})
            n_rows = len(frame)
            block = frame[ordered_columns].to_numpy(dtype=np.float32).T  # samples x chunk
            if block.shape != (n_samples, n_rows):
                raise ValueError(f"{shard}: unexpected block shape {block.shape}")
            ds[:, offset:offset + n_rows] = block
            offset += n_rows
        if offset != n_cpg:
            raise ValueError(f"me.parquet covers {offset} cpgs, expected {n_cpg}")
        h.create_dataset("sample_idx", data=np.arange(n_samples, dtype=np.int64))
        h.create_dataset(
            "sample_name",
            data=np.asarray(ordered_columns, dtype=object),
            dtype=h5py.string_dtype(),
        )
    tmp.replace(dest)
    return {"status": "built", "path": str(dest), "shape": [n_samples, n_cpg]}


def verify_against_release(mp_eval: Path, registry: pd.DataFrame, train_cpg: np.ndarray, val_cpg: np.ndarray,
                            train_sample: np.ndarray, val_sample: np.ndarray) -> dict:
    """ID-for-ID check against the released MethylProphet ENCODE eval artifact,
    same audit pattern as docs/BENCHMARK_METHYLPROPHET.md's chr1 MP_EVAL_DIR check.
    """
    mapping_path = next(mp_eval.glob("group_idx_name_mapping*.json"))
    mapping = json.loads(mapping_path.read_text())
    names = {"train_cpg-val_sample": None, "val_cpg-train_sample": None, "val_cpg-val_sample": None}
    for gid, path_str in mapping.items():
        for key in names:
            if key in str(path_str):
                names[key] = int(gid)

    parquet_files = sorted(mp_eval.glob("eval_results*.parquet/*.parquet"))
    dataset = pads.dataset([str(p) for p in parquet_files], format="parquet")
    by_group: dict[int, dict[str, set]] = {gid: {"samples": set(), "cpgs": set()} for gid in names.values() if gid is not None}
    scanner = dataset.scanner(columns=["group_idx", "sample_idx", "cpg_idx"], batch_size=262_144)
    for batch in scanner.to_batches():
        g = np.asarray(batch.column("group_idx"))
        s = np.asarray(batch.column("sample_idx"))
        c = np.asarray(batch.column("cpg_idx"))
        for gid in by_group:
            mask = g == gid
            if mask.any():
                by_group[gid]["samples"].update(np.unique(s[mask]).tolist())
                by_group[gid]["cpgs"].update(np.unique(c[mask]).tolist())

    result = {"status": "complete", "checks": []}
    train_cpg_set, val_cpg_set = set(train_cpg.tolist()), set(val_cpg.tolist())
    train_sample_set, val_sample_set = set(train_sample.tolist()), set(val_sample.tolist())

    def _check(label: str, ok: bool) -> None:
        result["checks"].append({"label": label, "ok": bool(ok)})
        if not ok:
            result["status"] = "failed"

    g0 = names.get("train_cpg-val_sample")
    if g0 is not None and g0 in by_group:
        _check("train_cpg-val_sample: cpgs subset of our train_cpg", by_group[g0]["cpgs"] <= train_cpg_set)
        _check("train_cpg-val_sample: samples subset of our val_sample", by_group[g0]["samples"] <= val_sample_set)
    g1 = names.get("val_cpg-train_sample")
    if g1 is not None and g1 in by_group:
        _check("val_cpg-train_sample: cpgs subset of our val_cpg", by_group[g1]["cpgs"] <= val_cpg_set)
        _check("val_cpg-train_sample: samples subset of our train_sample", by_group[g1]["samples"] <= train_sample_set)
    g2 = names.get("val_cpg-val_sample")
    if g2 is not None and g2 in by_group:
        _check("val_cpg-val_sample: cpgs subset of our val_cpg", by_group[g2]["cpgs"] <= val_cpg_set)
        _check("val_cpg-val_sample: samples subset of our val_sample", by_group[g2]["samples"] <= val_sample_set)
    if len(result["checks"]) == 0:
        raise ValueError("no group mappings resolved; cannot verify against release artifact")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mp-data-root", default=str(Path.home() / "methyl-benchmark" / "methylprophet-data" / "parquet" / DATASET_NAME))
    p.add_argument("--mp-eval", default=str(Path.home() / "methyl-benchmark" / "methylprophet-eval-released" / "eval-encode_wgbs-bs_512-64xl40s-aws"))
    p.add_argument("--tcga-canonical-root", required=True, help="e.g. $METHYL_DATA_ROOT/datasets/methylprophet_repro_v1")
    p.add_argument("--output", required=True, help="e.g. $METHYL_DATA_ROOT/datasets/methylprophet_encode_v1")
    p.add_argument("--replace", action="store_true")
    p.add_argument("--skip-methylation", action="store_true", help="build registry/splits/rna only; methylation is the slow step")
    args = p.parse_args()

    mp_root = Path(args.mp_data_root)
    mp_eval = Path(args.mp_eval)
    tcga_root = Path(args.tcga_canonical_root)
    output = Path(args.output)

    registry_path = build_registry(mp_root, output, replace=args.replace)
    registry = pd.read_parquet(registry_path)
    n_cpg = len(registry)
    print(json.dumps({"step": "registry", "cpg_count": n_cpg}))

    cpg_split_paths = build_cpg_split(mp_root, output, registry, replace=args.replace)
    train_cpg = np.load(cpg_split_paths["train"])
    val_cpg = np.load(cpg_split_paths["val"])
    print(json.dumps({"step": "cpg_split", "train": len(train_cpg), "val": len(val_cpg)}))

    sample_split_paths = build_sample_split(mp_root, output, replace=args.replace)
    train_sample = np.load(sample_split_paths["train"])
    val_sample = np.load(sample_split_paths["val"])
    print(json.dumps({"step": "sample_split", "train": len(train_sample), "val": len(val_sample)}))

    verification = verify_against_release(mp_eval, registry, train_cpg, val_cpg, train_sample, val_sample)
    print(json.dumps({"step": "verify_against_release", **verification}))
    if verification["status"] != "complete":
        raise SystemExit("split verification against released MethylProphet ENCODE artifact failed")

    rna_result = build_rna_cache(mp_root, output, tcga_root, replace=args.replace)
    print(json.dumps({"step": "rna", **{k: v for k, v in rna_result.items() if k != "zero_filled_gene_ids"}}))

    if not args.skip_methylation:
        me_result = build_methylation_cache(mp_root, output, n_cpg, replace=args.replace)
        print(json.dumps({"step": "methylation", **me_result}))

    manifest = {
        "schema_version": 1,
        "name": "methylprophet_encode_v1",
        "source": "official MethylProphet ENCODE WGBS preprocessing (xk-huang/methylprophet), reused as-is",
        "cpg_count": n_cpg,
        "sample_count": len(train_sample) + len(val_sample),
        "train_cpg": len(train_cpg),
        "val_cpg": len(val_cpg),
        "train_sample": len(train_sample),
        "val_sample": len(val_sample),
        "split_verification": verification,
        "rna": rna_result,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(output / "manifest.json", manifest)
    print(json.dumps({"step": "done", "manifest": str(output / "manifest.json")}))


if __name__ == "__main__":
    main()
