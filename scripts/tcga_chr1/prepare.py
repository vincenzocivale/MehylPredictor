#!/usr/bin/env python3
"""Prepare the exact MethylProphet TCGA Table-5 training/evaluation universe.

The Table-5 TCGA benchmark is chromosome 1.  Preparation is fail-closed on the
published sample/CpG/finite-pair counts.  It performs no NTv3 inference:
embeddings are copied from the consolidated NTv3 atlas.

Unlike the earlier generic ``tcga_mix_chr1`` adapter, this command reproduces
MethylProphet's MDS filtering rule that drops CpGs whose 1000-bp hg38 window
contains ``N``.  It also reconstructs the official Array sample split
(seed=42, cancer-stratified 90/10) or, when ``--mp-eval`` is supplied, takes
the exact Array IDs directly from the released MethylProphet evaluation rows.

The genomic prior is rebuilt specifically for this benchmark from Table-5
Array TRAIN data only (8260 samples x 33885 train CpGs), with 5-fold OOF prior
predictions for train CpGs and a full-fit NTv3->prior probe for unseen loci.
No genome-wide TCGA methylation supervision enters this benchmark.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path
import re

import pyarrow  # noqa: F401 -- load before h5py/torch on the target server
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml

from methylation_predictor.full_suite.cache import prepare_rna_cache
from methylation_predictor.full_suite.probe import ScalarProbeNet
from methylation_predictor.benchmark.table5.protocol import (
    ARRAY_VIEW_EXPECTED_OBSERVED,
    SOURCE_EXPECTED_OBSERVED,
    TABLE5_EXPECTED,
    TABLE5_PROTOCOL_NAME,
    Table5Protocol,
    sha256_ids,
)
from methylation_predictor.tcga_canonical import TCGACanonicalBundle, load_protocol


PROTOCOL_FILES = {
    "array_train_sample_idx": "array_train_sample_idx.npy",
    "array_val_sample_idx": "array_val_sample_idx.npy",
    "array_train_cpg_idx": "array_train_cpg_idx.npy",
    "array_val_cpg_idx": "array_val_cpg_idx.npy",
    "epic_train_cpg_idx": "epic_train_cpg_idx.npy",
    "wgbs_train_cpg_idx": "wgbs_train_cpg_idx.npy",
}


def positions(ids: np.ndarray, query: np.ndarray, name: str) -> np.ndarray:
    ids = np.asarray(ids, np.int64); query = np.asarray(query, np.int64)
    order = np.argsort(ids, kind="mergesort"); sorted_ids = ids[order]
    if len(sorted_ids) and np.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise RuntimeError(f"{name} contains duplicate IDs")
    pos = np.searchsorted(sorted_ids, query)
    clipped = np.minimum(pos, max(len(sorted_ids) - 1, 0))
    ok = (pos < len(sorted_ids)) & (sorted_ids[clipped] == query)
    if not np.all(ok):
        missing = query[~ok]
        raise RuntimeError(f"{name} missing {len(missing)} required IDs; examples={missing[:10].tolist()}")
    return order[pos]


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
        for x in np.asarray(values).tolist()
    ], dtype=object)


def _reconstruct_array_sample_split(root: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Reproduce MethylProphet ``split_sample_tcga.py`` for the 9178 Array rows."""
    with h5py.File(root / "methylation" / "tcga_array_official_full.h5", "r") as h:
        array_ids = np.asarray(h["sample_idx"][...], dtype=np.int64)
    if len(array_ids) != TABLE5_EXPECTED["array_train_samples"] + TABLE5_EXPECTED["array_val_samples"]:
        raise RuntimeError(f"Table-5 Array source must contain 9178 rows, found {len(array_ids)}")

    with h5py.File(root / "rna" / "tcga_rna_official_full.h5", "r") as h:
        rna_ids = np.asarray(h["sample_idx"][...], dtype=np.int64)
        if "tissue_idx" in h:
            tissue_idx = np.asarray(h["tissue_idx"][...], dtype=np.int64)
        elif "tissue_name" in h:
            # Canonical RNA carries cancer-type as a string label rather than
            # a precomputed integer code; factorize it into the same role
            # (only relative grouping matters for the stratified split).
            tissue_idx = pd.factorize(_decode(h["tissue_name"][...]), sort=True)[0]
        else:
            raise RuntimeError(
                "canonical RNA lacks tissue_idx/tissue_name required for MethylProphet ind_cancer split"
            )
    p = positions(rna_ids, array_ids, "RNA sample_idx")
    frame = pd.DataFrame(
        {"sample_idx": array_ids, "tissue_idx": tissue_idx[p]}, index=array_ids
    ).sort_index()

    rng = np.random.default_rng(42)
    train_parts = []; val_parts = []
    for _, group in frame.groupby("tissue_idx", sort=True):
        if len(group) == 1:
            train_parts.append(group)
            continue
        train = group.sample(frac=0.9, random_state=rng)
        val = group.drop(train.index)
        train_parts.append(train); val_parts.append(val)
    train_ids = np.sort(pd.concat(train_parts)["sample_idx"].to_numpy(np.int64))
    val_ids = np.sort(pd.concat(val_parts)["sample_idx"].to_numpy(np.int64))
    if len(train_ids) != TABLE5_EXPECTED["array_train_samples"] or len(val_ids) != TABLE5_EXPECTED["array_val_samples"]:
        raise RuntimeError(
            "official seed=42 reconstruction did not yield Table-5 sample counts: "
            f"train={len(train_ids)}, val={len(val_ids)}"
        )
    return train_ids, val_ids, {
        "method": "reimplemented xk-huang/MethylProphet split_sample_tcga.py",
        "seed": 42,
        "val_fraction": 0.1,
        "stratification": "tissue_idx (ind_cancer)",
        "input_order": "sample_idx ascending after canonical Array/WGBS-overlap exclusion",
    }


def _eval_group_mapping(root: Path) -> dict[str, int]:
    names = {
        "train_cpg_x_val_sample": "train_cpg-val_sample",
        "val_cpg_x_train_sample": "val_cpg-train_sample",
        "val_cpg_x_val_sample": "val_cpg-val_sample",
    }
    for path in sorted(root.rglob("group_idx_name_mapping*.json")):
        try:
            raw = json.loads(path.read_text())
        except Exception:
            continue
        found = {}
        for key, value in raw.items():
            text = str(value)
            for canonical, needle in names.items():
                if needle in text:
                    found[canonical] = int(key)
        if len(found) == 3:
            return found
    return {
        "train_cpg_x_val_sample": 0,
        "val_cpg_x_train_sample": 1,
        "val_cpg_x_val_sample": 2,
    }


def _array_ids_from_mp_eval(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Extract exact Array ID sets from released MethylProphet prediction rows."""
    mapping = _eval_group_mapping(path)
    by_group = {gid: {"samples": set(), "cpgs": set()} for gid in mapping.values()}
    parquet_candidates = sorted(path.rglob("*.parquet")) if path.is_dir() else ([path] if path.suffix == ".parquet" else [])
    required_columns = {"group_idx", "sample_idx", "cpg_idx"}
    parquet_files = []
    for candidate in parquet_candidates:
        try:
            if required_columns.issubset(set(pq.read_schema(candidate).names)):
                parquet_files.append(candidate)
        except Exception:
            continue
    csv_files = [] if parquet_files else (
        sorted(path.rglob("eval_results*.csv")) if path.is_dir() else ([path] if path.suffix == ".csv" else [])
    )
    if parquet_files:
        dataset = pads.dataset([str(x) for x in parquet_files], format="parquet")
        scanner = dataset.scanner(columns=["group_idx", "sample_idx", "cpg_idx"], batch_size=262_144)
        for batch in scanner.to_batches():
            g = np.asarray(batch.column("group_idx"), dtype=np.int64)
            s = np.asarray(batch.column("sample_idx"), dtype=np.int64)
            c = np.asarray(batch.column("cpg_idx"), dtype=np.int64)
            for gid in by_group:
                mask = g == gid
                if mask.any():
                    by_group[gid]["samples"].update(np.unique(s[mask]).tolist())
                    by_group[gid]["cpgs"].update(np.unique(c[mask]).tolist())
    elif csv_files:
        for csv in csv_files:
            for chunk in pd.read_csv(csv, usecols=["group_idx", "sample_idx", "cpg_idx"], chunksize=1_000_000):
                for gid in by_group:
                    local = chunk.loc[chunk.group_idx.eq(gid)]
                    if len(local):
                        by_group[gid]["samples"].update(local.sample_idx.astype(int).unique().tolist())
                        by_group[gid]["cpgs"].update(local.cpg_idx.astype(int).unique().tolist())
    else:
        raise RuntimeError(
            f"--mp-eval must contain released eval parquet files or eval_results*.csv: {path}"
        )

    g0 = by_group[mapping["train_cpg_x_val_sample"]]
    g1 = by_group[mapping["val_cpg_x_train_sample"]]
    g2 = by_group[mapping["val_cpg_x_val_sample"]]
    train_s = np.sort(np.asarray(list(g1["samples"]), np.int64))
    val_s = np.sort(np.asarray(list(g0["samples"]), np.int64))
    train_c = np.sort(np.asarray(list(g0["cpgs"]), np.int64))
    val_c = np.sort(np.asarray(list(g1["cpgs"]), np.int64))
    if set(g2["samples"]) != set(val_s.tolist()) or set(g2["cpgs"]) != set(val_c.tolist()):
        raise RuntimeError("released MethylProphet eval groups disagree on validation IDs")
    expected = (
        TABLE5_EXPECTED["array_train_samples"],
        TABLE5_EXPECTED["array_val_samples"],
        TABLE5_EXPECTED["array_train_cpgs"],
        TABLE5_EXPECTED["array_val_cpgs"],
    )
    actual = (len(train_s), len(val_s), len(train_c), len(val_c))
    if actual != expected:
        raise RuntimeError(f"released MethylProphet eval ID counts {actual} != Table-5 {expected}")
    return train_s, val_s, train_c, val_c, {
        "method": "IDs extracted directly from released MethylProphet evaluation rows",
        "path": str(path),
        "group_mapping": mapping,
    }


def _registry_path(root: Path, source: str) -> Path:
    for candidate in (
        root / "cpg" / "registries" / f"{source}_cpg_map.parquet",
        root / "registries" / f"{source}_cpg_map.parquet",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot find {source}_cpg_map.parquet under {root}")


def _positions_from_registry(path: Path, target_ids: np.ndarray) -> np.ndarray:
    """Stream a potentially 23M-row registry and align only requested positions."""
    target_ids = np.asarray(target_ids, np.int64)
    order = np.argsort(target_ids); target_sorted = target_ids[order]
    found_id = []; found_pos = []
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=["cpg_idx", "pos"], batch_size=524_288):
        ids = np.asarray(batch.column(0).to_numpy(zero_copy_only=False), dtype=np.int64)
        p = np.searchsorted(target_sorted, ids)
        clipped = np.minimum(p, max(len(target_sorted) - 1, 0))
        mask = (p < len(target_sorted)) & (target_sorted[clipped] == ids)
        if mask.any():
            found_id.append(ids[mask])
            found_pos.append(np.asarray(batch.column(1).to_numpy(zero_copy_only=False), dtype=np.int64)[mask])
    if not found_id:
        raise RuntimeError(f"registry {path} contains none of the required IDs")
    ids = np.concatenate(found_id); pos = np.concatenate(found_pos)
    if len(ids) != len(target_ids) or len(np.unique(ids)) != len(target_ids):
        raise RuntimeError(f"registry {path} resolved {len(np.unique(ids))}/{len(target_ids)} required IDs")
    aligned = pos[positions(ids, target_ids, f"{path.name} cpg_idx")]
    return aligned.astype(np.int64)


def _chr1_n_intervals(fasta: Path) -> tuple[int, np.ndarray, np.ndarray]:
    """Return inclusive CpG-position intervals whose 1000-bp window has N/padding."""
    opener = gzip.open if fasta.suffix.lower() == ".gz" else open
    target = False; offset = 0; intervals: list[list[int]] = []
    with opener(fasta, "rt") as handle:
        for raw in handle:
            if raw.startswith(">"):
                name = raw[1:].split()[0]
                if target:
                    break
                target = name.lower() in {"chr1", "1"}
                continue
            if not target:
                continue
            seq = raw.strip()
            for match in re.finditer(r"[Nn]+", seq):
                a = offset + match.start(); b = offset + match.end()  # [a,b)
                lo = a - 499; hi = b + 499  # inclusive positions whose [p-500,p+500) intersects run
                if intervals and lo <= intervals[-1][1] + 1:
                    intervals[-1][1] = max(intervals[-1][1], hi)
                else:
                    intervals.append([lo, hi])
            offset += len(seq)
    if offset == 0:
        raise RuntimeError(f"FASTA {fasta} has no chr1/1 sequence")
    # Official preprocessing pads sequence beyond chromosome edges with N.
    intervals = [[0, 499], *intervals, [offset - 499, offset - 1]]
    intervals.sort()
    merged: list[list[int]] = []
    for lo, hi in intervals:
        lo = max(0, lo); hi = min(offset - 1, hi)
        if merged and lo <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    starts = np.asarray([x[0] for x in merged], np.int64)
    ends = np.asarray([x[1] for x in merged], np.int64)
    return offset, starts, ends


def _filter_no_n(ids: np.ndarray, pos: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.searchsorted(starts, pos, side="right") - 1
    invalid = idx >= 0
    safe_idx = np.maximum(idx, 0)
    invalid &= pos <= ends[safe_idx]
    return np.asarray(ids[~invalid], np.int64), np.asarray(ids[invalid], np.int64)


def _verify_cached_protocol_against_mp_eval(protocol: Table5Protocol, mp_eval: Path) -> None:
    train_s, val_s, train_c, val_c, _ = _array_ids_from_mp_eval(mp_eval)
    checks = {
        "array_train_sample_idx": (protocol.array_train_sample_idx, train_s),
        "array_val_sample_idx": (protocol.array_val_sample_idx, val_s),
        "array_train_cpg_idx": (protocol.array_train_cpg_idx, train_c),
        "array_val_cpg_idx": (protocol.array_val_cpg_idx, val_c),
    }
    mismatches = [name for name, (left, right) in checks.items() if not np.array_equal(left, right)]
    if mismatches:
        raise RuntimeError(
            "cached Table-5 protocol disagrees with released MethylProphet evaluation IDs: "
            + ", ".join(mismatches)
        )
    print("[Table5] cached Array IDs verified directly against released MethylProphet eval", flush=True)


def build_table5_protocol(root: Path, fasta: Path, output: Path, mp_eval: Path | None) -> Table5Protocol:
    output.mkdir(parents=True, exist_ok=True)
    done = output / ".ids_done"
    if done.is_file():
        protocol = Table5Protocol.load(output)
        if mp_eval is not None:
            _verify_cached_protocol_against_mp_eval(protocol, mp_eval)
        return protocol

    with TCGACanonicalBundle.from_root(root) as bundle:
        current_array = load_protocol("tcga_array_chr1", bundle)
        current_mix = load_protocol("tcga_mix_chr1", bundle)
        array_train_c = np.sort(np.asarray(current_array.array_train_cpg_idx, np.int64))
        array_val_c = np.sort(np.asarray(current_array.array_val_cpg_idx, np.int64))
        epic_candidates = np.sort(np.asarray(current_mix.auxiliary_cpg_idx["epic"], np.int64))
        wgbs_candidates = np.sort(np.asarray(current_mix.auxiliary_cpg_idx["wgbs"], np.int64))

    if mp_eval is not None:
        train_s, val_s, eval_train_c, eval_val_c, sample_provenance = _array_ids_from_mp_eval(mp_eval)
        if not np.array_equal(array_train_c, eval_train_c) or not np.array_equal(array_val_c, eval_val_c):
            raise RuntimeError("canonical Array CpG split does not match released Table-5 evaluation IDs")
        array_provenance = sample_provenance
    else:
        train_s, val_s, sample_provenance = _reconstruct_array_sample_split(root)
        array_provenance = sample_provenance

    if len(array_train_c) != TABLE5_EXPECTED["array_train_cpgs"] or len(array_val_c) != TABLE5_EXPECTED["array_val_cpgs"]:
        raise RuntimeError("canonical Array chr1 CpG split is not the Table-5 split")

    chr_len, n_starts, n_ends = _chr1_n_intervals(fasta)
    epic_pos = _positions_from_registry(_registry_path(root, "epic"), epic_candidates)
    wgbs_pos = _positions_from_registry(_registry_path(root, "wgbs"), wgbs_candidates)
    epic_exact, epic_removed = _filter_no_n(epic_candidates, epic_pos, n_starts, n_ends)
    wgbs_exact, wgbs_removed = _filter_no_n(wgbs_candidates, wgbs_pos, n_starts, n_ends)
    epic_exact.sort(); wgbs_exact.sort(); epic_removed.sort(); wgbs_removed.sort()
    if len(epic_exact) != TABLE5_EXPECTED["epic_train_cpgs"]:
        raise RuntimeError(
            f"MethylProphet N-filter gives {len(epic_exact):,} EPIC chr1 CpGs, expected 71,748"
        )
    if len(wgbs_exact) != TABLE5_EXPECTED["wgbs_train_cpgs"]:
        raise RuntimeError(
            f"MethylProphet N-filter gives {len(wgbs_exact):,} WGBS chr1 CpGs, expected 1,999,446"
        )

    arrays = {
        "array_train_sample_idx": np.sort(train_s),
        "array_val_sample_idx": np.sort(val_s),
        "array_train_cpg_idx": array_train_c,
        "array_val_cpg_idx": array_val_c,
        "epic_train_cpg_idx": epic_exact,
        "wgbs_train_cpg_idx": wgbs_exact,
    }
    for field, filename in PROTOCOL_FILES.items():
        np.save(output / filename, arrays[field])
    np.save(output / "epic_removed_methylprophet_n_window.npy", epic_removed)
    np.save(output / "wgbs_removed_methylprophet_n_window.npy", wgbs_removed)

    report = {
        "protocol": TABLE5_PROTOCOL_NAME,
        "status": "ids_reconstructed_pending_finite_pair_audit",
        "reference": "MethylProphet Table 5 TCGA chromosome-1 experiment",
        "array_split": array_provenance,
        "auxiliary_cpg_filter": {
            "source": "MethylProphet convert_parquet_to_mds.py",
            "rule": "chr1 AND central 1000-bp hg38 sequence contains zero N bases",
            "hg38_fasta": str(fasta),
            "chr1_length": int(chr_len),
            "epic_candidates_before_n_filter": int(len(epic_candidates)),
            "epic_removed_by_n_filter": int(len(epic_removed)),
            "wgbs_candidates_before_n_filter": int(len(wgbs_candidates)),
            "wgbs_removed_by_n_filter": int(len(wgbs_removed)),
        },
        "counts": {
            "array_train_samples": int(len(train_s)),
            "array_val_samples": int(len(val_s)),
            "array_train_cpgs": int(len(array_train_c)),
            "array_val_cpgs": int(len(array_val_c)),
            "epic_train_cpgs": int(len(epic_exact)),
            "wgbs_train_cpgs": int(len(wgbs_exact)),
        },
        "sha256_ids": {field: sha256_ids(values) for field, values in arrays.items()},
        "finite_pair_audit": None,
    }
    (output / "protocol.json").write_text(json.dumps(report, indent=2) + "\n")
    done.write_text("ok\n")
    return Table5Protocol.load(output)


def build_exact_compact_cache(
    root: Path,
    source_name: str,
    output: Path,
    *,
    sample_ids: np.ndarray,
    cpg_ids: np.ndarray,
    row_chunk: int = 16,
) -> dict[str, object]:
    """Materialize exact Table-5 rows/columns in training-first contiguous order."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        with h5py.File(output, "r") as h:
            if h.attrs.get("protocol", "") != TABLE5_PROTOCOL_NAME:
                raise RuntimeError(f"stale non-Table5 cache: {output}")
            return {"status": "cached", "shape": list(h["beta"].shape)}
    with TCGACanonicalBundle.from_root(root) as bundle:
        src = bundle.sources[source_name]
        source_rows = positions(src.sample_idx, np.asarray(sample_ids, np.int64), f"{source_name} sample_idx")
        source_cols = src._cpg_index.positions_of(np.asarray(cpg_ids, np.int64))
        beta = src.h5["beta"]
        tmp = output.with_suffix(".tmp.h5")
        with h5py.File(tmp, "w") as h:
            ds = h.create_dataset(
                "beta",
                shape=(len(sample_ids), len(cpg_ids)),
                dtype="f4",
                chunks=(min(128, len(sample_ids)), min(2048, len(cpg_ids))),
            )
            for start in range(0, len(sample_ids), row_chunk):
                stop = min(start + row_chunk, len(sample_ids))
                physical = source_rows[start:stop]
                read_order = np.argsort(physical)
                full = np.asarray(beta[physical[read_order], :], np.float32)
                inverse = np.empty_like(read_order); inverse[read_order] = np.arange(len(read_order))
                ds[start:stop] = full[inverse][:, source_cols]
                if start == 0 or stop == len(sample_ids) or (start // row_chunk) % 50 == 0:
                    print(f"[{source_name}-cache] rows {stop:,}/{len(sample_ids):,}", flush=True)
            h.create_dataset("sample_idx", data=np.asarray(sample_ids, np.int64), dtype="i8")
            h.create_dataset("cpg_idx", data=np.asarray(cpg_ids, np.int64), dtype="i8")
            h.attrs["source"] = source_name
            h.attrs["protocol"] = TABLE5_PROTOCOL_NAME
        os.replace(tmp, output)
    return {"status": "built", "shape": [int(len(sample_ids)), int(len(cpg_ids))]}


def _finite_cache(path: Path, sample_ids: np.ndarray, cpg_ids: np.ndarray, row_chunk: int = 128) -> int:
    with h5py.File(path, "r") as h:
        rows = positions(np.asarray(h["sample_idx"][...], np.int64), sample_ids, f"{path.name} samples")
        cols = positions(np.asarray(h["cpg_idx"][...], np.int64), cpg_ids, f"{path.name} CpGs")
        total = 0
        for start in range(0, len(rows), row_chunk):
            local = rows[start:start + row_chunk]
            # The exact caches are ordered by Table-5 IDs, so these are normally contiguous.
            if len(local) and np.all(np.diff(local) == 1):
                block = np.asarray(h["beta"][local[0]:local[-1] + 1, :], np.float32)[:, cols]
            else:
                read_order = np.argsort(local)
                block = np.asarray(h["beta"][local[read_order], :], np.float32)
                inverse = np.empty_like(read_order); inverse[read_order] = np.arange(len(read_order))
                block = block[inverse][:, cols]
            total += int(np.isfinite(block).sum())
        return total


def audit_finite_pairs(root: Path, protocol: Table5Protocol, array_cache: Path, epic_cache: Path) -> dict[str, object]:
    with h5py.File(root / "methylation" / "epic_full.h5", "r") as h:
        epic_sample_idx = np.asarray(h["sample_idx"][...], np.int64)
    source = {
        "array": _finite_cache(array_cache, protocol.array_train_sample_idx, protocol.array_train_cpg_idx),
        "epic": _finite_cache(epic_cache, epic_sample_idx, protocol.epic_train_cpg_idx),
    }
    with TCGACanonicalBundle.from_root(root) as bundle:
        wgbs = bundle.sources["wgbs"]
        source["wgbs"] = wgbs.finite_count(np.arange(wgbs.n_rows, dtype=np.int64), protocol.wgbs_train_cpg_idx)
    views = {
        "train_cpg_x_val_sample": _finite_cache(array_cache, protocol.array_val_sample_idx, protocol.array_train_cpg_idx),
        "val_cpg_x_train_sample": _finite_cache(array_cache, protocol.array_train_sample_idx, protocol.array_val_cpg_idx),
        "val_cpg_x_val_sample": _finite_cache(array_cache, protocol.array_val_sample_idx, protocol.array_val_cpg_idx),
    }
    errors = []
    for name, expected in SOURCE_EXPECTED_OBSERVED.items():
        if source[name] != expected:
            errors.append(f"train {name}: {source[name]:,} != {expected:,}")
    for name, expected in ARRAY_VIEW_EXPECTED_OBSERVED.items():
        if views[name] != expected:
            errors.append(f"view {name}: {views[name]:,} != {expected:,}")
    total = int(sum(source.values()))
    if total != TABLE5_EXPECTED["total_train_observed"]:
        errors.append(f"total training pairs: {total:,} != {TABLE5_EXPECTED['total_train_observed']:,}")
    if errors:
        raise RuntimeError("Table-5 finite-pair audit FAILED:\n  " + "\n  ".join(errors))
    return {
        "status": "exact_match",
        "training_observed": source,
        "training_total_observed": total,
        "evaluation_observed": views,
    }


class PriorEnsemble(nn.Module):
    def __init__(self, seeds: tuple[int, ...], dim: int = 1536, dropout: float = 0.1):
        super().__init__()
        nets = []
        for seed in seeds:
            torch.manual_seed(seed)
            nets.append(ScalarProbeNet(dim=dim, dropout=dropout))
        self.nets = nn.ModuleList(nets)

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([net(x) for net in self.nets], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_all(x).mean(dim=0)


def _embedding_stats(x: np.ndarray, rows: np.ndarray, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    sx = np.zeros(x.shape[1], np.float64); sxx = np.zeros(x.shape[1], np.float64)
    for start in range(0, len(rows), chunk):
        b = np.asarray(x[rows[start:start + chunk]], np.float32)
        sx += b.sum(0, dtype=np.float64); sxx += np.square(b, dtype=np.float64).sum(0)
    mean = sx / len(rows); var = np.maximum(sxx / len(rows) - mean * mean, 1e-8)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def _fit_prior_probe(
    embeddings: np.ndarray,
    target_logit: np.ndarray,
    fit_rows: np.ndarray,
    predict_rows: np.ndarray,
    *,
    device: str,
    scope_seed: int,
    max_epochs: int = 40,
    patience: int = 6,
    block_rows: int = 4096,
) -> tuple[np.ndarray, dict[str, object], dict[str, object]]:
    ids = np.arange(len(embeddings), dtype=np.int64)
    ordered = fit_rows[np.argsort(ids[fit_rows])]
    rng = np.random.default_rng(scope_seed)
    shuffled = ordered[rng.permutation(len(ordered))]
    n_val = max(1000, int(round(0.10 * len(shuffled))))
    inner_val = np.sort(shuffled[:n_val]); inner_train = np.sort(shuffled[n_val:])
    xm, xs = _embedding_stats(embeddings, inner_train, block_rows)
    y = target_logit
    ym = np.float32(y[inner_train].mean()); ys = np.float32(max(y[inner_train].std(), 1e-6))
    seeds = (17, 29, 43)
    model = PriorEnsemble(seeds).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    amp = str(device).startswith("cuda")
    xmt = torch.from_numpy(xm).to(device); xst = torch.from_numpy(xs).to(device)
    best = math.inf; best_epoch = 0; bad = 0; history = []
    blocks = [inner_train[s:s + block_rows] for s in range(0, len(inner_train), block_rows)]
    for epoch in range(1, max_epochs + 1):
        model.train(); losses = []
        for bi in rng.permutation(len(blocks)):
            rows = blocks[int(bi)]
            xb = torch.from_numpy(np.asarray(embeddings[rows], np.float32)).to(device)
            yb = torch.from_numpy(((y[rows] - ym) / ys).astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                pred = model.forward_all((xb - xmt) / xst)
                loss = torch.stack([torch.nn.functional.mse_loss(v, yb) for v in pred]).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval(); sse = 0.0; n = 0
        with torch.inference_mode():
            for start in range(0, len(inner_val), block_rows):
                rows = inner_val[start:start + block_rows]
                xb = torch.from_numpy(np.asarray(embeddings[rows], np.float32)).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                    pred = model((xb - xmt) / xst)
                p = pred.float().cpu().numpy() * ys + ym
                sse += float(np.square(p - y[rows], dtype=np.float64).sum()); n += len(rows)
        val = sse / n
        history.append({"epoch": epoch, "train_mse_z": float(np.mean(losses)), "validation_mse_logit": val})
        if val < best - 1e-5:
            best = val; best_epoch = epoch; bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    if best_epoch < 1:
        raise RuntimeError("Table-5 prior probe selection failed")

    # Deterministic full refit for exactly the selected number of epochs.
    xm, xs = _embedding_stats(embeddings, fit_rows, block_rows)
    ym = np.float32(y[fit_rows].mean()); ys = np.float32(max(y[fit_rows].std(), 1e-6))
    xmt = torch.from_numpy(xm).to(device); xst = torch.from_numpy(xs).to(device)
    model = PriorEnsemble(seeds).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    blocks = [fit_rows[s:s + block_rows] for s in range(0, len(fit_rows), block_rows)]
    rng = np.random.default_rng(scope_seed)
    for epoch in range(1, best_epoch + 1):
        model.train()
        for bi in rng.permutation(len(blocks)):
            rows = blocks[int(bi)]
            xb = torch.from_numpy(np.asarray(embeddings[rows], np.float32)).to(device)
            yb = torch.from_numpy(((y[rows] - ym) / ys).astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                pred = model.forward_all((xb - xmt) / xst)
                loss = torch.stack([torch.nn.functional.mse_loss(v, yb) for v in pred]).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    prediction = np.empty(len(predict_rows), np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(predict_rows), block_rows):
            stop = min(start + block_rows, len(predict_rows)); rows = predict_rows[start:stop]
            xb = torch.from_numpy(np.asarray(embeddings[rows], np.float32)).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                pred = model((xb - xmt) / xst)
            prediction[start:stop] = pred.float().cpu().numpy() * ys + ym
    state = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "seeds": list(seeds), "dim": 1536, "dropout": 0.1,
        "x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys,
        "best_epoch": int(best_epoch), "selection_validation_mse_logit": float(best),
        "target": "prior_logit",
    }
    metrics = {"best_epoch": int(best_epoch), "selection_validation_mse_logit": float(best), "history": history}
    return prediction, state, metrics


def _atlas_embeddings(atlas: Path, ids: np.ndarray, output: Path) -> np.ndarray:
    if output.is_file():
        x = np.load(output, mmap_mode="r")
        if x.shape != (len(ids), 1536):
            raise RuntimeError("stale Table-5 prior embedding cache")
        return x
    with h5py.File(atlas, "r") as h:
        atlas_ids = np.asarray(h["cpg_idx"][...], np.int64)
        rows = positions(atlas_ids, ids, "NTv3 atlas")
        mmap = np.lib.format.open_memmap(output, mode="w+", dtype=np.float16, shape=(len(ids), 1536))
        for start in range(0, len(ids), 4096):
            stop = min(start + 4096, len(ids)); physical = rows[start:stop]
            order = np.argsort(physical); block = np.asarray(h["embedding"][physical[order], :], np.float16)
            inverse = np.empty_like(order); inverse[order] = np.arange(len(order)); mmap[start:stop] = block[inverse]
        del mmap
    return np.load(output, mmap_mode="r")


def build_table5_prior(
    protocol: Table5Protocol,
    atlas: Path,
    array_cache: Path,
    output: Path,
    *,
    device: str,
) -> dict[str, object]:
    """Build leakage-safe Table-5-only NTv3 prior; no genome-wide methylation labels."""
    output.mkdir(parents=True, exist_ok=True); done = output / ".done"
    locus_path = output / "locus_features.parquet"; checkpoint = output / "full_fit_prior_probe.pt"
    sigma_checkpoint = output / "full_fit_sigma_probe.pt"
    if done.is_file() and locus_path.is_file() and checkpoint.is_file() and sigma_checkpoint.is_file():
        return json.loads((output / "manifest.json").read_text())

    array_ids = np.concatenate([protocol.array_train_cpg_idx, protocol.array_val_cpg_idx]).astype(np.int64)
    embeddings = _atlas_embeddings(atlas, array_ids, output / "array_chr1_embeddings.f16.npy")
    with h5py.File(array_cache, "r") as h:
        # Cache construction guarantees train rows and train CpGs are first.
        train_beta = h["beta"]
        n_s = len(protocol.array_train_sample_idx); n_c = len(protocol.array_train_cpg_idx)
        sums = np.zeros(n_c, np.float64); counts = np.zeros(n_c, np.int64)
        # Second moment accumulated in logit space (not beta space) since the
        # model's residual is added in logit space -- sigma_i must be the
        # inter-sample std of logit(beta), the natural scale of that residual.
        logit_sums = np.zeros(n_c, np.float64); logit_sumsq = np.zeros(n_c, np.float64)
        for start in range(0, n_s, 128):
            block = np.asarray(train_beta[start:min(start + 128, n_s), :n_c], np.float32)
            finite = np.isfinite(block); sums += np.where(finite, block, 0.0).sum(0, dtype=np.float64); counts += finite.sum(0)
            clipped = np.clip(block, 1e-4, 1 - 1e-4)
            logit_block = np.log(clipped / (1 - clipped))
            logit_block = np.where(finite, logit_block, 0.0)
            logit_sums += logit_block.sum(0, dtype=np.float64)
            logit_sumsq += (logit_block * logit_block).sum(0, dtype=np.float64)
    if np.any(counts == 0):
        raise RuntimeError("Table-5 train CpG with no finite training methylation")
    target_prior = np.clip(sums / counts, 1e-4, 1 - 1e-4).astype(np.float32)
    target_logit = np.log(target_prior / (1 - target_prior)).astype(np.float32)
    logit_mean = logit_sums / counts
    logit_var = np.maximum(logit_sumsq / counts - logit_mean ** 2, 0.0)
    # 1e-3 is a numerical floor only (avoid log(0) for the handful of exactly-
    # constant train CpGs); it is not the sigma_min used at training time to
    # standardize residual targets -- that is a separate, loss-side knob
    # (config.sigma_min) applied when consuming this cache.
    target_sigma = np.sqrt(logit_var).astype(np.float32)
    target_log_sigma = np.log(np.maximum(target_sigma, 1e-3)).astype(np.float32)

    n_train = len(protocol.array_train_cpg_idx); n_val = len(protocol.array_val_cpg_idx)
    fold_id = np.empty(n_train, np.int8)
    rng = np.random.default_rng(20260813); perm = rng.permutation(n_train)
    for fold, rows in enumerate(np.array_split(perm, 5)):
        fold_id[rows] = fold
    pred_logit = np.full(n_train + n_val, np.nan, np.float32)
    fold_metrics = {}
    for fold in range(5):
        predict = np.flatnonzero(fold_id == fold).astype(np.int64)
        fit = np.flatnonzero(fold_id != fold).astype(np.int64)
        pred, _, metrics = _fit_prior_probe(
            embeddings[:n_train], target_logit, fit, predict,
            device=device, scope_seed=20260813 + fold,
        )
        pred_logit[predict] = pred; fold_metrics[f"fold_{fold}"] = metrics
        print(f"[table5-prior] OOF fold {fold} complete", flush=True)
    full_predict = np.arange(n_train, n_train + n_val, dtype=np.int64)
    # Full-fit training target exists only for the first n_train rows; append dummy
    # values solely so prediction row indexing can address held-out embeddings.
    target_extended = np.concatenate([target_logit, np.zeros(n_val, np.float32)])
    pred, state, full_metrics = _fit_prior_probe(
        embeddings, target_extended, np.arange(n_train, dtype=np.int64), full_predict,
        device=device, scope_seed=20260913,
    )
    pred_logit[full_predict] = pred
    if not np.isfinite(pred_logit).all():
        raise RuntimeError("Table-5 prior predictions incomplete")
    torch.save(state, checkpoint)

    # V1 experiment: sigma_i, the per-CpG inter-sample std of logit(beta), fit
    # with the exact same leakage-safe OOF/full-fit probe machinery as mu_i
    # above (same 5-fold split, same embeddings) -- just against
    # target_log_sigma instead of target_logit.
    pred_log_sigma = np.full(n_train + n_val, np.nan, np.float32)
    sigma_fold_metrics = {}
    for fold in range(5):
        predict = np.flatnonzero(fold_id == fold).astype(np.int64)
        fit = np.flatnonzero(fold_id != fold).astype(np.int64)
        pred, _, metrics = _fit_prior_probe(
            embeddings[:n_train], target_log_sigma, fit, predict,
            device=device, scope_seed=20260814 + fold,
        )
        pred_log_sigma[predict] = pred; sigma_fold_metrics[f"fold_{fold}"] = metrics
        print(f"[table5-sigma] OOF fold {fold} complete", flush=True)
    target_log_sigma_extended = np.concatenate([target_log_sigma, np.zeros(n_val, np.float32)])
    pred, sigma_state, sigma_full_metrics = _fit_prior_probe(
        embeddings, target_log_sigma_extended, np.arange(n_train, dtype=np.int64), full_predict,
        device=device, scope_seed=20260914,
    )
    pred_log_sigma[full_predict] = pred
    if not np.isfinite(pred_log_sigma).all():
        raise RuntimeError("Table-5 sigma predictions incomplete")
    sigma_state["target"] = "log_sigma"
    torch.save(sigma_state, sigma_checkpoint)
    oof_sigma_mse = float(np.mean(np.square(
        pred_log_sigma[:n_train].astype(np.float64) - target_log_sigma.astype(np.float64)
    )))
    # Served sigma: exact empirical std for train CpGs (same leakage-safe
    # reasoning as the mu fix above); NTv3-probe prediction for held-out Array
    # CpGs (and, downstream in build_feature_cache, EPIC/WGBS-only CpGs).
    oof_sigma = np.exp(np.clip(pred_log_sigma, -20, 20)).astype(np.float32)
    sigma = oof_sigma.copy()
    sigma[:n_train] = target_sigma

    # Served prior: for the n_train Array train CpGs, the *exact* leakage-safe
    # empirical mean (target_prior, computed above from train samples only) is
    # available and strictly more accurate than routing it through the NTv3
    # probe -- the OOF prediction is still fit and scored (oof_mse below) as a
    # diagnostic of probe quality, but no longer served as the prior itself.
    # For the n_val held-out Array CpGs (and, downstream in build_feature_cache,
    # any EPIC/WGBS-only CpG), no true value exists at train time, so the
    # NTv3-probe prediction remains the only leakage-safe option.
    oof_prior = np.clip(1.0 / (1.0 + np.exp(-np.clip(pred_logit, -35, 35))), 1e-4, 1 - 1e-4).astype(np.float32)
    prior = oof_prior.copy()
    prior[:n_train] = target_prior
    pd.DataFrame({
        "cpg_idx": array_ids,
        "pred_ntv3_prior": prior,
        "pred_ntv3_prior_probe_only": oof_prior,  # diagnostic: what the NTv3 probe alone would have served
        "served_sigma": sigma,  # V1 experiment: per-CpG inter-sample std of logit(beta)
        "served_sigma_probe_only": oof_sigma,  # diagnostic: what the NTv3 probe alone would have served
        # Compatibility only: the final model ignores variability after the gate removal.
        "pred_log_var_between": np.zeros(len(array_ids), np.float32),
        "pred_log_var_within": np.zeros(len(array_ids), np.float32),
    }).to_parquet(locus_path, index=False, compression="zstd")
    oof_mse = float(np.mean(np.square(pred_logit[:n_train].astype(np.float64) - target_logit.astype(np.float64))))
    manifest = {
        "status": "built",
        "protocol": TABLE5_PROTOCOL_NAME,
        "training_supervision": "Array Table-5 train samples x train CpGs only",
        "train_samples": n_s,
        "train_cpgs": n_train,
        "heldout_cpgs": n_val,
        "served_train_cpg_prior": "exact leakage-safe empirical mean over Array train samples",
        "served_val_cpg_prior": "full-fit NTv3-to-prior probe (OOD leg, no true value available)",
        "served_train_cpg_sigma": "exact leakage-safe empirical std (logit space) over Array train samples",
        "served_val_cpg_sigma": "full-fit NTv3-to-log-sigma probe (OOD leg, no true value available)",
        "oof_sigma_folds": 5,
        "oof_sigma_train_mse_log_sigma": oof_sigma_mse,
        "sigma_full_fit": sigma_full_metrics,
        "sigma_folds": sigma_fold_metrics,
        "sigma_probe": str(sigma_checkpoint),
        "oof_folds": 5,
        "oof_train_mse_prior_logit": oof_mse,
        "full_fit": full_metrics,
        "folds": fold_metrics,
        "no_ntv3_inference": True,
        "no_genomewide_tcga_methylation_supervision": True,
        "locus_features": str(locus_path),
        "full_fit_probe": str(checkpoint),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    done.write_text("ok\n")
    return manifest


def _infer_fullfit_prior(state: dict, embeddings: np.ndarray, rows: np.ndarray, device: str, batch: int = 8192) -> np.ndarray:
    model = PriorEnsemble(tuple(state["seeds"]), dim=int(state["dim"]), dropout=float(state["dropout"]))
    model.load_state_dict(state["model_state"], strict=True); model.to(device).eval()
    xm = torch.from_numpy(np.asarray(state["x_mean"], np.float32)).to(device)
    xs = torch.from_numpy(np.asarray(state["x_std"], np.float32)).to(device)
    ym = float(state["y_mean"]); ys = float(state["y_std"]); amp = str(device).startswith("cuda")
    result = np.empty(len(rows), np.float32)
    with torch.inference_mode():
        for start in range(0, len(rows), batch):
            stop = min(start + batch, len(rows)); local = rows[start:stop]
            x = torch.from_numpy(np.asarray(embeddings[local], np.float32)).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                z = model((x - xm) / xs)
            logits = z.float().cpu().numpy() * ys + ym
            result[start:stop] = np.clip(1 / (1 + np.exp(-np.clip(logits, -35, 35))), 1e-4, 1 - 1e-4)
    return result


def _infer_fullfit_sigma(state: dict, embeddings: np.ndarray, rows: np.ndarray, device: str, batch: int = 8192) -> np.ndarray:
    """Same probe-inference machinery as _infer_fullfit_prior, but the target
    is log_sigma, not a logit-mean, so the output is exponentiated rather than
    passed through a sigmoid."""
    model = PriorEnsemble(tuple(state["seeds"]), dim=int(state["dim"]), dropout=float(state["dropout"]))
    model.load_state_dict(state["model_state"], strict=True); model.to(device).eval()
    xm = torch.from_numpy(np.asarray(state["x_mean"], np.float32)).to(device)
    xs = torch.from_numpy(np.asarray(state["x_std"], np.float32)).to(device)
    ym = float(state["y_mean"]); ys = float(state["y_std"]); amp = str(device).startswith("cuda")
    result = np.empty(len(rows), np.float32)
    with torch.inference_mode():
        for start in range(0, len(rows), batch):
            stop = min(start + batch, len(rows)); local = rows[start:stop]
            x = torch.from_numpy(np.asarray(embeddings[local], np.float32)).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                z = model((x - xm) / xs)
            log_sigma = z.float().cpu().numpy() * ys + ym
            result[start:stop] = np.exp(np.clip(log_sigma, -20, 20))
    return result


def build_feature_cache(protocol: Table5Protocol, atlas: Path, prior_root: Path, output: Path, *, device: str) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True); done = output / ".done"
    ids_path = output / "cpg_idx.npy"; emb_path = output / "embeddings.f16.npy"; prior_path = output / "prior.npy"
    sigma_path = output / "sigma.npy"
    if done.is_file() and all(x.is_file() for x in (ids_path, emb_path, prior_path, sigma_path)):
        return json.loads((output / "manifest.json").read_text())
    required = protocol.unique_required_cpgs(); np.save(ids_path, required)
    embeddings = _atlas_embeddings(atlas, required, emb_path)
    array_features = pd.read_parquet(
        prior_root / "locus_features.parquet", columns=["cpg_idx", "pred_ntv3_prior", "served_sigma"]
    )
    base_ids = array_features.cpg_idx.to_numpy(np.int64)
    base_prior = array_features.pred_ntv3_prior.to_numpy(np.float32)
    base_sigma = array_features.served_sigma.to_numpy(np.float32)
    p = np.argsort(base_ids); base_ids = base_ids[p]; base_prior = base_prior[p]; base_sigma = base_sigma[p]
    search = np.searchsorted(base_ids, required); clipped = np.minimum(search, len(base_ids) - 1)
    in_base = (search < len(base_ids)) & (base_ids[clipped] == required)
    prior = np.lib.format.open_memmap(prior_path, mode="w+", dtype=np.float32, shape=(len(required),))
    sigma = np.lib.format.open_memmap(sigma_path, mode="w+", dtype=np.float32, shape=(len(required),))
    prior[in_base] = base_prior[search[in_base]]
    sigma[in_base] = base_sigma[search[in_base]]
    missing = np.flatnonzero(~in_base).astype(np.int64)
    state = torch.load(prior_root / "full_fit_prior_probe.pt", map_location="cpu", weights_only=False)
    sigma_state = torch.load(prior_root / "full_fit_sigma_probe.pt", map_location="cpu", weights_only=False)
    if len(missing):
        prior[missing] = _infer_fullfit_prior(state, embeddings, missing, device)
        sigma[missing] = _infer_fullfit_sigma(sigma_state, embeddings, missing, device)
    prior.flush(); del prior
    sigma.flush(); del sigma
    manifest = {
        "status": "built", "protocol": TABLE5_PROTOCOL_NAME,
        "rows": int(len(required)), "array_prior_rows_oof_or_heldout_fullfit": int(in_base.sum()),
        "auxiliary_prior_rows_fullfit_probe": int(len(missing)),
        "atlas": str(atlas), "table5_prior": str(prior_root), "no_ntv3_inference": True,
        "sigma_cache": str(sigma_path),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n"); done.write_text("ok\n")
    return manifest


def build_eval_adapter(root: Path, final_config: Path, protocol: Table5Protocol, feature_root: Path, prior_root: Path, output: Path) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True); manifests = output / "manifests"; manifests.mkdir(exist_ok=True)
    train_s, val_s = protocol.array_train_sample_idx, protocol.array_val_sample_idx
    train_c, val_c = protocol.array_train_cpg_idx, protocol.array_val_cpg_idx
    with h5py.File(root / "rna" / "tcga_rna_official_full.h5", "r") as h:
        rna_ids = np.asarray(h["sample_idx"][...], np.int64); labels = _decode(h["tissue_name"][...])
    all_s = np.concatenate([train_s, val_s]); cancer = labels[positions(rna_ids, all_s, "RNA eval sample_idx")]
    pd.DataFrame({
        "sample_idx": all_s, "cancer_type": cancer,
        "split": np.concatenate([np.full(len(train_s), "train", object), np.full(len(val_s), "validation", object)]),
    }).to_parquet(manifests / "sample_metadata.parquet", index=False)
    all_c = np.concatenate([train_c, val_c])
    pd.DataFrame({
        "cpg_idx": all_c,
        "split": np.concatenate([np.full(len(train_c), "train", object), np.full(len(val_c), "validation", object)]),
    }).to_parquet(manifests / "cpg_splits.parquet", index=False)
    cache_ids = np.load(feature_root / "cpg_idx.npy", mmap_mode="r"); cache_emb = np.load(feature_root / "embeddings.f16.npy", mmap_mode="r")
    rows = positions(cache_ids, all_c, "Table-5 final feature cache")
    emb_h5 = output / "array_chr1_embeddings.h5"; tmp = emb_h5.with_suffix(".tmp.h5")
    with h5py.File(tmp, "w") as h:
        h.create_dataset("cpg_idx", data=all_c, dtype="i8")
        ds = h.create_dataset("embeddings", shape=(len(all_c), 1536), dtype="f2", chunks=(min(1024, len(all_c)), 1536))
        for start in range(0, len(all_c), 1024):
            stop = min(start + 1024, len(all_c)); ds[start:stop] = np.asarray(cache_emb[rows[start:stop]], np.float16)
    os.replace(tmp, emb_h5)
    raw = yaml.safe_load(final_config.read_text()); raw["run_name"] = "methylprophet-table5-tcga-chr1-eval"; raw["output_dir"] = str(output / "runner")
    raw["data"] = {
        "rna": {"path": str(root / "rna" / "tcga_rna_official_full.h5"), "values_key": "X", "row_ids_key": "sample_idx", "col_ids_key": "gene_ids"},
        "methylation": {"path": str(root / "methylation" / "tcga_array_official_full.h5"), "values_key": "beta", "row_ids_key": "sample_idx", "col_ids_key": "cpg_idx"},
        "locus_embeddings": {"path": str(emb_h5), "values_key": "embeddings", "row_ids_key": "cpg_idx"},
        "locus_features": {"path": str(prior_root / "locus_features.parquet"), "id_column": "cpg_idx"},
        "sample_metadata": {"path": str(manifests / "sample_metadata.parquet"), "id_column": "sample_idx"},
        "cpg_splits": {"path": str(manifests / "cpg_splits.parquet"), "id_column": "cpg_idx"},
        "rna_control": "real", "rna_transform": "zscore", "standardize_rna": True,
        "clip_beta_epsilon": 1e-4, "allow_partial_overlap": False,
    }
    raw["tracking"] = {"backend": "none", "project": "MethylPredictor", "mode": "disabled", "log_checkpoint": False}
    eval_config = output / "eval_config.yaml"; eval_config.write_text(yaml.safe_dump(raw, sort_keys=False))
    return {"eval_config": str(eval_config), "embeddings": str(emb_h5)}


def _update_protocol_audit(protocol_root: Path, audit: dict[str, object]) -> None:
    path = protocol_root / "protocol.json"; raw = json.loads(path.read_text())
    raw["finite_pair_audit"] = audit; raw["status"] = "exact_table5_ready"
    path.write_text(json.dumps(raw, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--canonical-root", required=True)
    p.add_argument("--atlas", required=True)
    p.add_argument("--hg38-fasta", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mp-eval", help="optional released MethylProphet Table-5 eval directory for direct ID verification")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    root = Path(args.canonical_root); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    atlas = Path(args.atlas); fasta = Path(args.hg38_fasta); cfg = Path(args.config)
    for path in [root, atlas, fasta, cfg]:
        if not path.exists(): raise FileNotFoundError(path)

    print("=== MethylProphet Table-5 exact TCGA chr1 preparation ===", flush=True)
    protocol_root = out / "table5_protocol"
    protocol = build_table5_protocol(root, fasta, protocol_root, Path(args.mp_eval) if args.mp_eval else None)

    compact = out / "methylation"
    array_order_samples = np.concatenate([protocol.array_train_sample_idx, protocol.array_val_sample_idx])
    array_order_cpgs = np.concatenate([protocol.array_train_cpg_idx, protocol.array_val_cpg_idx])
    with h5py.File(root / "methylation" / "epic_full.h5", "r") as h:
        epic_samples = np.asarray(h["sample_idx"][...], np.int64)
    array_path = compact / "array_table5_chr1.h5"; epic_path = compact / "epic_table5_chr1.h5"
    array_result = build_exact_compact_cache(root, "array", array_path, sample_ids=array_order_samples, cpg_ids=array_order_cpgs)
    epic_result = build_exact_compact_cache(root, "epic", epic_path, sample_ids=epic_samples, cpg_ids=protocol.epic_train_cpg_idx)
    audit = audit_finite_pairs(root, protocol, array_path, epic_path); _update_protocol_audit(protocol_root, audit)
    protocol = Table5Protocol.load(protocol_root)  # now requires exact audit status
    print("[Table5] exact finite-pair audit PASS", json.dumps(audit, indent=2), flush=True)

    prior_root = out / "table5_genomic_prior"
    prior_result = build_table5_prior(protocol, atlas, array_path, prior_root, device=args.device)
    feature_result = build_feature_cache(protocol, atlas, prior_root, out / "features", device=args.device)

    # Preserve the RNA normalization contract used during architecture selection:
    # z-score statistics are fitted on the exact Array training split only.
    # EPIC/WGBS RNA rows are still used as training inputs, transformed by these
    # frozen statistics; no held-out Array sample enters normalization.
    rna_result = prepare_rna_cache(
        root, out / "rna", train_sample_idx=protocol.array_train_sample_idx
    )
    eval_result = build_eval_adapter(root, cfg, protocol, out / "features", prior_root, out / "evaluation_adapter")
    manifest = {
        "status": "ready_exact_table5",
        "protocol": TABLE5_PROTOCOL_NAME,
        "protocol_root": str(protocol_root),
        "finite_pair_audit": audit,
        "prior": prior_result,
        "features": feature_result,
        "rna": rna_result,
        "array_cache": array_result,
        "epic_cache": epic_result,
        "evaluation_adapter": eval_result,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (out / ".done").write_text("ok\n")
    print(json.dumps(manifest, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
