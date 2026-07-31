#!/usr/bin/env python3
"""Align an MSigDB Hallmark GMT to RNA IDs and create a matched random control."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from methylation_predictor.rna_branch.config import load_config
from methylation_predictor.rna_branch.data import MatrixStore


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_rna_id_map(gene_ids: np.ndarray) -> dict[str, str]:
    """Map unambiguous symbols/Ensembl IDs to the exact RNA column IDs."""
    candidates: dict[str, set[str]] = defaultdict(set)
    for raw in gene_ids.astype(str):
        for value in raw.split(";"):
            value = value.strip()
            if not value:
                continue
            candidates[value.upper()].add(raw)
            if value.upper().startswith("ENSG"):
                candidates[value.split(".", 1)[0].upper()].add(raw)
    return {key: next(iter(values)) for key, values in candidates.items() if len(values) == 1}


def _read_gmt(path: Path) -> dict[str, list[str]]:
    modules: dict[str, list[str]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 3:
            raise ValueError(f"GMT row {line_number} has fewer than three columns")
        module_id = fields[0].strip()
        if not module_id or module_id in modules:
            raise ValueError(f"invalid or duplicate module_id on GMT row {line_number}: {module_id!r}")
        modules[module_id] = [gene.strip().upper() for gene in fields[2:] if gene.strip()]
    return modules


def _write_membership(path: Path, modules: dict[str, list[str]]) -> None:
    rows = [
        {"module_id": module_id, "gene_id": gene_id, "weight": 1.0}
        for module_id, genes in modules.items()
        for gene_id in genes
    ]
    pd.DataFrame(rows, columns=["module_id", "gene_id", "weight"]).to_csv(path, sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--hallmark-gmt", required=True, help="official MSigDB Hallmark GMT")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-genes", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260730)
    args = parser.parse_args()

    config = load_config(args.config)
    store = MatrixStore(config.data.rna)
    try:
        if store.col_ids is None:
            raise ValueError("RNA matrix requires col_ids_key")
        rna_ids = store.col_ids.astype(str)
    finally:
        store.close()

    gmt_path = Path(args.hallmark_gmt)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rna_id_by_alias = _unique_rna_id_map(rna_ids)
    source = _read_gmt(gmt_path)
    aligned: dict[str, list[str]] = {}
    dropped: dict[str, int] = {}
    for module_id, genes in source.items():
        matched = sorted({rna_id_by_alias[gene] for gene in genes if gene in rna_id_by_alias})
        if len(matched) >= args.min_genes:
            aligned[module_id] = matched
        else:
            dropped[module_id] = len(matched)
    if not aligned:
        raise ValueError("no Hallmark modules met --min-genes after RNA alignment")

    real_path = output_dir / "hallmark_aligned.tsv"
    random_path = output_dir / "hallmark_random_matched.tsv"
    _write_membership(real_path, aligned)
    rng = np.random.default_rng(args.random_seed)
    random_modules = {
        module_id: sorted(rng.choice(rna_ids, size=len(genes), replace=False).astype(str).tolist())
        for module_id, genes in aligned.items()
    }
    _write_membership(random_path, random_modules)
    report = {
        "source_gmt": str(gmt_path),
        "source_gmt_sha256": _sha256(gmt_path),
        "rna_genes": len(rna_ids),
        "source_modules": len(source),
        "retained_modules": len(aligned),
        "dropped_modules": dropped,
        "retained_memberships": int(sum(map(len, aligned.values()))),
        "random_seed": args.random_seed,
        "min_genes": args.min_genes,
        "outputs": {"hallmark": str(real_path), "random_matched": str(random_path)},
    }
    (output_dir / "hallmark_alignment_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
