#!/usr/bin/env python3
"""Build a SurvPath-style gene/pathway membership in canonical TCGA gene order.

The source is pinned to the public SurvPath commit below and combines Reactome
and MSigDB Hallmark signatures. The output contains only a sparse structural
edge list; all pathway weights are learned by GenePathwayEncoder.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

from methylation_predictor.tcga_canonical.bundle import TCGACanonicalBundle
from methylation_predictor.tcga_canonical.config import resolve_bundle_root

SURVPATH_COMMIT = "3f73ddd6705ec67d643020c5bb04fb13f9f382cc"
SURVPATH_PATH = "datasets_csv/metadata/combine_signatures.csv"
SURVPATH_URL = f"https://raw.githubusercontent.com/mahmoodlab/SurvPath/{SURVPATH_COMMIT}/{SURVPATH_PATH}"
DEFAULT_OUTPUT = "resources/pathways/survpath_combine_tcga.npz"


def canonical_symbols(gene_ids: np.ndarray) -> list[str]:
    out = []
    for raw in gene_ids:
        symbol = str(raw).split(";", 1)[0].strip()
        if not symbol:
            raise ValueError(f"cannot extract symbol from canonical gene id {raw!r}")
        out.append(symbol)
    return out


def load_signatures(path: str | None):
    if path:
        raw = Path(path).read_bytes(); source = str(Path(path).resolve())
    else:
        with urllib.request.urlopen(SURVPATH_URL, timeout=120) as response:
            raw = response.read()
        source = SURVPATH_URL
    return pd.read_csv(io.BytesIO(raw)), source, hashlib.sha256(raw).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical-root", default=None)
    ap.add_argument("--signatures", help="optional local combine_signatures.csv")
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    root = resolve_bundle_root(args.canonical_root)
    bundle = TCGACanonicalBundle.from_root(root)
    try:
        symbols = canonical_symbols(bundle.rna.gene_ids)
    finally:
        bundle.close()

    by_symbol: dict[str, list[int]] = {}
    for idx, symbol in enumerate(symbols):
        by_symbol.setdefault(symbol, []).append(idx)

    signatures, source, source_sha = load_signatures(args.signatures)
    if signatures.shape[1] < 300:
        raise RuntimeError(f"expected ~331 combined pathways, got {signatures.shape[1]}")
    candidate_names = [str(c) for c in signatures.columns]
    names: list[str] = []
    dropped_pathways: list[str] = []
    gene_edges, pathway_edges, matched_per_pathway = [], [], []
    source_memberships = matched_memberships = 0
    for candidate in candidate_names:
        genes = list(dict.fromkeys(str(x).strip() for x in signatures[candidate].dropna() if str(x).strip()))
        source_memberships += len(genes)
        local = sorted({idx for gene in genes for idx in by_symbol.get(gene, [])})
        if not local:
            # A handful of SurvPath signatures (mostly single/few-gene Reactome
            # reactions) name genes absent from this bundle's ~25k-gene RNA
            # panel entirely; there's no local match to fall back to, so we
            # drop the pathway rather than emitting an empty membership column.
            dropped_pathways.append(candidate)
            continue
        p_idx = len(names)
        names.append(candidate)
        matched_per_pathway.append(len(local)); matched_memberships += len(local)
        gene_edges.extend(local); pathway_edges.extend([p_idx] * len(local))
    if dropped_pathways:
        print(
            f"[gene-pathway] dropping {len(dropped_pathways)} pathway(s) with zero genes "
            f"in canonical RNA axis: {dropped_pathways}"
        )
    if not names:
        raise RuntimeError("no pathway retained any genes in canonical RNA axis")

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    width = max(map(len, names))
    np.savez_compressed(
        out,
        n_genes=np.asarray([len(symbols)], dtype=np.int64),
        n_pathways=np.asarray([len(names)], dtype=np.int64),
        gene_idx=np.asarray(gene_edges, dtype=np.int64),
        pathway_idx=np.asarray(pathway_edges, dtype=np.int64),
        pathway_names=np.asarray(names, dtype=f"<U{width}"),
        matched_genes_per_pathway=np.asarray(matched_per_pathway, dtype=np.int64),
    )
    meta = {
        "source": source,
        "source_commit": SURVPATH_COMMIT,
        "source_path": SURVPATH_PATH,
        "source_sha256": source_sha,
        "canonical_root": str(root),
        "n_canonical_genes": len(symbols),
        "n_source_pathways": len(candidate_names),
        "n_pathways": len(names),
        "dropped_pathways": dropped_pathways,
        "source_memberships": source_memberships,
        "matched_memberships": matched_memberships,
        "membership_coverage": matched_memberships / max(source_memberships, 1),
        "min_matched_genes_per_pathway": int(min(matched_per_pathway)),
        "median_matched_genes_per_pathway": float(np.median(matched_per_pathway)),
        "max_matched_genes_per_pathway": int(max(matched_per_pathway)),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2)); print(f"[gene-pathway] wrote {out}")


if __name__ == "__main__":
    main()
