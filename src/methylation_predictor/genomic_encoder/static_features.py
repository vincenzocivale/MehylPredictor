#!/usr/bin/env python3
"""Build the annotated, one-row-per-CpG table for Experiment 2 (hg38).

Processes one chromosome per invocation (inferred from --prior/--cpg-map,
not hardcoded) -- run once per chromosome to cover the genome."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fasta(path: Path) -> str:
    with gzip.open(path, "rt") as f:
        return "".join(line.strip() for line in f if not line.startswith(">")).upper()


def cgi_annotations(pos: np.ndarray, path: Path, chromosome: str) -> tuple[np.ndarray, np.ndarray]:
    # UCSC positions are 0-based half-open; CpG coordinates are 1-based.
    table = pd.read_csv(path, sep="\t", header=None, compression="gzip", usecols=[1, 2, 3],
                        names=["chromosome", "start", "end"])
    table = table[table.chromosome == chromosome].sort_values("start")
    starts, ends = table.start.to_numpy(), table.end.to_numpy()
    ix = np.searchsorted(starts, pos - 1, side="right") - 1
    left_distance = np.where(ix >= 0, np.maximum(0, (pos - 1) - ends[np.maximum(ix, 0)]), np.inf)
    next_ix = np.clip(ix + 1, 0, len(starts) - 1)
    right_distance = np.maximum(0, starts[next_ix] - (pos - 1))
    inside = (ix >= 0) & ((pos - 1) < ends[np.maximum(ix, 0)])
    distance = np.where(inside, 0, np.minimum(left_distance, right_distance))
    category = np.where(inside, "island", np.where(distance <= 2_000, "shore", np.where(distance <= 4_000, "shelf", "open_sea")))
    return category, distance.astype(np.int64)


def genes(path: Path, chromosome: str) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            values = line.rstrip("\n").split("\t")
            if values[0] == chromosome and values[2] == "gene":
                rows.append((int(values[3]), int(values[4]), values[6]))
    result = pd.DataFrame(rows, columns=["start", "end", "strand"]).sort_values("start")
    if result.empty:
        raise ValueError(f"No {chromosome} gene rows in GENCODE annotation")
    return result


def gene_annotations(pos: np.ndarray, gene: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    starts, ends, strands = gene.start.to_numpy(), gene.end.to_numpy(), gene.strand.to_numpy()
    tss = np.where(strands == "+", starts, ends)
    distance_tss = np.min(np.abs(pos[:, None] - tss[None, :]), axis=1)
    # 6.7k x ~6k is safe, and avoids ambiguous interval-tree dependencies.
    body = ((pos[:, None] >= starts[None, :]) & (pos[:, None] <= ends[None, :])).any(axis=1)
    promoter = distance_tss <= 2_000
    region = np.where(promoter, "promoter", np.where(body, "gene_body", "intergenic"))
    return region, distance_tss.astype(np.int64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior", required=True, help="Training-only per-CpG prior parquet (cpg_idx, mean_train)")
    p.add_argument("--cpg-map", required=True, help="Released cpg_chr_pos_df.parquet; row index is cpg_idx")
    p.add_argument("--fasta", required=True, help="UCSC hg38 single-chromosome FASTA matching --prior's chromosome")
    p.add_argument("--cpg-islands", required=True, help="UCSC hg38 cpgIslandExt.txt.gz")
    p.add_argument("--gencode-gtf", required=True, help="GENCODE v41 annotation GTF")
    p.add_argument("--output", required=True)
    p.add_argument("--within-stats", help="Optional group summary parquet; used only for error stratification")
    p.add_argument("--window-radius", type=int, default=100)
    args = p.parse_args()
    prior, mapping = pd.read_parquet(args.prior), pd.read_parquet(args.cpg_map)
    if prior.cpg_idx.duplicated().any():
        raise ValueError("Prior input contains duplicate cpg_idx")
    loci = mapping.iloc[prior.cpg_idx.to_numpy()].reset_index(names="cpg_idx")
    if not np.array_equal(loci.cpg_idx.to_numpy(), prior.cpg_idx.to_numpy()):
        raise ValueError("cpg_idx is not the row index of the released mapping")
    if loci.chr.nunique() != 1:
        raise ValueError("This builder processes exactly one chromosome per invocation (found: "
                          f"{sorted(loci.chr.unique())}) -- run once per chromosome")
    chromosome = loci.chr.iloc[0]
    genome, pos = fasta(Path(args.fasta)), loci.pos.to_numpy(np.int64)
    if not np.all([genome[x - 1:x + 1] == "CG" for x in pos]):
        raise ValueError("Coordinates do not match hg38 CpG dinucleotides (expected 1-based positions)")
    r = args.window_radius
    windows = [genome[max(0, x - 1 - r):min(len(genome), x - 1 + r + 1)] for x in pos]
    gc = np.array([(s.count("G") + s.count("C")) / len(s) for s in windows])
    density = np.array([sum(s[i:i + 2] == "CG" for i in range(len(s) - 1)) / len(s) * 100 for s in windows])
    cgi, cgi_distance = cgi_annotations(pos, Path(args.cpg_islands), chromosome)
    region, tss_distance = gene_annotations(pos, genes(Path(args.gencode_gtf), chromosome))
    out = pd.DataFrame({"cpg_idx": prior.cpg_idx, "mean_train": prior.mean_train, "chromosome": loci.chr,
                        "position": pos, "sequence_201bp": windows, "gc_content": gc, "cpg_density_per_100bp": density,
                        "cgi_category": cgi, "distance_to_cgi_bp": cgi_distance, "tss_distance": tss_distance,
                        "genomic_region": region})
    if "split" in prior.columns:
        out["split"] = prior.split.to_numpy()
    if args.within_stats:
        stats = pd.read_parquet(args.within_stats, columns=["cpg_idx", "n", "gt_within_ss"])
        stats["within_cpg_variance"] = stats.gt_within_ss / stats.n
        out = out.merge(stats[["cpg_idx", "within_cpg_variance"]], on="cpg_idx", how="left", validate="one_to_one")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); out.to_parquet(output, index=False)
    manifest = {"genome_build": "GRCh38/hg38", "coordinate_convention": "1-based CpG position; validated against hg38 CG",
                "window": f"{2 * r + 1} bp centred on CpG C", "n_cpg": len(out),
                "input_sha256": {name: sha256(Path(value)) for name, value in vars(args).items()
                                 if value and name in {"prior", "cpg_map", "fasta", "cpg_islands", "gencode_gtf", "within_stats"}},
                "output": str(output), "category_counts": {k: int(v) for k, v in out.cgi_category.value_counts().items()}}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
