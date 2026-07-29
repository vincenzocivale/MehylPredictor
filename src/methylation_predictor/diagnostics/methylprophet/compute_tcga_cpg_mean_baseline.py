"""Exact sufficient-statistics evaluation of the TCGA per-CpG mean baseline."""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max_cpg_idx", type=int, default=30_000_000)
    p.add_argument("--max_sample_idx", type=int, default=20_000)
    p.add_argument("--chromosome", default="chr1")
    p.add_argument("--mean_source", choices=["train", "test"], default="train")
    p.add_argument("--cpg-manifest", help="Optional parquet with cpg_idx; restrict to exactly evaluated loci.")
    args = p.parse_args()
    selected_cpg = None
    if args.cpg_manifest:
        selected_cpg = set(pd.read_parquet(args.cpg_manifest, columns=["cpg_idx"])["cpg_idx"].astype(int))
    root = Path(args.data_root) / "processed/241231-tcga_array-index_files-ind_cancer-non_nan/me_cpg_bg"
    train_files = sorted(glob.glob(str(root / "train_cpg-train_sample.parquet/*.parquet")))
    test_files = sorted(glob.glob(str(root / "train_cpg-val_sample.parquet/*.parquet")))
    if not train_files or not test_files:
        raise FileNotFoundError("Expected TCGA train_cpg-train_sample and train_cpg-val_sample parquet shards")
    if args.mean_source == "test":
        train_files = test_files

    sums = np.zeros(args.max_cpg_idx, dtype=np.float64)
    counts = np.zeros(args.max_cpg_idx, dtype=np.int64)
    for i, f in enumerate(train_files, 1):
        d = pd.read_parquet(f, columns=["cpg_idx", "cpg_chr_pos", "methylation"]).dropna()
        d = d[d.cpg_chr_pos.str.startswith(args.chromosome + "_")]
        if selected_cpg is not None:
            d = d[d.cpg_idx.isin(selected_cpg)]
        np.add.at(sums, d.cpg_idx.to_numpy(), d.methylation.to_numpy())
        np.add.at(counts, d.cpg_idx.to_numpy(), 1)
        if i % 200 == 0:
            print(f"training shards: {i}/{len(train_files)}", flush=True)
    means = sums / np.maximum(counts, 1)

    n = np.zeros(args.max_sample_idx, dtype=np.int64)
    sx = np.zeros(args.max_sample_idx); sy = np.zeros(args.max_sample_idx)
    sxx = np.zeros(args.max_sample_idx); syy = np.zeros(args.max_sample_idx); sxy = np.zeros(args.max_sample_idx)
    total_n = 0; sse = 0.0; sae = 0.0
    for i, f in enumerate(test_files, 1):
        d = pd.read_parquet(f, columns=["cpg_idx", "cpg_chr_pos", "sample_idx", "methylation"]).dropna()
        d = d[d.cpg_chr_pos.str.startswith(args.chromosome + "_")]
        if selected_cpg is not None:
            d = d[d.cpg_idx.isin(selected_cpg)]
        x = means[d.cpg_idx.to_numpy()]; y = d.methylation.to_numpy(); s = d.sample_idx.to_numpy()
        valid = counts[d.cpg_idx.to_numpy()] > 0
        x, y, s = x[valid], y[valid], s[valid]
        np.add.at(n, s, 1); np.add.at(sx, s, x); np.add.at(sy, s, y)
        np.add.at(sxx, s, x*x); np.add.at(syy, s, y*y); np.add.at(sxy, s, x*y)
        err = x-y; total_n += len(err); sse += float(np.dot(err, err)); sae += float(np.abs(err).sum())
        if i % 200 == 0:
            print(f"test shards: {i}/{len(test_files)}", flush=True)
    den = np.sqrt((n*sxx-sx*sx)*(n*syy-sy*sy))
    pcc = (n*sxy-sx*sy) / den
    pcc = pcc[np.isfinite(pcc) & (n > 1)]
    result = {
        "baseline": f"per_cpg_{args.mean_source}_sample_mean",
        "chromosome": args.chromosome,
        "split": "train_cpg-val_sample",
        "pairs": int(total_n),
        "mas_pcc": None,
        "mac_pcc": float(np.median(pcc)),
        "mse": sse / total_n,
        "mae": sae / total_n,
        "samples_with_pcc": int(len(pcc)),
        "cpg_manifest": args.cpg_manifest,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
