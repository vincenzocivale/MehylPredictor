#!/usr/bin/env python3
"""Stratify Table-5 Array evaluation skill by CpG genomic context (island /
shore / shelf / open sea) and by ground-truth inter-sample variability.

This is a diagnostic, not part of the fail-closed benchmark pipeline: it
reuses the frozen final.pt checkpoint and the exact Table-5 evaluation views,
but breaks the aggregate mas_pcc/mse down per CpG so we can see *where* the
model over- or under-performs, rather than only the single scalar per view.

Important caveat: MethylProphet only publishes aggregate metrics per view (no
per-CpG predictions), so we cannot compute a genuine per-CpG OURS-vs-MP delta.
What we *can* do:
  1) show where OUR model's skill concentrates (by context, by variability);
  2) show whether the train-CpG pool and the val-CpG pool -- which get
     compared against MethylProphet's aggregate numbers in different views --
     differ systematically in context/variability composition, which is a
     legitimate, evidence-based explanation for why the OURS-vs-MP delta
     differs across views.
"""
from __future__ import annotations

import argparse
import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow  # noqa: F401
import pyarrow.parquet as pq
import torch

from methylation_predictor.benchmark.table5.trainer import ArrayMomentMetrics, Table5Trainer

UCSC_CPG_ISLAND_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz"
SHORE_BP = 2_000
SHELF_BP = 4_000


def fetch_cpg_islands(cache_path: Path, chrom: str = "chr1") -> np.ndarray:
    """Return sorted (start, end) island intervals for one chromosome."""
    if not cache_path.is_file():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(UCSC_CPG_ISLAND_URL, timeout=30) as resp:
            cache_path.write_bytes(resp.read())
    rows = []
    with gzip.open(cache_path, "rt") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            # bin, chrom, chromStart, chromEnd, name, ...
            if parts[1] != chrom:
                continue
            rows.append((int(parts[2]), int(parts[3])))
    islands = np.array(sorted(rows), dtype=np.int64)
    return islands


def classify_context(pos: np.ndarray, islands: np.ndarray) -> np.ndarray:
    """Vectorized island/shore/shelf/opensea classification.

    shore = within SHORE_BP of an island but not inside one; shelf = beyond
    that and within SHELF_BP; open sea = everything else. Matches the
    standard Illumina/UCSC convention.
    """
    starts, ends = islands[:, 0], islands[:, 1]
    # distance to nearest island edge (0 if inside an island)
    order = np.argsort(starts)
    starts_s, ends_s = starts[order], ends[order]
    idx = np.searchsorted(starts_s, pos, side="right") - 1
    idx = np.clip(idx, 0, len(starts_s) - 1)
    left_end = ends_s[idx]
    left_start = starts_s[idx]
    next_idx = np.clip(idx + 1, 0, len(starts_s) - 1)
    right_start = starts_s[next_idx]

    inside = (pos >= left_start) & (pos <= left_end)
    dist_left = np.where(pos > left_end, pos - left_end, 0)
    dist_right = np.where(pos < right_start, right_start - pos, 0)
    dist = np.minimum(np.where(pos > left_end, dist_left, np.iinfo(np.int64).max),
                       np.where(pos < right_start, dist_right, np.iinfo(np.int64).max))
    dist = np.where(inside, 0, dist)

    context = np.full(len(pos), "open_sea", dtype=object)
    context[np.isfinite(dist.astype(float)) & (dist <= SHELF_BP)] = "shelf"
    context[dist <= SHORE_BP] = "shore"
    context[inside] = "island"
    return context


@torch.no_grad()
def evaluate_per_cpg(trainer: Table5Trainer, sample_ids: np.ndarray, cpg_ids: np.ndarray) -> dict[str, np.ndarray]:
    trainer.model.eval()
    compact = trainer.compact["array"]
    rows = compact.rows_of_samples(sample_ids)
    metrics = ArrayMomentMetrics(len(sample_ids), len(cpg_ids))
    sc, cc = 128, 2048
    for s0 in range(0, len(sample_ids), sc):
        s1 = min(s0 + sc, len(sample_ids)); local_ids = sample_ids[s0:s1]; local_rows = rows[s0:s1]
        rna = torch.from_numpy(trainer.rna.rows(local_ids)).to(trainer.device)
        for c0 in range(0, len(cpg_ids), cc):
            c1 = min(c0 + cc, len(cpg_ids)); local_c = cpg_ids[c0:c1]
            emb_np, prior_np = trainer.features.get(local_c)
            emb = torch.from_numpy(emb_np).to(trainer.device)
            prior = torch.from_numpy(prior_np).to(trainer.device)
            with trainer._autocast():
                pred = trainer.model(rna, emb, prior)["beta"]
            target = compact.block(local_rows, local_c)
            metrics.add(s0, c0, target, pred.float().cpu().numpy(), prior_np)
    return metrics.finalize_per_cpg()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--canonical-root", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--derived-root", required=True)
    p.add_argument("--checkpoint-dir", required=True, help="run dir containing final.pt / .train_done")
    p.add_argument("--epochs", type=int, required=True, help="must match the epoch budget the checkpoint was trained with")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--island-cache", default=None)
    args = p.parse_args()

    root = args.derived_root
    trainer = Table5Trainer(
        canonical_root=args.canonical_root,
        config_path=args.config,
        protocol_root=f"{root}/table5_protocol",
        feature_cache=f"{root}/features",
        rna_cache=f"{root}/rna",
        array_cache=f"{root}/methylation/array_table5_chr1.h5",
        epic_cache=f"{root}/methylation/epic_table5_chr1.h5",
        output_dir=args.checkpoint_dir,
        epochs=args.epochs,
        seed=args.seed,
    )
    trainer.train()  # no-op weight load: .train_done + final.pt already exist

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # --- CpG genomic context -------------------------------------------------
    registry = pq.read_table(
        Path(args.canonical_root) / "cpg" / "registries" / "array_cpg_map.parquet",
        columns=["cpg_idx", "chr", "pos"],
    ).to_pandas()
    registry = registry[registry["chr"] == "chr1"].set_index("cpg_idx")

    island_cache = Path(args.island_cache) if args.island_cache else out / "cpgIslandExt.txt.gz"
    islands = fetch_cpg_islands(island_cache, "chr1")

    def context_for(cpg_ids: np.ndarray) -> pd.Series:
        pos = registry.loc[cpg_ids, "pos"].to_numpy()
        return pd.Series(classify_context(pos, islands), index=cpg_ids)

    protocol = trainer.protocol
    train_cpg_ctx = context_for(protocol.array_train_cpg_idx)
    val_cpg_ctx = context_for(protocol.array_val_cpg_idx)

    # --- composition of the two CpG pools (explains cross-view MP deltas) ---
    composition = pd.DataFrame({
        "train_cpg_pool": train_cpg_ctx.value_counts(normalize=True),
        "val_cpg_pool": val_cpg_ctx.value_counts(normalize=True),
    }).fillna(0.0).reindex(["island", "shore", "shelf", "open_sea"])
    composition.to_csv(out / "cpg_pool_context_composition.csv")
    print("=== CpG pool composition (train-CpG vs val-CpG), fraction ===")
    print(composition.round(3).to_string())

    # --- per-CpG evaluation for the three official views ---------------------
    views = protocol.evaluation_views()
    all_rows = []
    for view_name, (sample_ids, cpg_ids) in views.items():
        detail = evaluate_per_cpg(trainer, sample_ids, cpg_ids)
        ctx = (train_cpg_ctx if view_name == "train_cpg_x_val_sample" else val_cpg_ctx).loc[cpg_ids].to_numpy()
        df = pd.DataFrame({
            "view": view_name,
            "cpg_idx": cpg_ids,
            "context": ctx,
            "n": detail["n"],
            "pearson": detail["pearson"],
            "mse": detail["mse"],
            "prior_mse": detail["prior_mse"],
            "skill_vs_prior": detail["skill_vs_prior"],
            "true_var": detail["true_var"],
        })
        df = df[df["n"] > 1]
        df["variability_quartile"] = pd.qcut(df["true_var"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
        all_rows.append(df)

    full = pd.concat(all_rows, ignore_index=True)
    full.to_parquet(out / "per_cpg_detail.parquet")

    # --- stratified summaries -------------------------------------------------
    by_context = (
        full.groupby(["view", "context"])
        .agg(n_cpgs=("cpg_idx", "size"), mean_pearson=("pearson", "mean"),
             median_pearson=("pearson", "median"), mean_mse=("mse", "mean"),
             mean_skill_vs_prior=("skill_vs_prior", "mean"))
        .reset_index()
    )
    by_context.to_csv(out / "skill_by_genomic_context.csv", index=False)
    print("\n=== Skill by genomic context ===")
    print(by_context.round(4).to_string(index=False))

    by_variability = (
        full.groupby(["view", "variability_quartile"], observed=True)
        .agg(n_cpgs=("cpg_idx", "size"), mean_pearson=("pearson", "mean"),
             median_pearson=("pearson", "median"), mean_mse=("mse", "mean"),
             mean_skill_vs_prior=("skill_vs_prior", "mean"))
        .reset_index()
    )
    by_variability.to_csv(out / "skill_by_variability_quartile.csv", index=False)
    print("\n=== Skill by inter-sample variability quartile (of ground truth) ===")
    print(by_variability.round(4).to_string(index=False))

    by_both = (
        full.groupby(["view", "context", "variability_quartile"], observed=True)
        .agg(n_cpgs=("cpg_idx", "size"), mean_pearson=("pearson", "mean"), mean_mse=("mse", "mean"))
        .reset_index()
    )
    by_both.to_csv(out / "skill_by_context_and_variability.csv", index=False)

    (out / "summary.json").write_text(json.dumps({
        "cpg_pool_composition": composition.to_dict(),
        "by_context": by_context.to_dict(orient="records"),
        "by_variability": by_variability.astype({"variability_quartile": str}).to_dict(orient="records"),
    }, indent=2, default=float) + "\n")
    print(f"\nwrote detail + summaries to {out}")


if __name__ == "__main__":
    main()
