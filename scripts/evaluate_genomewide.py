#!/usr/bin/env python3
"""Evaluate a frozen V1 checkpoint on all Array chromosomes, globally and per chromosome."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylation_predictor.config import load_config
from methylation_predictor.benchmark.table5.trainer import ArrayMomentMetrics, FinalFeatureCache
from methylation_predictor.full_suite.cache import RNACache
from methylation_predictor.models import VarianceNormalizedResidualModel
from methylation_predictor.tcga_canonical import TCGACanonicalBundle, load_protocol


def _chromosome_lookup(registry: Path, cpg_ids: np.ndarray) -> np.ndarray:
    frame = pd.read_parquet(registry)
    if "cpg_idx" not in frame:
        raise RuntimeError("registry must contain cpg_idx")
    chrom_col = next((c for c in ("chrom", "chr", "chromosome") if c in frame), None)
    if chrom_col is None:
        raise RuntimeError("registry must contain a chromosome column")
    values = frame.set_index("cpg_idx").loc[cpg_ids, chrom_col].astype(str).to_numpy()
    return np.asarray([x if x.startswith("chr") else f"chr{x}" for x in values], dtype=object)


class GenomewideEvaluator:
    def __init__(self, args):
        self.args = args
        self.cfg = load_config(args.config)
        if not self.cfg.model.variance_normalized_residual:
            raise RuntimeError("genome-wide evaluator requires canonical V1 variance_normalized_residual=true")
        self.bundle = TCGACanonicalBundle.from_root(args.canonical_root)
        self.protocol = load_protocol("array_genomewide", self.bundle, root=args.canonical_root)
        self.features = FinalFeatureCache(args.feature_cache)
        self.rna = RNACache(args.rna_cache)
        required = np.unique(np.concatenate([self.protocol.array_train_cpg_idx, self.protocol.array_val_cpg_idx]))
        self.features.index.positions_of(required)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("genome-wide evaluation requires CUDA")
        self.model = VarianceNormalizedResidualModel(25_017, 1536, self.cfg.model, epsilon=self.cfg.data.clip_beta_epsilon).to(self.device)
        state = torch.load(args.checkpoint, map_location=self.device, weights_only=False)
        if state.get("protocol") != "tcga_array_genomewide_v1":
            raise RuntimeError(
                "checkpoint protocol mismatch: expected tcga_array_genomewide_v1, "
                f"got {state.get('protocol')!r}"
            )
        if int(state.get("epoch", -1)) != int(state.get("epochs_planned", -2)):
            raise RuntimeError("checkpoint is not the completed fixed-budget model")
        self.model.load_state_dict(state["model_state"], strict=True)
        self.model.eval()
        all_ids = np.asarray(self.bundle.sources["array"].h5["cpg_idx"][...], np.int64)
        registry = Path(args.registry) if args.registry else Path(args.canonical_root) / "cpg" / "registries" / "array_cpg_map.parquet"
        self.chrom = dict(zip(all_ids.tolist(), _chromosome_lookup(registry, all_ids).tolist()))

    def close(self):
        self.bundle.close()

    @torch.no_grad()
    def evaluate(self, sample_ids: np.ndarray, cpg_ids: np.ndarray) -> dict[str, float | int]:
        source = self.bundle.sources["array"]
        rows = source.rows_of_samples(sample_ids)
        metrics = ArrayMomentMetrics(len(sample_ids), len(cpg_ids))
        for s0 in range(0, len(sample_ids), self.args.sample_chunk):
            s1 = min(s0 + self.args.sample_chunk, len(sample_ids)); local_s = sample_ids[s0:s1]
            rna = torch.from_numpy(self.rna.rows(local_s)).to(self.device)
            for c0 in range(0, len(cpg_ids), self.args.cpg_chunk):
                c1 = min(c0 + self.args.cpg_chunk, len(cpg_ids)); local_c = cpg_ids[c0:c1]
                emb_np, prior_np, sigma_np = self.features.get(local_c)
                emb = torch.from_numpy(emb_np).to(self.device)
                prior = torch.from_numpy(prior_np).to(self.device)
                sigma = torch.from_numpy(sigma_np).to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.cfg.training.amp):
                    pred = self.model(rna, emb, prior, sigma=sigma)["beta"]
                target = source.block(rows[s0:s1], local_c)
                metrics.add(s0, c0, target, pred.float().cpu().numpy(), prior_np)
        return metrics.finalize()

    @torch.no_grad()
    def evaluate_global_and_per_chromosome(
        self, sample_ids: np.ndarray, cpg_ids: np.ndarray
    ) -> tuple[dict[str, float | int], dict[str, dict[str, float | int]]]:
        """Evaluate a panel once while accumulating global and chromosome metrics.

        The previous implementation first evaluated the complete panel and then
        read/predicted it again for every chromosome.  Ordering CpGs by
        chromosome lets the exact same prediction blocks feed both accumulators;
        ArrayMomentMetrics is invariant to this column permutation.

        `MethylationSource.block()` always materializes full HDF5 rows (every
        CpG column) internally, however many columns are actually requested --
        cheap for one call, but the original chromosome-outer/cpg-inner loop
        called it once per (chromosome, sample-chunk, cpg-chunk) triple, so the
        same full rows were re-read from disk tens of times over. Sample-chunk
        is now the outer loop: each chunk's full-width row block is read once
        via `source.block(..., ordered_ids)` and every chromosome/cpg
        sub-chunk below slices it in memory.
        """
        source = self.bundle.sources["array"]
        rows = source.rows_of_samples(sample_ids)
        chrom_values = np.asarray([self.chrom[int(x)] for x in cpg_ids], dtype=object)
        chroms = sorted(
            np.unique(chrom_values),
            key=lambda x: int(x.removeprefix("chr")) if x.removeprefix("chr").isdigit() else 10_000,
        )
        groups = [(chrom, cpg_ids[chrom_values == chrom]) for chrom in chroms]
        groups = [(chrom, ids) for chrom, ids in groups if len(ids) >= 2]
        ordered_ids = np.concatenate([ids for _, ids in groups])
        global_metrics = ArrayMomentMetrics(len(sample_ids), len(ordered_ids))
        chromosome_metrics = {
            chrom: ArrayMomentMetrics(len(sample_ids), len(ids)) for chrom, ids in groups
        }

        group_offsets = []
        offset = 0
        for chrom, ids in groups:
            group_offsets.append((chrom, offset, len(ids)))
            offset += len(ids)

        for s0 in range(0, len(sample_ids), self.args.sample_chunk):
            s1 = min(s0 + self.args.sample_chunk, len(sample_ids))
            rna = torch.from_numpy(self.rna.rows(sample_ids[s0:s1])).to(self.device)
            target_full = source.block(rows[s0:s1], ordered_ids)
            for chrom, chrom_offset, chrom_len in group_offsets:
                local_metrics = chromosome_metrics[chrom]
                for c0 in range(0, chrom_len, self.args.cpg_chunk):
                    c1 = min(c0 + self.args.cpg_chunk, chrom_len)
                    local_c = ordered_ids[chrom_offset + c0 : chrom_offset + c1]
                    emb_np, prior_np, sigma_np = self.features.get(local_c)
                    emb = torch.from_numpy(emb_np).to(self.device)
                    prior = torch.from_numpy(prior_np).to(self.device)
                    sigma = torch.from_numpy(sigma_np).to(self.device)
                    with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=self.cfg.training.amp
                    ):
                        pred = self.model(rna, emb, prior, sigma=sigma)["beta"]
                    target = target_full[:, chrom_offset + c0 : chrom_offset + c1]
                    pred_np = pred.float().cpu().numpy()
                    local_metrics.add(s0, c0, target, pred_np, prior_np)
                    global_metrics.add(s0, chrom_offset + c0, target, pred_np, prior_np)

        return global_metrics.finalize(), {
            chrom: {"cpgs": int(chrom_len), **chromosome_metrics[chrom].finalize()}
            for chrom, _, chrom_len in group_offsets
        }

    def run(self) -> dict[str, object]:
        views = self.protocol.evaluation_views()
        result: dict[str, object] = {"protocol": "tcga_array_genomewide", "views": {}}
        rows = []
        for view_name, view in views.items():
            started = time.time()
            global_metrics, per_chrom = self.evaluate_global_and_per_chromosome(
                view.sample_idx, view.cpg_idx
            )
            print(
                f"[view:{view_name}] samples={len(view.sample_idx)} cpgs={len(view.cpg_idx)} "
                f"mas_pcc={global_metrics.get('mas_pcc')} mse={global_metrics.get('mse')} "
                f"seconds={time.time() - started:.1f}",
                flush=True,
            )
            for chrom, metrics in per_chrom.items():
                rows.append({"view": view_name, "chromosome": chrom, **metrics})
            result["views"][view_name] = {"global": global_metrics, "per_chromosome": per_chrom}
        out = Path(self.args.output); out.mkdir(parents=True, exist_ok=True)
        (out / "genomewide_evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
        pd.DataFrame(rows).to_csv(out / "per_chromosome.csv", index=False)
        if self.args.wandb:
            import wandb
            run = wandb.init(project="MethylPredictor", group="tcga-array-genomewide-evaluation", name=self.args.wandb_name or Path(self.args.checkpoint).parent.name, job_type="evaluation")
            flat = {}
            for view, payload in result["views"].items():
                for metric, value in payload["global"].items():
                    if isinstance(value, (int, float)):
                        flat[f"genomewide/{view}/{metric}"] = value
                for chrom, metrics in payload["per_chromosome"].items():
                    for metric in ("mse", "mae", "mas_pcc", "mac_pcc", "skill_vs_prior"):
                        flat[f"chromosome/{view}/{chrom}/{metric}"] = metrics[metric]
            run.log(flat); run.summary.update(flat); run.finish()
        return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--canonical-root", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--feature-cache", required=True, help="cache containing all Array cpg_idx, embeddings, prior and sigma")
    p.add_argument("--rna-cache", required=True, help="frozen standardized RNA cache used by training")
    p.add_argument("--registry", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--sample-chunk", type=int, default=128)
    p.add_argument("--cpg-chunk", type=int, default=2048)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-name", default=None)
    args = p.parse_args()
    evaluator = GenomewideEvaluator(args)
    try:
        result = evaluator.run()
        print(json.dumps(result, indent=2), flush=True)
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
