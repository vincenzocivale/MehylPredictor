#!/usr/bin/env python3
"""Mean-specific diagnostics for mean_contribution_2026_09.

Modes
-----
checkpoint
    Evaluate one trained checkpoint without materializing full prediction
    matrices. Writes mean-head accuracy, an h_mean linear-probe test,
    locus-level bias, and performance stratified by target variance.
dataset
    Compute an exact missing-aware variance decomposition of the official
    Array matrix. No checkpoint and no GPU are required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from methylation_predictor.rna_training import RNAMethylationTrainer  # noqa: E402
from methylation_predictor.rna_training.matched_chr1_data import load_matched_chr1_protocol_and_sources  # noqa: E402
from methylation_predictor.run_store import write_json  # noqa: E402


def _basic_regression(target: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(target) & np.isfinite(pred)
    t = target[valid].astype(np.float64)
    p = pred[valid].astype(np.float64)
    if len(t) == 0:
        return {"n": 0, "mse": float("nan"), "mae": float("nan"), "pearson": float("nan"), "r2": float("nan")}
    mse = float(np.mean((p - t) ** 2))
    mae = float(np.mean(np.abs(p - t)))
    pearson = float(np.corrcoef(t, p)[0, 1]) if len(t) >= 2 and np.std(t) > 0 and np.std(p) > 0 else float("nan")
    denom = float(np.sum((t - t.mean()) ** 2))
    r2 = float(1.0 - np.sum((p - t) ** 2) / denom) if denom > 0 else float("nan")
    return {"n": int(len(t)), "mse": mse, "mae": mae, "pearson": pearson, "r2": r2}


def _target_mu(trainer: RNAMethylationTrainer, cpg_ids: np.ndarray) -> np.ndarray:
    present = trainer.cpg_target_index.contains(cpg_ids)
    out = np.full(len(cpg_ids), np.nan, dtype=np.float64)
    if present.any():
        rows = trainer.cpg_target_index.positions_of(cpg_ids[present])
        out[present] = trainer.cpg_target_mu[rows].astype(np.float64)
    return out


def _functional_inputs(
    trainer: RNAMethylationTrainer,
    cpg_ids: np.ndarray,
) -> dict[str, torch.Tensor]:
    if trainer.functional is None:
        raise RuntimeError("functional diagnostics require FunctionalLocusCache")
    batch = trainer.functional.get(cpg_ids)
    return {
        "functional_track_indices": torch.from_numpy(
            batch["track_indices"]
        ).to(trainer.device),
        "functional_offsets": torch.from_numpy(
            batch["offsets"]
        ).to(trainer.device),
        "functional_dense": torch.from_numpy(
            batch["dense"]
        ).to(trainer.device).float(),
    }


@torch.no_grad()
def _extract_functional_locus(
    trainer: RNAMethylationTrainer,
    cpg_ids: np.ndarray,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not trainer.use_mean_branch or trainer.model.mean_head is None:
        return None

    reps, head_pred = [], []
    trainer.model.eval()
    for c0 in range(0, len(cpg_ids), chunk):
        local = cpg_ids[c0:c0 + chunk]
        kwargs = _functional_inputs(trainer, local)
        with trainer._autocast():
            peak = trainer.model.track_embedding(
                kwargs["functional_track_indices"],
                kwargs["functional_offsets"],
            )
            dense = trainer.model.dense_encoder(
                kwargs["functional_dense"]
            )
            h_c = trainer.model.locus_norm(peak + dense)
            mu = torch.sigmoid(
                trainer.model.mean_head(h_c).squeeze(-1)
            )
        reps.append(h_c.float().cpu().numpy())
        head_pred.append(mu.float().cpu().numpy())

    return np.concatenate(reps), np.concatenate(head_pred)


def _linear_probe(
    train_x: np.ndarray,
    train_mu: np.ndarray,
    val_x: np.ndarray,
    val_mu: np.ndarray,
) -> dict:
    eps = 1e-4
    keep_train = np.isfinite(train_mu) & np.all(np.isfinite(train_x), axis=1)
    keep_val = np.isfinite(val_mu) & np.all(np.isfinite(val_x), axis=1)
    x = train_x[keep_train].astype(np.float64)
    y_beta = np.clip(train_mu[keep_train], eps, 1 - eps)
    y = np.log(y_beta / (1.0 - y_beta))
    xv = val_x[keep_val].astype(np.float64)
    if len(x) == 0 or len(xv) == 0:
        return {
            "fit_n": int(len(x)),
            "eval_n": int(len(xv)),
            "metrics": _basic_regression(np.array([]), np.array([])),
        }

    # OLS with intercept: no probe hyperparameter is tuned on held-out CpGs.
    design = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred_logit = np.concatenate([xv, np.ones((len(xv), 1))], axis=1) @ coef
    pred_mu = 1.0 / (1.0 + np.exp(-np.clip(pred_logit, -30, 30)))
    return {
        "fit_n": int(len(x)),
        "eval_n": int(len(xv)),
        "target": "training-sample CpG mean; OLS fit on logit(mu), metrics in beta space",
        "metrics": _basic_regression(val_mu[keep_val], pred_mu),
    }


@torch.no_grad()
def _view_diagnostics(
    trainer: RNAMethylationTrainer,
    sample_ids: np.ndarray,
    cpg_ids: np.ndarray,
    *,
    sample_chunk: int,
    cpg_chunk: int,
) -> dict:
    source = trainer._sources["array"]
    rows = source.rows_of_samples(sample_ids)
    n = np.zeros(len(cpg_ids), dtype=np.float64)
    st = np.zeros(len(cpg_ids), dtype=np.float64)
    sp = np.zeros(len(cpg_ids), dtype=np.float64)
    stt = np.zeros(len(cpg_ids), dtype=np.float64)
    sse = np.zeros(len(cpg_ids), dtype=np.float64)

    trainer.model.eval()
    for c0 in range(0, len(cpg_ids), cpg_chunk):
        c1 = min(c0 + cpg_chunk, len(cpg_ids))
        local_c = cpg_ids[c0:c1]
        functional_kwargs = _functional_inputs(trainer, local_c)
        for s0 in range(0, len(sample_ids), sample_chunk):
            s1 = min(s0 + sample_chunk, len(sample_ids))
            local_s = sample_ids[s0:s1]
            rna_x = torch.from_numpy(
                trainer.rna.rows(local_s)
            ).to(trainer.device)
            with trainer._autocast():
                pred = trainer.model(
                    rna_x,
                    None,
                    **functional_kwargs,
                )["beta"]
            p = pred.float().cpu().numpy().astype(np.float64)
            t = source.block(rows[s0:s1], local_c).astype(np.float64)
            valid = np.isfinite(t) & np.isfinite(p)
            tv = np.where(valid, t, 0.0)
            pv = np.where(valid, p, 0.0)
            sl = slice(c0, c1)
            n[sl] += valid.sum(axis=0)
            st[sl] += tv.sum(axis=0)
            sp[sl] += pv.sum(axis=0)
            stt[sl] += (tv * tv).sum(axis=0)
            sse[sl] += ((pv - tv) ** 2).sum(axis=0)

    good = n >= 2
    true_mean = np.divide(st, n, out=np.full_like(st, np.nan), where=n > 0)
    pred_mean = np.divide(sp, n, out=np.full_like(sp, np.nan), where=n > 0)
    variance = np.divide(stt, n, out=np.full_like(stt, np.nan), where=n > 0) - true_mean ** 2
    variance = np.maximum(variance, 0.0)
    per_cpg_mse = np.divide(sse, n, out=np.full_like(sse, np.nan), where=n > 0)
    bias = pred_mean - true_mean
    abs_bias = np.abs(bias)

    valid_bias = good & np.isfinite(abs_bias) & np.isfinite(per_cpg_mse)
    bias_summary = {
        "n_cpg": int(valid_bias.sum()),
        "mean_abs_bias": float(np.mean(abs_bias[valid_bias])),
        "median_abs_bias": float(np.median(abs_bias[valid_bias])),
        "p90_abs_bias": float(np.quantile(abs_bias[valid_bias], 0.90)),
        "rmse_locus_mean": float(np.sqrt(np.mean(bias[valid_bias] ** 2))),
        "mean_per_cpg_mse": float(np.mean(per_cpg_mse[valid_bias])),
    }

    idx = np.flatnonzero(valid_bias & np.isfinite(variance))
    idx = idx[np.argsort(variance[idx], kind="stable")]
    deciles = []
    for bin_i, part in enumerate(np.array_split(idx, 10), start=1):
        if len(part) == 0:
            continue
        deciles.append({
            "decile": bin_i,
            "n_cpg": int(len(part)),
            "variance_min": float(np.min(variance[part])),
            "variance_median": float(np.median(variance[part])),
            "variance_max": float(np.max(variance[part])),
            "mse_mean": float(np.mean(per_cpg_mse[part])),
            "mse_median": float(np.median(per_cpg_mse[part])),
            "abs_bias_mean": float(np.mean(abs_bias[part])),
            "abs_bias_median": float(np.median(abs_bias[part])),
        })
    return {
        "locus_bias": bias_summary,
        "target_variance": {
            "median": float(np.nanmedian(variance[good])),
            "mean": float(np.nanmean(variance[good])),
        },
        "variance_deciles": deciles,
    }


def checkpoint_mode(args) -> int:
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    lc = ckpt.get("locus_cls") or {}
    scratch = Path(args.output).parent / ".mean_diag_eval"
    trainer = RNAMethylationTrainer(
        canonical_root=args.canonical_root,
        scope=args.scope,
        recipe_path=args.recipe,
        rna_cache=args.rna_cache,
        prior_cache=args.prior_cache,
        registry=args.registry,
        cpg_targets_dir=args.cpg_targets_dir,
        matched_chr1_root=args.prepared_root,
        output_root=scratch,
        mode="final",
        run_id=f"mean-diag-{Path(args.checkpoint).stem}",
        use_mean_branch=lc.get("use_mean_branch", True),
        aux_weight=lc.get("aux_weight", 0.15),
        functional_atlas=args.functional_atlas,
        annotation_cache=args.annotation_cache,
        track=False,
    )
    try:
        trainer.model.load_state_dict(ckpt["model_state"])
        trainer.model.eval()
        p = trainer.protocol
        train_c, val_c = p.array_train_cpg_idx, p.array_val_cpg_idx
        train_mu, val_mu = _target_mu(trainer, train_c), _target_mu(trainer, val_c)

        direct_mean = None
        probe = None
        train_repr = _extract_functional_locus(
            trainer, train_c, args.representation_chunk
        )
        val_repr = _extract_functional_locus(
            trainer, val_c, args.representation_chunk
        )
        if train_repr is not None and val_repr is not None:
            train_h, _ = train_repr
            val_h, val_head = val_repr
            direct_mean = _basic_regression(val_mu, val_head)
            probe = _linear_probe(train_h, train_mu, val_h, val_mu)

        views = {
            "val_cpg_x_train_sample": (p.array_train_sample_idx, p.array_val_cpg_idx),
            "val_cpg_x_val_sample": (p.array_val_sample_idx, p.array_val_cpg_idx),
        }
        view_results = {
            name: _view_diagnostics(
                trainer,
                sample_ids,
                cpg_ids,
                sample_chunk=args.sample_chunk,
                cpg_chunk=args.cpg_chunk,
            )
            for name, (sample_ids, cpg_ids) in views.items()
        }
        result = {
            "study": "mean_contribution_functional_2026_09",
            "scope": args.scope,
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": ckpt.get("epoch"),
            "locus_cls": lc,
            "mean_head_on_unseen_cpg": direct_mean,
            "functional_locus_linear_probe_train_cpg_to_val_cpg": probe,
            "views": view_results,
            "interpretation_contract": {
                "mean_head": "Direct auxiliary-head accuracy on official val CpGs; null if mean branch is absent.",
                "linear_probe": "OLS probe fit on functional h_c using train CpGs only and evaluated on official val CpGs.",
                "locus_bias": "Final beta averaged across samples per CpG; lower absolute bias is better.",
                "variance_deciles": "CpGs ranked by true across-sample variance; decile 1 is lowest variance.",
            },
        }
        write_json(Path(args.output), result)
        print(json.dumps(result, indent=2))
        return 0
    finally:
        trainer.close()


def _variance_decomposition(
    source,
    sample_ids: np.ndarray,
    cpg_ids: np.ndarray,
    *,
    sample_chunk: int,
    cpg_chunk: int,
) -> dict:
    rows = source.rows_of_samples(sample_ids)
    n = np.zeros(len(cpg_ids), dtype=np.float64)
    st = np.zeros(len(cpg_ids), dtype=np.float64)
    stt = np.zeros(len(cpg_ids), dtype=np.float64)
    for c0 in range(0, len(cpg_ids), cpg_chunk):
        c1 = min(c0 + cpg_chunk, len(cpg_ids))
        local_c = cpg_ids[c0:c1]
        for s0 in range(0, len(sample_ids), sample_chunk):
            s1 = min(s0 + sample_chunk, len(sample_ids))
            t = source.block(rows[s0:s1], local_c).astype(np.float64)
            valid = np.isfinite(t)
            tv = np.where(valid, t, 0.0)
            sl = slice(c0, c1)
            n[sl] += valid.sum(axis=0)
            st[sl] += tv.sum(axis=0)
            stt[sl] += (tv * tv).sum(axis=0)

    good = n > 0
    total_n = float(np.sum(n[good]))
    grand = float(np.sum(st[good]) / total_n)
    mu = np.divide(st, n, out=np.zeros_like(st), where=n > 0)
    within_ss = float(np.sum(stt[good] - (st[good] ** 2) / n[good]))
    between_ss = float(np.sum(n[good] * (mu[good] - grand) ** 2))
    within = within_ss / total_n
    between = between_ss / total_n
    total = within + between
    return {
        "observed_cells": int(total_n),
        "n_cpg": int(good.sum()),
        "grand_mean": grand,
        "within_locus_variance": within,
        "between_locus_mean_variance": between,
        "total_variance": total,
        "fraction_between_locus": float(between / total) if total > 0 else float("nan"),
        "fraction_within_locus": float(within / total) if total > 0 else float("nan"),
    }


def dataset_mode(args) -> int:
    protocol, sources = load_matched_chr1_protocol_and_sources(
        Path(args.prepared_root),
        Path(args.canonical_root),
        hdf5_cache_mb=256,
    )
    try:
        source = sources["array"]
        result = {
            "study": "mean_contribution_functional_2026_09",
            "definition": "Exact missing-aware law-of-total-variance decomposition across CpGs, weighted by observed cells.",
            "train_cpg_x_train_sample": _variance_decomposition(
                source,
                protocol.array_train_sample_idx,
                protocol.array_train_cpg_idx,
                sample_chunk=args.sample_chunk,
                cpg_chunk=args.cpg_chunk,
            ),
            "val_cpg_x_train_sample_confirmation": _variance_decomposition(
                source,
                protocol.array_train_sample_idx,
                protocol.array_val_cpg_idx,
                sample_chunk=args.sample_chunk,
                cpg_chunk=args.cpg_chunk,
            ),
        }
        write_json(Path(args.output), result)
        print(json.dumps(result, indent=2))
        return 0
    finally:
        for src in sources.values():
            try:
                src.close()
            except Exception:
                pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    ck = sub.add_parser("checkpoint")
    ck.add_argument("--checkpoint", required=True)
    ck.add_argument("--recipe", required=True)
    ck.add_argument("--scope", default="chr1", choices=["chr1", "chr123"])
    ck.add_argument("--canonical-root", required=True)
    ck.add_argument("--prepared-root", required=True)
    ck.add_argument("--prior-cache", required=True)
    ck.add_argument("--rna-cache", required=True)
    ck.add_argument("--registry", required=True)
    ck.add_argument("--cpg-targets-dir", required=True)
    ck.add_argument("--functional-atlas", required=True)
    ck.add_argument("--annotation-cache", required=True)
    ck.add_argument("--output", required=True)
    ck.add_argument("--sample-chunk", type=int, default=128)
    ck.add_argument("--cpg-chunk", type=int, default=2048)
    ck.add_argument("--representation-chunk", type=int, default=4096)

    ds = sub.add_parser("dataset")
    ds.add_argument("--canonical-root", required=True)
    ds.add_argument("--prepared-root", required=True)
    ds.add_argument("--output", required=True)
    ds.add_argument("--sample-chunk", type=int, default=256)
    ds.add_argument("--cpg-chunk", type=int, default=4096)

    args = ap.parse_args()
    return checkpoint_mode(args) if args.mode == "checkpoint" else dataset_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
