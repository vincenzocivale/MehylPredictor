#!/usr/bin/env python3
"""Strict matched chr1 test evaluation for our model and MethylProphet.

Our model is run from its selected checkpoint on the complete configured
``test samples x test CpGs`` panel. MethylProphet is evaluated from the official
released prediction rows attributable to its released checkpoint; this follows
the repository's existing, safer convention rather than constructing a new
unverified live forward-pass harness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from methylation_predictor.rna_branch.biological_metrics import (
    BiologicalMetricConfig,
    differential_effect_recovery,
    per_locus_biological_metrics,
    per_sample_biological_metrics,
    regional_effect_recovery,
    structural_fidelity,
    summarize_biological_fidelity,
)
from methylation_predictor.rna_branch.calibration import apply_calibration, fit_alpha
from methylation_predictor.rna_branch.config import load_config
from methylation_predictor.rna_branch.metrics import evaluate_predictions
from methylation_predictor.rna_branch.trainer import ExperimentRunner


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t" if path.suffix.lower() in {".tsv", ".txt"} else ",")
    raise ValueError(f"unsupported table format: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    """sha256 of a file, or -- for a sharded parquet dataset directory (e.g. a
    HF-released `eval_results-test.parquet/000000.parquet, ...` dataset dir,
    which is not a single file) -- sha256 of the sorted, newline-joined
    per-shard sha256 digests. This is its own self-contained, path-independent
    convention (deterministic regardless of absolute/relative invocation path);
    it is a distinct, equally valid digest from the `sha256sum *.parquet |
    sort | sha256sum` one manually recorded in
    `artifacts/methylprophet_audit/recommended_artifacts.json` (that one
    includes filenames in each hashed line), not required to match it
    bit-for-bit.
    """
    if path.is_file():
        return _sha256_file(path)
    shard_hashes = sorted(_sha256_file(shard) for shard in path.iterdir() if shard.is_file())
    digest = hashlib.sha256()
    digest.update("\n".join(shard_hashes).encode("utf-8"))
    return digest.hexdigest()


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _mapping(
    path: Path | None,
    source_column: str,
    target_column: str,
) -> dict[str, str] | None:
    if path is None:
        return None
    frame = _read_table(path)
    missing = [column for column in (source_column, target_column) if column not in frame]
    if missing:
        raise ValueError(f"mapping {path} is missing columns {missing}")
    frame = frame[[source_column, target_column]].dropna().copy()
    frame[source_column] = frame[source_column].astype(str)
    frame[target_column] = frame[target_column].astype(str)
    if frame[source_column].duplicated().any():
        raise ValueError(f"mapping {path} has duplicate source IDs")
    return dict(zip(frame[source_column], frame[target_column], strict=True))


def _region_ids(
    path: Path | None,
    cpg_ids: np.ndarray,
    *,
    id_column: str,
    region_column: str,
) -> tuple[np.ndarray | None, dict[str, int] | None]:
    if path is None:
        return None, None
    frame = _read_table(path)
    missing = [column for column in (id_column, region_column) if column not in frame]
    if missing:
        raise ValueError(f"region annotation {path} is missing columns {missing}")
    frame = frame[[id_column, region_column]].dropna().copy()
    frame[id_column] = frame[id_column].astype(str)
    frame[region_column] = frame[region_column].astype(str)
    if frame[id_column].duplicated().any():
        raise ValueError(f"region annotation {path} has duplicate CpG IDs")
    mapping = dict(zip(frame[id_column], frame[region_column], strict=True))
    aligned = np.array([mapping.get(str(cpg), "") for cpg in cpg_ids], dtype=object)
    annotated = aligned != ""
    return aligned.astype(str), {
        "annotated_cpgs": int(annotated.sum()),
        "unannotated_cpgs": int((~annotated).sum()),
        "unique_regions": int(len(np.unique(aligned[annotated]))) if annotated.any() else 0,
    }


def _apply_mapping(values: pd.Series, mapping: dict[str, str] | None, name: str) -> pd.Series:
    source = values.astype(str)
    if mapping is None:
        return source
    mapped = source.map(mapping)
    if mapped.isna().any():
        examples = source[mapped.isna()].drop_duplicates().head().tolist()
        raise ValueError(f"{name} mapping is incomplete; examples={examples}")
    return mapped.astype(str)


def _methylprophet_static_prior(
    path: Path,
    cpg_ids: np.ndarray,
    *,
    cpg_column: str,
    value_column: str,
) -> np.ndarray:
    """Aligns MethylProphet's OWN static (mean-RNA) prior to ``cpg_ids``.

    MethylProphet's calibration alpha must be fit against its own static
    prior, never against our NTv3 prior -- conflating the two anchors was a
    real bug caught in this repo's history (see docs/methylprophet_final_artifact_audit.md
    and the [[rna_branch_stage_d2_f2_benchmark]] memory: "do NOT calibrate
    MethylProphet's dynamic component against the NTv3 prior, it must be
    against MP's own static prior"). Fails loudly on incomplete coverage
    rather than silently falling back to a partial/wrong anchor.
    """
    frame = _read_table(path)
    missing = [column for column in (cpg_column, value_column) if column not in frame]
    if missing:
        raise ValueError(f"MethylProphet static prior {path} is missing columns {missing}")
    frame = frame[[cpg_column, value_column]].dropna().copy()
    frame[cpg_column] = frame[cpg_column].astype(str)
    if frame[cpg_column].duplicated().any():
        raise ValueError(f"MethylProphet static prior {path} has duplicate CpG IDs")
    lookup = dict(zip(frame[cpg_column], frame[value_column], strict=True))
    missing_ids = [cpg for cpg in cpg_ids if cpg not in lookup]
    if missing_ids:
        raise ValueError(
            f"MethylProphet static prior {path} is missing {len(missing_ids)} of "
            f"{len(cpg_ids)} required CpGs; examples={missing_ids[:5]}"
        )
    return np.asarray([lookup[cpg] for cpg in cpg_ids], dtype=np.float64)


def _load_our_panel(config_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    """Loads the test panel (for the headline comparison) and, separately, the
    validation panel (for fitting the frozen calibration alpha -- see
    calibration.py). The two must never be mixed: alpha is fit exclusively on
    the validation arrays returned here, then applied to the test arrays.
    """
    config = load_config(config_path)
    runner = ExperimentRunner(config)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=runner.device, weights_only=False)
        runner.model.load_state_dict(checkpoint["model_state"])
        runner._refresh_train_centroids()
        panel = runner.predict_panel("test", "test", max_cpgs=None, keep_predictions=True)
        if panel.target is None or panel.prediction is None:
            raise RuntimeError("our full test prediction matrix was not retained")
        validation_panel = runner.predict_panel(
            config.training.validation_sample_split,
            config.training.validation_cpg_split,
            max_cpgs=None,
            keep_predictions=True,
        )
        if validation_panel.target is None or validation_panel.prediction is None:
            raise RuntimeError("our full validation prediction matrix was not retained")
        train_sample_indices = runner.bundle.sample_indices("train")
        return {
            "target": panel.target.astype(np.float64),
            "prediction": panel.prediction.astype(np.float64),
            "prior": runner.bundle.loci.prior[panel.cpg_indices].astype(np.float64),
            "sample_ids": runner.bundle.samples.ids[panel.sample_indices].astype(str),
            "cpg_ids": runner.bundle.loci.ids[panel.cpg_indices].astype(str),
            "cancer_types": runner.bundle.samples.cancer_types[panel.sample_indices].astype(str),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "runner": runner,
            "train_sample_indices": train_sample_indices,
            "test_cpg_indices": panel.cpg_indices,
            "validation_target": validation_panel.target.astype(np.float64),
            "validation_prediction": validation_panel.prediction.astype(np.float64),
            "validation_prior": runner.bundle.loci.prior[validation_panel.cpg_indices].astype(np.float64),
            "validation_sample_ids": runner.bundle.samples.ids[validation_panel.sample_indices].astype(str),
            "validation_cpg_ids": runner.bundle.loci.ids[validation_panel.cpg_indices].astype(str),
            "validation_cancer_types": runner.bundle.samples.cancer_types[
                validation_panel.sample_indices
            ].astype(str),
            "validation_cpg_indices": validation_panel.cpg_indices,
            "validation_eligibility_target": runner.bundle.beta(
                train_sample_indices, validation_panel.cpg_indices
            ).astype(np.float64),
        }
    except Exception:
        runner.close()
        raise


def _load_methylprophet_matrix(
    path: Path,
    *,
    canonical_sample_ids: np.ndarray,
    canonical_cpg_ids: np.ndarray,
    canonical_target: np.ndarray,
    group_column: str,
    group_idx: int | None,
    sample_column: str,
    cpg_column: str,
    prediction_column: str,
    target_column: str | None,
    sample_mapping: dict[str, str] | None,
    cpg_mapping: dict[str, str] | None,
    allow_partial_overlap: bool,
    target_tolerance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    frame = _read_table(path)
    required = [sample_column, cpg_column, prediction_column]
    if group_idx is not None:
        required.append(group_column)
    if target_column is not None:
        required.append(target_column)
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"MethylProphet predictions are missing columns {missing}")
    if group_idx is not None:
        frame = frame.loc[frame[group_column] == group_idx].copy()
    if frame.empty:
        raise ValueError("no MethylProphet rows remain after group filtering")
    frame["canonical_sample_id"] = _apply_mapping(frame[sample_column], sample_mapping, "sample")
    frame["canonical_cpg_id"] = _apply_mapping(frame[cpg_column], cpg_mapping, "CpG")
    frame[prediction_column] = pd.to_numeric(frame[prediction_column], errors="coerce")
    if frame[["canonical_sample_id", "canonical_cpg_id"]].duplicated().any():
        duplicate = frame.loc[
            frame[["canonical_sample_id", "canonical_cpg_id"]].duplicated(False),
            ["canonical_sample_id", "canonical_cpg_id"],
        ].head().to_dict("records")
        raise ValueError(f"duplicate MethylProphet sample-CpG rows: {duplicate}")

    sample_position = {value: index for index, value in enumerate(canonical_sample_ids.tolist())}
    cpg_position = {value: index for index, value in enumerate(canonical_cpg_ids.tolist())}
    in_panel = frame["canonical_sample_id"].isin(sample_position) & frame["canonical_cpg_id"].isin(cpg_position)
    panel_rows = frame.loc[in_panel].copy()
    matrix = np.full(canonical_target.shape, np.nan, dtype=np.float64)
    row_indices = panel_rows["canonical_sample_id"].map(sample_position).to_numpy(dtype=np.int64)
    col_indices = panel_rows["canonical_cpg_id"].map(cpg_position).to_numpy(dtype=np.int64)
    matrix[row_indices, col_indices] = panel_rows[prediction_column].to_numpy(dtype=np.float64)

    observed = np.isfinite(canonical_target)
    missing_observed = observed & ~np.isfinite(matrix)
    if missing_observed.any() and not allow_partial_overlap:
        first = np.argwhere(missing_observed)[:5]
        examples = [
            {"sample_id": canonical_sample_ids[r], "cpg_id": canonical_cpg_ids[c]}
            for r, c in first
        ]
        raise ValueError(
            f"MethylProphet lacks {int(missing_observed.sum())} observed cells from the complete test panel; "
            f"examples={examples}. Supply the correct official group/mappings, or explicitly use "
            "--allow-partial-overlap for an exploratory intersection-only analysis."
        )

    target_check = None
    if target_column is not None:
        provided_target = pd.to_numeric(panel_rows[target_column], errors="coerce").to_numpy(dtype=np.float64)
        canonical_values = canonical_target[row_indices, col_indices]
        valid = np.isfinite(provided_target) & np.isfinite(canonical_values)
        maximum_error = float(np.max(np.abs(provided_target[valid] - canonical_values[valid]))) if valid.any() else float("nan")
        if valid.any() and maximum_error > target_tolerance:
            raise ValueError(
                f"MethylProphet gt_methyl disagrees with canonical target: max_abs_error={maximum_error} "
                f"> tolerance={target_tolerance}"
            )
        target_check = {"rows_checked": int(valid.sum()), "max_abs_error": maximum_error}

    return matrix, {
        "input_rows": int(len(frame)),
        "rows_in_canonical_panel": int(len(panel_rows)),
        "observed_test_cells": int(observed.sum()),
        "missing_observed_predictions": int(missing_observed.sum()),
        "coverage_fraction_observed": float(1.0 - missing_observed.sum() / max(observed.sum(), 1)),
        "target_consistency": target_check,
    }


def _coordinate_blocks(
    cpg_ids: np.ndarray,
    coordinates: Path | None,
    *,
    id_column: str,
    chromosome_column: str,
    position_column: str,
    block_size: int,
) -> np.ndarray:
    if coordinates is not None:
        frame = _read_table(coordinates)
        needed = [id_column, chromosome_column, position_column]
        missing = [column for column in needed if column not in frame]
        if missing:
            raise ValueError(f"coordinate table is missing columns {missing}")
        frame[id_column] = frame[id_column].astype(str)
        if frame[id_column].duplicated().any():
            raise ValueError("coordinate table contains duplicate CpG IDs")
        indexed = frame.set_index(id_column).reindex(cpg_ids)
        if indexed[[chromosome_column, position_column]].isna().any().any():
            missing_ids = indexed.index[indexed[position_column].isna()].tolist()[:5]
            raise ValueError(f"coordinate table is incomplete; examples={missing_ids}")
        chrom = indexed[chromosome_column].astype(str).to_numpy()
        pos = indexed[position_column].to_numpy(dtype=np.int64)
    else:
        chrom, pos = [], []
        pattern = re.compile(r"^(chr[^:_]+)[:_](\d+)$", re.IGNORECASE)
        for cpg_id in cpg_ids:
            match = pattern.match(str(cpg_id))
            if not match:
                raise ValueError(
                    "cannot derive genomic coordinates from cpg_id; provide --cpg-coordinates "
                    "for the hierarchical 5-Mb block bootstrap"
                )
            chrom.append(match.group(1))
            pos.append(int(match.group(2)))
        chrom, pos = np.asarray(chrom), np.asarray(pos, dtype=np.int64)
    return np.asarray([f"{c}:{p // block_size}" for c, p in zip(chrom, pos, strict=True)], dtype=object)


def _vectorized_locus_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    prior: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, float]:
    valid = np.isfinite(target) & np.isfinite(prediction)
    counts = valid.sum(axis=0)
    safe_counts = np.maximum(counts, 1)
    t = np.where(valid, target, 0.0)
    p = np.where(valid, prediction, 0.0)
    t_mean = t.sum(axis=0) / safe_counts
    p_mean = p.sum(axis=0) / safe_counts
    tc = np.where(valid, target - t_mean, 0.0)
    pc = np.where(valid, prediction - p_mean, 0.0)
    model_sse = np.where(valid, (prediction - target) ** 2, 0.0).sum(axis=0)
    prior_sse = np.where(valid, (prior[None, :] - target) ** 2, 0.0).sum(axis=0)
    skill = 1.0 - model_sse / np.maximum(prior_sse, 1e-12)
    covariance = (tc * pc).sum(axis=0)
    t_ss = (tc * tc).sum(axis=0)
    p_ss = (pc * pc).sum(axis=0)
    ccc = 2.0 * covariance / np.maximum(t_ss + p_ss + counts * (t_mean - p_mean) ** 2, 1e-12)
    dynamic_r2 = 1.0 - ((tc - pc) ** 2).sum(axis=0) / np.maximum(t_ss, 1e-12)
    valid_loci = eligible & (counts >= 2) & np.isfinite(skill)
    return {
        "mas_skill_vs_prior_variable": float(np.nanmedian(skill[valid_loci])),
        "mas_ccc_variable": float(np.nanmedian(ccc[valid_loci])),
        "mas_dynamic_r2_variable": float(np.nanmedian(dynamic_r2[valid_loci])),
    }


def _stratified_sample_bootstrap(cancer_types: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for cancer in np.unique(cancer_types):
        rows = np.flatnonzero(cancer_types == cancer)
        parts.append(rng.choice(rows, size=len(rows), replace=True))
    return np.concatenate(parts)


def paired_hierarchical_bootstrap(
    target: np.ndarray,
    ours: np.ndarray,
    methylprophet: np.ndarray,
    prior: np.ndarray,
    cancer_types: np.ndarray,
    block_labels: np.ndarray,
    eligible: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates <= 0:
        return {"replicates": 0}
    rng = np.random.default_rng(seed)
    unique_blocks = np.unique(block_labels[eligible])
    block_to_columns = {
        block: np.flatnonzero((block_labels == block) & eligible) for block in unique_blocks
    }
    differences = {
        key: np.empty(replicates, dtype=np.float64)
        for key in ("mas_skill_vs_prior_variable", "mas_ccc_variable", "mas_dynamic_r2_variable")
    }
    for replicate in range(replicates):
        rows = _stratified_sample_bootstrap(cancer_types, rng)
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        columns = np.concatenate([block_to_columns[block] for block in sampled_blocks])
        fixed_eligible = np.ones(len(columns), dtype=bool)
        t = target[np.ix_(rows, columns)]
        b = prior[columns]
        ours_metrics = _vectorized_locus_metrics(t, ours[np.ix_(rows, columns)], b, fixed_eligible)
        mp_metrics = _vectorized_locus_metrics(t, methylprophet[np.ix_(rows, columns)], b, fixed_eligible)
        for key in differences:
            differences[key][replicate] = ours_metrics[key] - mp_metrics[key]
    result: dict[str, Any] = {"replicates": replicates, "seed": seed, "resampling": "cancer-stratified patients x genomic blocks"}
    for key, values in differences.items():
        finite = values[np.isfinite(values)]
        result[key] = {
            "ours_minus_methylprophet_mean": float(np.mean(finite)) if len(finite) else float("nan"),
            "ci95": np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else [float("nan"), float("nan")],
            "probability_difference_gt_zero": float(np.mean(finite > 0)) if len(finite) else float("nan"),
            "finite_replicates": int(len(finite)),
        }
    return result


def _downstream_cancer_type_utility(
    train_target: np.ndarray,
    train_labels: np.ndarray,
    test_target: np.ndarray,
    ours: np.ndarray,
    methylprophet: np.ndarray,
    prior: np.ndarray,
    test_labels: np.ndarray,
    *,
    max_features: int,
) -> dict[str, Any]:
    try:
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import RidgeClassifier
        from sklearn.metrics import balanced_accuracy_score, f1_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("--run-downstream requires scikit-learn") from exc
    variance = np.nanvar(train_target, axis=0)
    chosen = np.argsort(-np.nan_to_num(variance, nan=-np.inf))[: min(max_features, train_target.shape[1])]
    pipeline = make_pipeline(
        SimpleImputer(strategy="mean"),
        VarianceThreshold(0.0),
        StandardScaler(),
        RidgeClassifier(class_weight="balanced"),
    )
    pipeline.fit(train_target[:, chosen], train_labels)
    matrices = {
        "measured_test": test_target[:, chosen],
        "ours": ours[:, chosen],
        "methylprophet": methylprophet[:, chosen],
        "prior": np.broadcast_to(prior[chosen], (len(test_labels), len(chosen))),
    }
    scores = {}
    for name, matrix in matrices.items():
        predicted = pipeline.predict(matrix)
        scores[name] = {
            "balanced_accuracy": float(balanced_accuracy_score(test_labels, predicted)),
            "macro_f1": float(f1_score(test_labels, predicted, average="macro")),
        }
    for model in ("ours", "methylprophet"):
        scores[model]["balanced_accuracy_utility_retention"] = (
            (scores[model]["balanced_accuracy"] - scores["prior"]["balanced_accuracy"])
            / max(scores["measured_test"]["balanced_accuracy"] - scores["prior"]["balanced_accuracy"], 1e-12)
        )
        scores[model]["macro_f1_utility_retention"] = (
            (scores[model]["macro_f1"] - scores["prior"]["macro_f1"])
            / max(scores["measured_test"]["macro_f1"] - scores["prior"]["macro_f1"], 1e-12)
        )
    return {"task": "cancer_type", "features": int(len(chosen)), "scores": scores}


def _model_report(
    name: str,
    target: np.ndarray,
    prediction: np.ndarray,
    prior: np.ndarray,
    sample_ids: np.ndarray,
    cpg_ids: np.ndarray,
    cancer_types: np.ndarray,
    metric_config: BiologicalMetricConfig,
    output_dir: Path,
    eligibility_target: np.ndarray | None = None,
    region_ids: np.ndarray | None = None,
    min_cpgs_per_region: int = 2,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    per_locus = per_locus_biological_metrics(
        target, prediction, prior, cancer_types, cpg_ids=cpg_ids,
        eligibility_target=eligibility_target, config=metric_config
    )
    per_sample = per_sample_biological_metrics(
        target, prediction, prior, sample_ids=sample_ids, cancer_types=cancer_types
    )
    summary = evaluate_predictions(
        target,
        prediction,
        prior,
        cancer_types,
        include_biological_fidelity=True,
        biological_min_observed_samples=metric_config.min_observed_samples,
        biological_min_target_std=metric_config.min_target_std,
        biological_min_cancer_group_samples=metric_config.min_cancer_group_samples,
    )
    # Overwrite from the persisted tables so every headline number has exact
    # row-level provenance.
    summary.update(summarize_biological_fidelity(per_locus, per_sample))
    eligible_mask = per_locus["eligible_variable_cpg"].to_numpy(dtype=bool)
    summary["structural_fidelity"] = structural_fidelity(
        target, prediction, eligible_cpg_mask=eligible_mask, config=metric_config
    )
    summary["differential_effect_recovery"] = differential_effect_recovery(
        target, prediction, cancer_types, config=metric_config
    )
    summary["regional_dmr_effect_recovery"] = (
        regional_effect_recovery(
            target, prediction, cancer_types, region_ids,
            min_cpgs_per_region=min_cpgs_per_region, config=metric_config
        )
        if region_ids is not None
        else None
    )
    per_locus.to_parquet(output_dir / f"{name}_per_cpg.parquet", index=False)
    per_sample.to_parquet(output_dir / f"{name}_per_sample.parquet", index=False)
    return summary, per_locus, per_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--our-config", type=Path, required=True)
    parser.add_argument("--our-checkpoint", type=Path, required=True)
    parser.add_argument("--methylprophet-checkpoint", type=Path, required=True)
    parser.add_argument("--methylprophet-predictions", type=Path, required=True)
    parser.add_argument("--methylprophet-group-idx", type=int)
    parser.add_argument("--mp-group-column", default="group_idx")
    parser.add_argument("--mp-sample-column", default="sample_idx")
    parser.add_argument("--mp-cpg-column", default="cpg_idx")
    parser.add_argument("--mp-prediction-column", default="pred_methyl")
    parser.add_argument("--mp-target-column", default="gt_methyl")
    parser.add_argument("--mp-sample-map", type=Path)
    parser.add_argument("--mp-sample-map-source-column", default="sample_idx")
    parser.add_argument("--mp-sample-map-target-column", default="sample_id")
    parser.add_argument("--mp-cpg-map", type=Path)
    parser.add_argument("--mp-cpg-map-source-column", default="cpg_idx")
    parser.add_argument("--mp-cpg-map-target-column", default="cpg_id")
    parser.add_argument("--allow-partial-overlap", action="store_true")
    parser.add_argument("--target-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--min-observed-samples", type=int, default=20)
    parser.add_argument("--min-target-std", type=float, default=0.05)
    parser.add_argument("--min-cancer-group-samples", type=int, default=4)
    parser.add_argument("--dmr-delta-threshold", type=float, default=0.10)
    parser.add_argument("--dmr-region-annotation", type=Path)
    parser.add_argument("--dmr-region-id-column", default="cpg_id")
    parser.add_argument("--dmr-region-column", default="region_id")
    parser.add_argument("--dmr-min-cpgs-per-region", type=int, default=2)
    parser.add_argument("--structure-max-cpgs", type=int, default=2048)
    parser.add_argument("--cpg-coordinates", type=Path)
    parser.add_argument("--coordinate-id-column", default="cpg_idx")
    parser.add_argument("--chromosome-column", default="chr")
    parser.add_argument("--position-column", default="pos")
    parser.add_argument("--genomic-block-size", type=int, default=5_000_000)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260805)
    parser.add_argument("--run-downstream", action="store_true")
    parser.add_argument("--downstream-max-features", type=int, default=4096)
    # Post-hoc global calibration of the dynamic component (frozen on
    # validation, applied to test -- see calibration.py docstring for the
    # exact formula). Primary headline stays raw; calibrated rows are
    # reported alongside it, never in place of it.
    parser.add_argument("--calibration-objective", choices=["mse", "median_skill"], default="mse")
    parser.add_argument("--calibration-alpha-min", type=float, default=0.0)
    parser.add_argument("--calibration-alpha-max", type=float, default=4.0)
    parser.add_argument("--calibration-alpha-grid-points", type=int, default=161)
    # Optional: a released MethylProphet prediction file covering OUR
    # validation split (may be the same file as --methylprophet-predictions
    # under a different group, or a separate file). Without this, MethylProphet
    # calibration is skipped and only its raw predictions are reported.
    parser.add_argument("--methylprophet-validation-predictions", type=Path)
    parser.add_argument("--methylprophet-validation-group-idx", type=int)
    # MethylProphet's OWN static (mean-RNA) prior -- required to correctly
    # anchor ITS calibration alpha fit (see _methylprophet_static_prior
    # docstring). Without this, MP's alpha is fit against our NTv3 prior,
    # which is a known-wrong anchor from prior project history; the script
    # still runs but loudly flags the fallback as unreliable.
    parser.add_argument("--methylprophet-static-prior", type=Path)
    parser.add_argument("--methylprophet-static-prior-cpg-column", default="cpg_idx")
    parser.add_argument("--methylprophet-static-prior-value-column", default="pred_methyl")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ours_bundle = _load_our_panel(args.our_config, args.our_checkpoint)
    runner: ExperimentRunner = ours_bundle.pop("runner")
    try:
        target = ours_bundle["target"]
        ours = ours_bundle["prediction"]
        prior = ours_bundle["prior"]
        sample_ids = ours_bundle["sample_ids"]
        cpg_ids = ours_bundle["cpg_ids"]
        cancer_types = ours_bundle["cancer_types"]
        sample_map = _mapping(
            args.mp_sample_map,
            args.mp_sample_map_source_column,
            args.mp_sample_map_target_column,
        )
        cpg_map = _mapping(
            args.mp_cpg_map,
            args.mp_cpg_map_source_column,
            args.mp_cpg_map_target_column,
        )
        mp, mp_alignment = _load_methylprophet_matrix(
            args.methylprophet_predictions,
            canonical_sample_ids=sample_ids,
            canonical_cpg_ids=cpg_ids,
            canonical_target=target,
            group_column=args.mp_group_column,
            group_idx=args.methylprophet_group_idx,
            sample_column=args.mp_sample_column,
            cpg_column=args.mp_cpg_column,
            prediction_column=args.mp_prediction_column,
            target_column=(None if args.mp_target_column.lower() == "none" else args.mp_target_column),
            sample_mapping=sample_map,
            cpg_mapping=cpg_map,
            allow_partial_overlap=args.allow_partial_overlap,
            target_tolerance=args.target_tolerance,
        )
        common_observed = np.isfinite(target) & np.isfinite(ours) & np.isfinite(mp)
        if args.allow_partial_overlap:
            target = np.where(common_observed, target, np.nan)
            ours = np.where(common_observed, ours, np.nan)
            mp = np.where(common_observed, mp, np.nan)

        metric_config = BiologicalMetricConfig(
            min_observed_samples=args.min_observed_samples,
            min_target_std=args.min_target_std,
            min_cancer_group_samples=args.min_cancer_group_samples,
            dmr_delta_threshold=args.dmr_delta_threshold,
            max_structure_cpgs=args.structure_max_cpgs,
            structure_seed=args.bootstrap_seed,
        )
        region_ids, region_audit = _region_ids(
            args.dmr_region_annotation,
            cpg_ids,
            id_column=args.dmr_region_id_column,
            region_column=args.dmr_region_column,
        )
        train_samples = ours_bundle["train_sample_indices"]
        test_cpgs = ours_bundle["test_cpg_indices"]
        # Eligibility is frozen from training patients at the held-out CpGs.
        # Test targets are used to score models, never to decide which loci count.
        train_target_at_test_cpgs = runner.bundle.beta(train_samples, test_cpgs).astype(np.float64)
        ours_report, ours_locus, _ = _model_report(
            "ours", target, ours, prior, sample_ids, cpg_ids, cancer_types, metric_config,
            args.output_dir, eligibility_target=train_target_at_test_cpgs,
            region_ids=region_ids, min_cpgs_per_region=args.dmr_min_cpgs_per_region
        )
        mp_report, mp_locus, _ = _model_report(
            "methylprophet", target, mp, prior, sample_ids, cpg_ids, cancer_types, metric_config,
            args.output_dir, eligibility_target=train_target_at_test_cpgs,
            region_ids=region_ids, min_cpgs_per_region=args.dmr_min_cpgs_per_region
        )

        # --- Post-hoc calibration: alpha is fit exclusively on the VALIDATION
        # panel (never test), then frozen and applied to the test matrices
        # above. Primary headline stays raw; calibrated rows are additional. ---
        validation_target = ours_bundle["validation_target"]
        validation_prediction = ours_bundle["validation_prediction"]
        validation_prior = ours_bundle["validation_prior"]
        validation_locus = per_locus_biological_metrics(
            validation_target, validation_prediction, validation_prior,
            ours_bundle["validation_cancer_types"],
            cpg_ids=ours_bundle["validation_cpg_ids"],
            eligibility_target=ours_bundle["validation_eligibility_target"],
            config=metric_config,
        )
        validation_eligible = validation_locus["eligible_variable_cpg"].to_numpy(dtype=bool)
        ours_alpha_mse = fit_alpha(
            validation_target, validation_prior, validation_prediction,
            objective="mse", alpha_min=args.calibration_alpha_min,
            alpha_max=args.calibration_alpha_max, alpha_grid_points=args.calibration_alpha_grid_points,
        )
        ours_alpha_skill = fit_alpha(
            validation_target, validation_prior, validation_prediction,
            objective="median_skill", eligible_cpg_mask=validation_eligible,
            alpha_min=args.calibration_alpha_min, alpha_max=args.calibration_alpha_max,
            alpha_grid_points=args.calibration_alpha_grid_points,
        )
        ours_frozen_alpha = ours_alpha_mse if args.calibration_objective == "mse" else ours_alpha_skill
        ours_calibrated = apply_calibration(prior, ours, ours_frozen_alpha.alpha)
        ours_calibrated_report, _, _ = _model_report(
            "ours_calibrated", target, ours_calibrated, prior, sample_ids, cpg_ids, cancer_types,
            metric_config, args.output_dir, eligibility_target=train_target_at_test_cpgs,
            region_ids=region_ids, min_cpgs_per_region=args.dmr_min_cpgs_per_region,
        )

        mp_calibrated = None
        mp_calibrated_report = None
        mp_alpha_mse = mp_alpha_skill = None
        mp_validation_alignment = None
        if args.methylprophet_validation_predictions is not None:
            mp_validation, mp_validation_alignment = _load_methylprophet_matrix(
                args.methylprophet_validation_predictions,
                canonical_sample_ids=ours_bundle["validation_sample_ids"],
                canonical_cpg_ids=ours_bundle["validation_cpg_ids"],
                canonical_target=validation_target,
                group_column=args.mp_group_column,
                group_idx=args.methylprophet_validation_group_idx,
                sample_column=args.mp_sample_column,
                cpg_column=args.mp_cpg_column,
                prediction_column=args.mp_prediction_column,
                target_column=(None if args.mp_target_column.lower() == "none" else args.mp_target_column),
                sample_mapping=sample_map,
                cpg_mapping=cpg_map,
                allow_partial_overlap=args.allow_partial_overlap,
                target_tolerance=args.target_tolerance,
            )
            if args.methylprophet_static_prior is not None:
                mp_anchor_prior_validation = _methylprophet_static_prior(
                    args.methylprophet_static_prior, ours_bundle["validation_cpg_ids"],
                    cpg_column=args.methylprophet_static_prior_cpg_column,
                    value_column=args.methylprophet_static_prior_value_column,
                )
                mp_anchor_prior_test = _methylprophet_static_prior(
                    args.methylprophet_static_prior, cpg_ids,
                    cpg_column=args.methylprophet_static_prior_cpg_column,
                    value_column=args.methylprophet_static_prior_value_column,
                )
                mp_anchor_source = "methylprophet_own_static_prior"
            else:
                print(
                    "WARNING: --methylprophet-static-prior not supplied; MethylProphet's "
                    "calibration alpha will be fit against OUR NTv3 prior instead of its own "
                    "static prior. This is a known-wrong anchor (see "
                    "docs/methylprophet_final_artifact_audit.md) -- treat the resulting "
                    "methylprophet_calibrated row as unreliable.",
                    file=sys.stderr,
                )
                mp_anchor_prior_validation = validation_prior
                mp_anchor_prior_test = prior
                mp_anchor_source = "shared_ntv3_fallback_NOT_RECOMMENDED"
            mp_alpha_mse = fit_alpha(
                validation_target, mp_anchor_prior_validation, mp_validation,
                objective="mse", alpha_min=args.calibration_alpha_min,
                alpha_max=args.calibration_alpha_max, alpha_grid_points=args.calibration_alpha_grid_points,
            )
            mp_alpha_skill = fit_alpha(
                validation_target, mp_anchor_prior_validation, mp_validation,
                objective="median_skill", eligible_cpg_mask=validation_eligible,
                alpha_min=args.calibration_alpha_min, alpha_max=args.calibration_alpha_max,
                alpha_grid_points=args.calibration_alpha_grid_points,
            )
            mp_frozen_alpha = mp_alpha_mse if args.calibration_objective == "mse" else mp_alpha_skill
            # Calibration is applied around MP's own anchor; the resulting
            # probability is then scored downstream against the SHARED NTv3
            # `prior` (via _model_report -> evaluate_predictions), which is the
            # correct, fair common yardstick for cross-model comparison.
            mp_calibrated = apply_calibration(mp_anchor_prior_test, mp, mp_frozen_alpha.alpha)
            mp_calibrated_report, _, _ = _model_report(
                "methylprophet_calibrated", target, mp_calibrated, prior, sample_ids, cpg_ids, cancer_types,
                metric_config, args.output_dir, eligibility_target=train_target_at_test_cpgs,
                region_ids=region_ids, min_cpgs_per_region=args.dmr_min_cpgs_per_region,
            )

        def _alpha_fit_dict(fit) -> dict[str, Any] | None:
            if fit is None:
                return None
            return {
                "alpha": fit.alpha,
                "objective": fit.objective,
                "objective_value": fit.objective_value,
                "fit_split": fit.fit_split,
                "alpha_grid_min": fit.alpha_grid_min,
                "alpha_grid_max": fit.alpha_grid_max,
                "alpha_grid_points": fit.alpha_grid_points,
                "observed_cells": fit.observed_cells,
                "eligible_cpgs": fit.eligible_cpgs,
            }

        calibration_report = {
            "frozen_objective": args.calibration_objective,
            "note": (
                "alpha is fit exclusively on the validation panel above, then "
                "applied unchanged to the test matrices used for every other "
                "metric in this report; primary headline rows stay raw."
            ),
            "ours": {
                "mse_objective_fit": _alpha_fit_dict(ours_alpha_mse),
                "median_skill_objective_fit": _alpha_fit_dict(ours_alpha_skill),
                "frozen_alpha": ours_frozen_alpha.alpha,
            },
            "methylprophet": (
                {
                    "mse_objective_fit": _alpha_fit_dict(mp_alpha_mse),
                    "median_skill_objective_fit": _alpha_fit_dict(mp_alpha_skill),
                    "frozen_alpha": mp_frozen_alpha.alpha,
                    "anchor_prior": mp_anchor_source,
                    "validation_alignment": mp_validation_alignment,
                }
                if mp_calibrated is not None
                else {
                    "status": "skipped",
                    "reason": "no --methylprophet-validation-predictions supplied; only raw MethylProphet reported",
                }
            ),
        }

        eligible = ours_locus["eligible_variable_cpg"].to_numpy(dtype=bool)
        if not np.array_equal(eligible, mp_locus["eligible_variable_cpg"].to_numpy(dtype=bool)):
            raise RuntimeError("model-dependent eligibility detected; eligibility must depend only on target")
        blocks = _coordinate_blocks(
            cpg_ids,
            args.cpg_coordinates,
            id_column=args.coordinate_id_column,
            chromosome_column=args.chromosome_column,
            position_column=args.position_column,
            block_size=args.genomic_block_size,
        )
        # Primary/headline comparison: raw vs raw.
        bootstrap = paired_hierarchical_bootstrap(
            target,
            ours,
            mp,
            prior,
            cancer_types,
            blocks,
            eligible,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        )
        # Secondary: our validation-calibrated model vs MethylProphet's raw
        # released predictions (MP calibration may not be available -- see
        # calibration_report["methylprophet"]).
        bootstrap_calibrated_vs_mp_raw = paired_hierarchical_bootstrap(
            target,
            ours_calibrated,
            mp,
            prior,
            cancer_types,
            blocks,
            eligible,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        )
        # Secondary: calibrated vs calibrated, only when MP calibration ran.
        bootstrap_calibrated_vs_mp_calibrated = (
            paired_hierarchical_bootstrap(
                target,
                ours_calibrated,
                mp_calibrated,
                prior,
                cancer_types,
                blocks,
                eligible,
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed,
            )
            if mp_calibrated is not None
            else None
        )

        downstream = None
        if args.run_downstream:
            downstream = _downstream_cancer_type_utility(
                train_target_at_test_cpgs,
                runner.bundle.samples.cancer_types[train_samples].astype(str),
                target,
                ours,
                mp,
                prior,
                cancer_types,
                max_features=args.downstream_max_features,
            )

        headline = [
            "mas_skill_vs_prior_variable",
            "mas_ccc_variable",
            "mas_dynamic_r2_variable",
            "within_cancer_mas_dynamic_r2_variable",
            "mas_pcc_variable",
            "median_amplitude_ratio_variable",
            "mse",
            "mae",
            "skill_vs_prior",
            "mac_ccc_median",
        ]
        comparison_rows = [
            {"model": "ours", "variant": "raw", **{key: ours_report.get(key) for key in headline}},
            {"model": "methylprophet", "variant": "raw", **{key: mp_report.get(key) for key in headline}},
            {
                "model": "ours", "variant": f"calibrated_{args.calibration_objective}",
                **{key: ours_calibrated_report.get(key) for key in headline},
            },
        ]
        if mp_calibrated_report is not None:
            comparison_rows.append({
                "model": "methylprophet", "variant": f"calibrated_{args.calibration_objective}",
                **{key: mp_calibrated_report.get(key) for key in headline},
            })
        comparison = pd.DataFrame(comparison_rows)
        comparison.to_csv(args.output_dir / "headline_metrics.tsv", sep="\t", index=False)
        matrices_to_save = {
            "target": target.astype(np.float32),
            "ours": ours.astype(np.float32),
            "ours_calibrated": ours_calibrated.astype(np.float32),
            "methylprophet": mp.astype(np.float32),
            "prior": prior.astype(np.float32),
            "sample_idx": sample_ids,
            "cpg_idx": cpg_ids,
            "cancer_type": cancer_types,
        }
        if mp_calibrated is not None:
            matrices_to_save["methylprophet_calibrated"] = mp_calibrated.astype(np.float32)
        np.savez_compressed(args.output_dir / "matched_test_matrices.npz", **matrices_to_save)
        report = {
            "scope": {
                "sample_split": "test",
                "cpg_split": "test",
                "samples": int(len(sample_ids)),
                "cpgs": int(len(cpg_ids)),
                "observed_cells": int(np.isfinite(target).sum()),
                "variable_cpg_definition": {
                    "reference_split": "train samples at test CpGs",
                    "min_observed_samples": args.min_observed_samples,
                    "min_target_std": args.min_target_std,
                },
                "partial_overlap": bool(args.allow_partial_overlap),
            },
            "ours": ours_report,
            "methylprophet": mp_report,
            "ours_calibrated": ours_calibrated_report,
            "methylprophet_calibrated": mp_calibrated_report,
            "calibration": calibration_report,
            # Primary/headline: raw vs raw. The two calibrated comparisons are
            # secondary and must not replace it in any summary table.
            "paired_hierarchical_bootstrap": bootstrap,
            "paired_hierarchical_bootstrap_calibrated_vs_mp_raw": bootstrap_calibrated_vs_mp_raw,
            "paired_hierarchical_bootstrap_calibrated_vs_mp_calibrated": bootstrap_calibrated_vs_mp_calibrated,
            "downstream_utility": downstream,
            "methylprophet_alignment": mp_alignment,
            "dmr_region_annotation": region_audit,
            "provenance": {
                "our_config": str(args.our_config.resolve()),
                "our_config_sha256": _sha256(args.our_config),
                "our_checkpoint": str(args.our_checkpoint.resolve()),
                "our_checkpoint_sha256": _sha256(args.our_checkpoint),
                "our_checkpoint_epoch": ours_bundle["checkpoint_epoch"],
                "methylprophet_checkpoint": str(args.methylprophet_checkpoint.resolve()),
                "methylprophet_checkpoint_sha256": _sha256(args.methylprophet_checkpoint),
                "methylprophet_predictions": str(args.methylprophet_predictions.resolve()),
                "methylprophet_predictions_sha256": _sha256(args.methylprophet_predictions),
                "methylprophet_group_idx": args.methylprophet_group_idx,
                "methylprophet_source": "official released prediction rows from the released checkpoint",
                "genomic_block_size": args.genomic_block_size,
                "dmr_region_annotation": (
                    str(args.dmr_region_annotation.resolve())
                    if args.dmr_region_annotation is not None else None
                ),
                "dmr_region_annotation_sha256": (
                    _sha256(args.dmr_region_annotation)
                    if args.dmr_region_annotation is not None else None
                ),
            },
        }
        _json_dump(args.output_dir / "biological_fidelity_report.json", report)
        print(comparison.to_string(index=False))
        print(f"Wrote {args.output_dir / 'biological_fidelity_report.json'}")
    finally:
        runner.close()


if __name__ == "__main__":
    main()
