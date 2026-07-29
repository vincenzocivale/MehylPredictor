#!/usr/bin/env python3
"""Strict matched-panel comparison and paired bootstrap for Stage D.

The evaluator consumes one *long* parquet file per model.  Every file must
describe precisely the same sample--CpG grid, including target NaNs, so that a
result cannot accidentally compare a different MethylProphet release split to
the RNA-branch double-OOD panel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from methylation_predictor.rna_branch.metrics import evaluate_predictions


REQUIRED = {"sample_idx", "cpg_idx", "cancer_type", "target", "prediction"}


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode() + b"\n").hexdigest()


def _read_long(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    frame = frame[list(REQUIRED)].copy()
    frame["sample_idx"] = frame.sample_idx.astype(str)
    frame["cpg_idx"] = frame.cpg_idx.astype(str)
    frame["cancer_type"] = frame.cancer_type.astype(str)
    if frame.duplicated(["sample_idx", "cpg_idx"]).any():
        raise ValueError(f"{path} has duplicate sample_idx/cpg_idx rows")
    if frame[["sample_idx", "cpg_idx", "cancer_type"]].isna().any().any():
        raise ValueError(f"{path} has null identifiers")
    return frame


def _as_panel(frame: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    samples = np.sort(frame.sample_idx.unique())
    cpgs = np.sort(frame.cpg_idx.unique())
    expected = len(samples) * len(cpgs)
    if len(frame) != expected:
        raise ValueError(f"{name} is not a complete sample x CpG panel ({len(frame)} rows, expected {expected})")
    cancer = frame[["sample_idx", "cancer_type"]].drop_duplicates()
    if len(cancer) != len(samples):
        raise ValueError(f"{name} assigns multiple cancer types to a patient")
    aligned = frame.set_index(["sample_idx", "cpg_idx"]).reindex(pd.MultiIndex.from_product([samples, cpgs]))
    return (samples, cpgs, cancer.set_index("sample_idx").loc[samples, "cancer_type"].to_numpy(),
            aligned.target.to_numpy(float).reshape(len(samples), len(cpgs)),
            aligned.prediction.to_numpy(float).reshape(len(samples), len(cpgs)))


def load_matched(models: dict[str, Path]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load models and reject every mismatch relevant to a paired comparison."""
    panels = {name: _as_panel(_read_long(path), name) for name, path in models.items()}
    first_name = next(iter(panels))
    samples, cpgs, cancer, target, _ = panels[first_name]
    target_mask = np.isfinite(target)
    predictions: dict[str, np.ndarray] = {}
    for name, (other_samples, other_cpgs, other_cancer, other_target, prediction) in panels.items():
        if not np.array_equal(samples, other_samples) or not np.array_equal(cpgs, other_cpgs):
            raise ValueError(f"{name} has different patients or CpGs from {first_name}")
        if not np.array_equal(cancer, other_cancer):
            raise ValueError(f"{name} has different cancer-type labels from {first_name}")
        if not np.array_equal(target_mask, np.isfinite(other_target)):
            raise ValueError(f"{name} has a different missing-value mask from {first_name}")
        if not np.allclose(target[target_mask], other_target[target_mask], rtol=0, atol=1e-12):
            raise ValueError(f"{name} has different observed targets from {first_name}")
        if not np.isfinite(prediction[target_mask]).all():
            raise ValueError(f"{name} has missing predictions on observed rows")
        predictions[name] = prediction
    return predictions, samples, cpgs, cancer, target


def _point_metrics(target: np.ndarray, prediction: np.ndarray, prior: np.ndarray, cancer: np.ndarray) -> dict[str, float]:
    value = evaluate_predictions(target, prediction, prior, cancer)
    return {"mse": float(value["mse"]), "skill": float(value["skill_vs_prior"]),
            "within_cancer_skill": float(value["within_cancer_skill"])}


def _bootstrap_indices(rng: np.random.Generator, mode: str, n_samples: int, block_members: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    sample_idx = np.arange(n_samples)
    block_idx = np.arange(len(block_members))
    if mode in {"patients", "hierarchical"}:
        sample_idx = rng.choice(sample_idx, n_samples, replace=True)
    if mode in {"genomic_blocks", "hierarchical"}:
        selected = rng.choice(block_idx, len(block_idx), replace=True)
        cpg_idx = np.concatenate([block_members[i] for i in selected])
    else:
        cpg_idx = np.concatenate(block_members)
    return sample_idx, cpg_idx


def paired_bootstrap(
    target: np.ndarray, prior: np.ndarray, candidate: np.ndarray, reference: np.ndarray,
    cancer: np.ndarray, block_members: list[np.ndarray], replicates: int, seed: int,
) -> dict[str, dict[str, float]]:
    """Paired CIs: delta MSE is candidate minus reference; skills vice versa."""
    rng = np.random.default_rng(seed)
    point_candidate = _point_metrics(target, candidate, prior, cancer)
    point_reference = _point_metrics(target, reference, prior, cancer)
    point = {"delta_mse": point_candidate["mse"] - point_reference["mse"],
             "delta_skill": point_candidate["skill"] - point_reference["skill"],
             "delta_within_cancer_skill": point_candidate["within_cancer_skill"] - point_reference["within_cancer_skill"]}
    output: dict[str, dict[str, float]] = {"point": point}
    for mode in ("patients", "genomic_blocks", "hierarchical"):
        values = {key: np.empty(replicates) for key in point}
        for i in range(replicates):
            rows, cols = _bootstrap_indices(rng, mode, target.shape[0], block_members)
            c = _point_metrics(target[np.ix_(rows, cols)], candidate[np.ix_(rows, cols)], prior[cols], cancer[rows])
            r = _point_metrics(target[np.ix_(rows, cols)], reference[np.ix_(rows, cols)], prior[cols], cancer[rows])
            values["delta_mse"][i] = c["mse"] - r["mse"]
            values["delta_skill"][i] = c["skill"] - r["skill"]
            values["delta_within_cancer_skill"][i] = c["within_cancer_skill"] - r["within_cancer_skill"]
        output[mode] = {f"{key}_{tail}": float(np.quantile(value, quantile))
                        for key, value in values.items() for tail, quantile in (("ci_low", .025), ("ci_high", .975))}
    return output


def _block_members(cpgs: np.ndarray, blocks: Path) -> list[np.ndarray]:
    frame = pd.read_parquet(blocks)
    if not {"cpg_idx", "genomic_block"} <= set(frame.columns):
        raise ValueError("block manifest must contain cpg_idx and genomic_block")
    lookup = frame.assign(cpg_idx=lambda x: x.cpg_idx.astype(str)).set_index("cpg_idx").genomic_block
    missing = sorted(set(cpgs) - set(lookup.index))
    if missing:
        raise ValueError(f"block manifest omits {len(missing)} evaluated CpGs")
    groups: dict[str, list[int]] = {}
    for i, cpg in enumerate(cpgs):
        groups.setdefault(str(lookup[cpg]), []).append(i)
    if not groups:
        raise ValueError("no genomic blocks")
    return [np.asarray(x, dtype=int) for x in groups.values()]


def calibrate_global(validation: Path, output: Path, anchor_column: str = "static_prediction") -> None:
    """Fit alpha on validation only and write an auditable calibrated model file."""
    frame = pd.read_parquet(validation)
    required = REQUIRED | {anchor_column}
    if missing := required - set(frame.columns):
        raise ValueError(f"{validation} lacks columns {sorted(missing)}")
    valid = np.isfinite(frame.target) & np.isfinite(frame.prediction) & np.isfinite(frame[anchor_column])
    if not valid.any():
        raise ValueError("no finite validation rows for calibration")
    dynamic = (frame.loc[valid, "prediction"] - frame.loc[valid, anchor_column]).to_numpy(float)
    target_dynamic = (frame.loc[valid, "target"] - frame.loc[valid, anchor_column]).to_numpy(float)
    denom = float(np.dot(dynamic, dynamic))
    if denom == 0:
        raise ValueError("zero MethylProphet dynamic variance on calibration data")
    alpha = float(np.dot(dynamic, target_dynamic) / denom)
    result = frame[list(REQUIRED)].copy()
    result["prediction"] = frame[anchor_column] + alpha * (frame.prediction - frame[anchor_column])
    result.to_parquet(output, index=False)
    output.with_suffix(".calibration.json").write_text(json.dumps({"alpha": alpha, "validation": str(validation), "anchor_column": anchor_column, "n_rows": int(valid.sum())}, indent=2) + "\n")


def export_rna_npz(source: Path, output: Path, prediction_key: str = "prediction") -> None:
    """Convert a saved RNA-branch panel to the exact long-table input contract."""
    data = np.load(source)
    required = {"target", "sample_idx", "cpg_idx", "cancer_type", prediction_key}
    if missing := required - set(data.files):
        raise ValueError(f"{source} lacks arrays {sorted(missing)}")
    target, prediction = data["target"], data[prediction_key]
    samples, cpgs, cancer = data["sample_idx"].astype(str), data["cpg_idx"].astype(str), data["cancer_type"].astype(str)
    if prediction.ndim == 1 and prediction.shape == (len(cpgs),):
        prediction = np.broadcast_to(prediction[None, :], target.shape)
    if target.shape != prediction.shape or target.shape != (len(samples), len(cpgs)) or len(cancer) != len(samples):
        raise ValueError(f"invalid saved panel dimensions in {source}")
    pd.DataFrame({"sample_idx": np.repeat(samples, len(cpgs)), "cpg_idx": np.tile(cpgs, len(samples)),
                  "cancer_type": np.repeat(cancer, len(cpgs)), "target": target.ravel(), "prediction": prediction.ravel()}).to_parquet(output, index=False)


def extract_checkpoint_predictions(released: Path, panel_npz: Path, output: Path, group: int = 2, atol: float = 1e-6) -> None:
    """Extract checkpoint rows only after proving they match the C0 panel targets."""
    panel = np.load(panel_npz)
    required = {"target", "sample_idx", "cpg_idx", "cancer_type"}
    if missing := required - set(panel.files):
        raise ValueError(f"{panel_npz} lacks arrays {sorted(missing)}")
    samples = panel["sample_idx"].astype(str)
    cpgs = panel["cpg_idx"].astype(str)
    target = panel["target"].astype(float)
    cancer = panel["cancer_type"].astype(str)
    dataset = ds.dataset(released, format="parquet")
    table = dataset.to_table(filter=(ds.field("group_idx") == group) & ds.field("sample_idx").isin([int(x) for x in samples]) & ds.field("cpg_idx").isin([int(x) for x in cpgs]),
                             columns=["sample_idx", "cpg_idx", "pred_methyl", "gt_methyl"])
    result = table.to_pandas()
    result["sample_idx"] = result.sample_idx.astype(str)
    result["cpg_idx"] = result.cpg_idx.astype(str)
    expected = pd.MultiIndex.from_product([samples, cpgs], names=["sample_idx", "cpg_idx"])
    result = result.set_index(["sample_idx", "cpg_idx"]).reindex(expected)
    observed = np.isfinite(target.ravel())
    if not observed.any() or not np.isfinite(result.pred_methyl.to_numpy(float)[observed]).all() or not np.isfinite(result.gt_methyl.to_numpy(float)[observed]).all():
        raise ValueError("checkpoint output does not cover every observed C0 row")
    if not np.allclose(result.gt_methyl.to_numpy(float)[observed], target.ravel()[observed], rtol=0, atol=atol):
        err = np.abs(result.gt_methyl.to_numpy(float)[observed] - target.ravel()[observed])
        raise ValueError(f"checkpoint targets differ from C0 panel (max abs error {err.max():.8g})")
    frame = result.reset_index().rename(columns={"gt_methyl": "target", "pred_methyl": "prediction"})
    # The released MDS uses a non-NaN dataset and therefore omits precisely the
    # unobserved C0 cells.  Restore C0's authoritative mask explicitly.
    frame["target"] = target.ravel()
    frame["cancer_type"] = np.repeat(cancer, len(cpgs))
    frame = frame[["sample_idx", "cpg_idx", "cancer_type", "target", "prediction"]]
    frame.to_parquet(output, index=False)
    output.with_suffix(".provenance.json").write_text(json.dumps({
        "kind": "released evaluation output of pretrained MethylProphet checkpoint",
        "released_predictions": str(released), "panel_npz": str(panel_npz), "group_idx": group,
        "checkpoint_target_match_atol": atol, "rows": len(frame),
        "checkpoint": "tcga_mix_chr1-bs_512-c2b2/version_0/finished.ckpt",
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate-global")
    cal.add_argument("--validation", type=Path, required=True)
    cal.add_argument("--output", type=Path, required=True)
    cal.add_argument("--anchor-column", default="static_prediction")
    npz = sub.add_parser("export-rna-npz")
    npz.add_argument("--input", type=Path, required=True)
    npz.add_argument("--output", type=Path, required=True)
    npz.add_argument("--prediction-key", default="prediction")
    extract = sub.add_parser("extract-checkpoint-predictions")
    extract.add_argument("--released", type=Path, required=True)
    extract.add_argument("--panel-npz", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--group", type=int, default=2)
    extract.add_argument("--atol", type=float, default=1e-6)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--model", action="append", required=True, metavar="NAME=PARQUET")
    ev.add_argument("--prior-model", required=True)
    ev.add_argument("--reference-model", required=True)
    ev.add_argument("--blocks", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--replicates", type=int, default=2000)
    ev.add_argument("--seed", type=int, default=9176)
    args = parser.parse_args()
    if args.command == "calibrate-global":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        calibrate_global(args.validation, args.output, args.anchor_column)
        return
    if args.command == "export-rna-npz":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        export_rna_npz(args.input, args.output, args.prediction_key)
        return
    if args.command == "extract-checkpoint-predictions":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        extract_checkpoint_predictions(args.released, args.panel_npz, args.output, args.group, args.atol)
        return
    models = {}
    for item in args.model:
        if "=" not in item:
            raise ValueError(f"--model must be NAME=PARQUET, received {item!r}")
        name, path = item.split("=", 1)
        if not name or name in models:
            raise ValueError(f"invalid or duplicate model name {name!r}")
        models[name] = Path(path)
    if args.prior_model not in models or args.reference_model not in models:
        raise ValueError("prior-model and reference-model must name supplied models")
    predictions, samples, cpgs, cancer, target = load_matched(models)
    members = _block_members(cpgs, args.blocks)
    prior = predictions[args.prior_model][0]
    if not np.allclose(predictions[args.prior_model], prior[None, :], rtol=0, atol=1e-12, equal_nan=True):
        raise ValueError("prior-model must be a static per-CpG prediction, identical for every patient")
    summary = {"contract": {"patients": len(samples), "cpgs": len(cpgs), "observed_rows": int(np.isfinite(target).sum()),
                            "patient_sha256": _digest(samples), "cpg_sha256": _digest(cpgs), "n_genomic_blocks": len(members),
                            "nan_handling": "identical target mask required; predictions must be finite on every observed row"},
               "metrics": {name: _point_metrics(target, prediction, prior, cancer) for name, prediction in predictions.items()},
               "paired_bootstrap": {name: paired_bootstrap(target, prior, prediction, predictions[args.reference_model], cancer, members, args.replicates, args.seed)
                                    for name, prediction in predictions.items() if name != args.reference_model},
               "bootstrap": {"replicates": args.replicates, "seed": args.seed, "ci": "percentile 95%; patients, genomic blocks, and joint hierarchical patient+block resampling"}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
