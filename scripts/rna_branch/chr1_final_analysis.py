#!/usr/bin/env python3
"""Final high-value post-hoc analyses for the chr1 matched test.

Reads ONLY the already-produced report artifacts under
`artifacts/rna_branch/chr1_biological_fidelity/test_comparison/`
(`matched_test_matrices.npz`, `*_per_cpg.parquet`, `biological_fidelity_report.json`)
plus one external, frozen, model-independent CpG-island annotation
(`/data/dataset/methylation/MethylProphetData/parquet/241231-tcga_array/cpg_island.parquet`).
Does not reload the model, the checkpoint, or MethylProphet's predictions --
purely a post-hoc breakdown of what evaluate_chr1_biological_fidelity.py
already computed and saved.

Produces `final_analysis.json` and `mse_contribution_by_cpg.csv` in the same
output directory.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EVALUATOR_SCRIPT = Path(__file__).parent / "evaluate_chr1_biological_fidelity.py"
ISLAND_ANNOTATION = Path(
    "/data/dataset/methylation/MethylProphetData/parquet/241231-tcga_array/cpg_island.parquet"
)


def _load_evaluator_module():
    spec = importlib.util.spec_from_file_location("evaluate_chr1_bio", EVALUATOR_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quantiles(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {}
    q = np.quantile(x, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return dict(zip(("min", "q10", "q25", "median", "q75", "q90", "max"), map(float, q), strict=True))


def _win_fraction(diff: np.ndarray, tol: float = 1e-9) -> dict[str, Any]:
    diff = diff[np.isfinite(diff)]
    ours_wins = int((diff > tol).sum())
    mp_wins = int((diff < -tol).sum())
    ties = int(len(diff) - ours_wins - mp_wins)
    n = max(len(diff), 1)
    return {
        "n_cpgs": len(diff),
        "ours_wins": ours_wins, "ours_win_fraction": ours_wins / n,
        "methylprophet_wins": mp_wins, "methylprophet_win_fraction": mp_wins / n,
        "ties": ties, "tie_fraction": ties / n,
    }


def _variance_bin_breakdown(target_std: np.ndarray, diff: np.ndarray, n_bins: int = 3) -> dict[str, Any]:
    valid = np.isfinite(target_std) & np.isfinite(diff)
    ts, d = target_std[valid], diff[valid]
    if len(ts) < n_bins:
        return {}
    labels = ["low", "mid", "high"] if n_bins == 3 else [f"bin{i}" for i in range(n_bins)]
    edges = np.quantile(ts, np.linspace(0, 1, n_bins + 1))
    bin_idx = np.clip(np.digitize(ts, edges[1:-1]), 0, n_bins - 1)
    out = {}
    for i, name in enumerate(labels):
        sel = bin_idx == i
        out[name] = {
            "n_cpgs": int(sel.sum()),
            "target_std_range": [float(ts[sel].min()), float(ts[sel].max())] if sel.any() else None,
            "mean_skill_diff": float(np.mean(d[sel])) if sel.any() else float("nan"),
            "median_skill_diff": float(np.median(d[sel])) if sel.any() else float("nan"),
            "ours_win_fraction": float(np.mean(d[sel] > 0)) if sel.any() else float("nan"),
        }
    return out


def _chr_pos_lookup(coordinates_path: Path) -> dict[str, str]:
    """cpg_idx (string) -> "chr_pos" string, needed because our per_cpg tables
    key on the bare integer cpg_idx while MethylProphet's island annotation
    keys on "chr1_12345"-style chr_pos strings."""
    frame = pd.read_parquet(coordinates_path)
    return dict(zip(frame["cpg_idx"].astype(str), frame["chr_pos"].astype(str), strict=True))


def _island_shore_shelf_opensea(cpg_ids: np.ndarray, coordinates_path: Path) -> pd.Series:
    """Consolidated island/shore/shelf/opensea label per cpg_id (our bare
    integer cpg_idx, as a string), from MethylProphet's own frozen island
    annotation joined via chr_pos. cgi=island, upshore1..4=shore, shelve=shelf,
    sea=opensea."""
    chr_pos = _chr_pos_lookup(coordinates_path)
    frame = pd.read_parquet(ISLAND_ANNOTATION)
    mapping = dict(zip(frame["cpg"].astype(str), frame["location"].astype(str), strict=True))
    consolidate = {
        "cgi": "island", "sea": "opensea", "shelve": "shelf",
        "upshore1": "shore", "upshore2": "shore", "upshore3": "shore", "upshore4": "shore",
    }
    raw = pd.Series([mapping.get(chr_pos.get(c)) for c in cpg_ids], index=cpg_ids)
    return raw.map(consolidate)


def _region_breakdown(region_label: pd.Series, diff: np.ndarray, cpg_ids: np.ndarray) -> dict[str, Any]:
    out = {}
    diff_by_id = dict(zip(cpg_ids, diff, strict=True))
    for region in sorted(region_label.dropna().unique()):
        members = region_label[region_label == region].index
        values = np.array([diff_by_id[c] for c in members if c in diff_by_id], dtype=np.float64)
        values = values[np.isfinite(values)]
        out[region] = {
            "n_cpgs": int(len(values)),
            "mean_skill_diff": float(np.mean(values)) if len(values) else float("nan"),
            "median_skill_diff": float(np.median(values)) if len(values) else float("nan"),
            "ours_win_fraction": float(np.mean(values > 0)) if len(values) else float("nan"),
        }
    out["_unannotated_cpgs"] = int(region_label.isna().sum())
    return out


def _mse_contribution_by_cpg(
    target: np.ndarray, ours: np.ndarray, mp: np.ndarray, cpg_ids: np.ndarray
) -> pd.DataFrame:
    """Per-CpG contribution to the global SSE gap (MP SSE - ours SSE, summed
    over observed samples): positive = that CpG favors ours; the sum of this
    column, divided by total observed cells, reconciles exactly with
    mse_methylprophet - mse_ours from the headline table."""
    valid = np.isfinite(target) & np.isfinite(ours) & np.isfinite(mp)
    ours_sse = np.where(valid, (ours - target) ** 2, 0.0).sum(axis=0)
    mp_sse = np.where(valid, (mp - target) ** 2, 0.0).sum(axis=0)
    counts = valid.sum(axis=0)
    contribution = mp_sse - ours_sse
    frame = pd.DataFrame({
        "cpg_id": cpg_ids,
        "observed_samples": counts,
        "ours_sse": ours_sse,
        "methylprophet_sse": mp_sse,
        "sse_contribution_favoring_ours": contribution,
    }).sort_values("sse_contribution_favoring_ours", ascending=False).reset_index(drop=True)
    return frame


def _global_metrics(target: np.ndarray, prediction: np.ndarray, prior: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(target) & np.isfinite(prediction)
    prior_matrix = np.broadcast_to(prior[None, :], target.shape)
    model_sse = np.where(valid, (prediction - target) ** 2, 0.0).sum()
    prior_sse = np.where(valid, (prior_matrix - target) ** 2, 0.0).sum()
    n = max(valid.sum(), 1)
    return {
        "mse": float(model_sse / n),
        "skill_vs_prior": float(1.0 - model_sse / max(prior_sse, 1e-12)),
    }


def _global_paired_bootstrap(
    target: np.ndarray, ours: np.ndarray, mp: np.ndarray, prior: np.ndarray,
    cancer_types: np.ndarray, block_labels: np.ndarray,
    *, replicates: int, seed: int, evaluator_module,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    unique_blocks = np.unique(block_labels)
    block_to_columns = {block: np.flatnonzero(block_labels == block) for block in unique_blocks}
    delta_mse = np.empty(replicates)
    delta_skill = np.empty(replicates)
    for replicate in range(replicates):
        rows = evaluator_module._stratified_sample_bootstrap(cancer_types, rng)
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        columns = np.concatenate([block_to_columns[b] for b in sampled_blocks])
        t = target[np.ix_(rows, columns)]
        o = ours[np.ix_(rows, columns)]
        m = mp[np.ix_(rows, columns)]
        p = prior[columns]
        gm_ours = _global_metrics(t, o, p)
        gm_mp = _global_metrics(t, m, p)
        delta_mse[replicate] = gm_ours["mse"] - gm_mp["mse"]
        delta_skill[replicate] = gm_ours["skill_vs_prior"] - gm_mp["skill_vs_prior"]
    result = {"replicates": replicates, "seed": seed, "resampling": "cancer-stratified patients x genomic blocks"}
    for name, values in (("delta_mse_ours_minus_mp", delta_mse), ("delta_skill_vs_prior_ours_minus_mp", delta_skill)):
        result[name] = {
            "mean": float(np.mean(values)),
            "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
            "probability_ours_better": float(np.mean(values < 0)) if "mse" in name else float(np.mean(values > 0)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-comparison-dir", type=Path, required=True)
    parser.add_argument(
        "--cpg-coordinates", type=Path,
        default=Path("artifacts/methylprophet_audit/cpg_chr_pos_chr1_6742.parquet"),
    )
    parser.add_argument("--genomic-block-size", type=int, default=5_000_000)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260805)
    args = parser.parse_args()

    evaluator_module = _load_evaluator_module()
    out_dir = args.test_comparison_dir

    npz = np.load(out_dir / "matched_test_matrices.npz", allow_pickle=True)
    target, ours, mp = npz["target"].astype(np.float64), npz["ours"].astype(np.float64), npz["methylprophet"].astype(np.float64)
    ours_cal = npz["ours_calibrated"].astype(np.float64) if "ours_calibrated" in npz else None
    mp_cal = npz["methylprophet_calibrated"].astype(np.float64) if "methylprophet_calibrated" in npz else None
    prior = npz["prior"].astype(np.float64)
    cpg_ids = npz["cpg_idx"].astype(str)
    cancer_types = npz["cancer_type"].astype(str)

    ours_locus = pd.read_parquet(out_dir / "ours_per_cpg.parquet")
    mp_locus = pd.read_parquet(out_dir / "methylprophet_per_cpg.parquet")
    report = json.loads((out_dir / "biological_fidelity_report.json").read_text())

    merged = ours_locus.merge(mp_locus, on="cpg_id", suffixes=("_ours", "_mp"))
    eligible = merged["eligible_variable_cpg_ours"].to_numpy(dtype=bool)
    assert np.array_equal(eligible, merged["eligible_variable_cpg_mp"].to_numpy(dtype=bool))
    elig = merged[eligible].reset_index(drop=True)
    skill_diff = (elig["skill_vs_prior_ours"] - elig["skill_vs_prior_mp"]).to_numpy(dtype=np.float64)

    analysis: dict[str, Any] = {"scope": {"eligible_variable_cpgs": int(eligible.sum()), "total_test_cpgs": int(len(merged))}}

    # 1) distribution of S_i^ours - S_i^MP (raw, eligible CpGs)
    analysis["skill_diff_distribution_raw"] = {
        "mean": float(np.mean(skill_diff)), "std": float(np.std(skill_diff)),
        **_quantiles(skill_diff),
    }

    # 2) win fraction
    analysis["win_fraction_raw"] = _win_fraction(skill_diff)

    # 3) per variance-tertile breakdown
    analysis["variance_tertile_breakdown_raw"] = _variance_bin_breakdown(
        elig["target_std_ours"].to_numpy(dtype=np.float64), skill_diff
    )

    # 4) island/shore/shelf/opensea breakdown
    region_label = _island_shore_shelf_opensea(elig["cpg_id"].to_numpy(dtype=str), args.cpg_coordinates)
    analysis["island_shore_shelf_opensea_breakdown_raw"] = _region_breakdown(
        region_label, skill_diff, elig["cpg_id"].to_numpy(dtype=str)
    )

    # 5) DMP effect size (already computed, raw + calibrated) + island-cluster DMR
    analysis["dmp_effect_size"] = {
        "ours_raw": report["ours"]["differential_effect_recovery"]["aggregate"],
        "methylprophet_raw": report["methylprophet"]["differential_effect_recovery"]["aggregate"],
        "ours_calibrated": (report.get("ours_calibrated") or {}).get("differential_effect_recovery", {}).get("aggregate"),
        "methylprophet_calibrated": (report.get("methylprophet_calibrated") or {}).get("differential_effect_recovery", {}).get("aggregate"),
    }
    chr_pos = _chr_pos_lookup(args.cpg_coordinates)
    island_frame = pd.read_parquet(ISLAND_ANNOTATION)
    island_index_by_cpg = dict(zip(island_frame["cpg"].astype(str), island_frame["cgiIndex"], strict=True))
    cgi_region_ids = np.array([
        (
            f"cgi_{int(island_index_by_cpg[chr_pos[c]])}"
            if c in chr_pos and chr_pos[c] in island_index_by_cpg and np.isfinite(island_index_by_cpg[chr_pos[c]])
            else ""
        )
        for c in cpg_ids
    ], dtype=object)
    from methylation_predictor.rna_branch.biological_metrics import BiologicalMetricConfig, regional_effect_recovery
    dmr_config = BiologicalMetricConfig(min_cancer_group_samples=4)
    analysis["dmr_island_cluster_effect_size_raw"] = {
        "note": "DMR proxy restricted to CpGs within an annotated island cluster (region_id=cgiIndex); NOT a genome-wide DMR set.",
        "ours": regional_effect_recovery(target, ours, cancer_types, cgi_region_ids, min_cpgs_per_region=2, config=dmr_config),
        "methylprophet": regional_effect_recovery(target, mp, cancer_types, cgi_region_ids, min_cpgs_per_region=2, config=dmr_config),
    }

    # 6) within-cancer CCC and dynamic R2 (already computed; tabulate raw+calibrated)
    def _wc(model_report: dict | None) -> dict | None:
        if not model_report:
            return None
        return {
            "within_cancer_mas_pcc_variable": model_report.get("within_cancer_mas_pcc_variable"),
            "within_cancer_mas_ccc_variable": model_report.get("within_cancer_mas_ccc_variable"),
            "within_cancer_mas_dynamic_r2_variable": model_report.get("within_cancer_mas_dynamic_r2_variable"),
        }
    analysis["within_cancer_ccc_and_dynamic_r2"] = {
        "ours_raw": _wc(report["ours"]), "methylprophet_raw": _wc(report["methylprophet"]),
        "ours_calibrated": _wc(report.get("ours_calibrated")),
        "methylprophet_calibrated": _wc(report.get("methylprophet_calibrated")),
    }

    # 7) per-CpG contribution to the global MSE advantage
    contribution = _mse_contribution_by_cpg(target, ours, mp, cpg_ids)
    contribution.to_csv(out_dir / "mse_contribution_by_cpg.csv", index=False)
    total_gap = float(contribution["sse_contribution_favoring_ours"].sum())
    n_observed = int(np.isfinite(target).sum())
    top_k = 15
    top_favoring_ours = contribution.head(top_k)[["cpg_id", "sse_contribution_favoring_ours"]].to_dict("records")
    top_favoring_mp = contribution.tail(top_k)[["cpg_id", "sse_contribution_favoring_ours"]].to_dict("records")
    analysis["mse_contribution_by_cpg"] = {
        "csv": "mse_contribution_by_cpg.csv",
        "total_sse_gap_favoring_ours": total_gap,
        "reconciles_to_mse_gap": total_gap / n_observed,
        "headline_mse_gap_mp_minus_ours": float(report["methylprophet"]["mse"] - report["ours"]["mse"]),
        f"top_{top_k}_cpgs_favoring_ours": top_favoring_ours,
        f"top_{top_k}_cpgs_favoring_methylprophet": top_favoring_mp,
        "cumulative_share_top_20_cpgs_of_total_abs_gap": float(
            contribution["sse_contribution_favoring_ours"].abs().sort_values(ascending=False).head(20).sum()
            / contribution["sse_contribution_favoring_ours"].abs().sum()
        ),
    }

    # 8) paired bootstrap of GLOBAL mse and GLOBAL skill_vs_prior
    blocks = evaluator_module._coordinate_blocks(
        cpg_ids, args.cpg_coordinates,
        id_column="cpg_idx", chromosome_column="chr", position_column="pos",
        block_size=args.genomic_block_size,
    )
    analysis["global_paired_bootstrap"] = {
        "raw": _global_paired_bootstrap(
            target, ours, mp, prior, cancer_types, blocks,
            replicates=args.bootstrap_replicates, seed=args.bootstrap_seed, evaluator_module=evaluator_module,
        ),
    }
    if ours_cal is not None:
        analysis["global_paired_bootstrap"]["calibrated_vs_mp_raw"] = _global_paired_bootstrap(
            target, ours_cal, mp, prior, cancer_types, blocks,
            replicates=args.bootstrap_replicates, seed=args.bootstrap_seed, evaluator_module=evaluator_module,
        )
    if ours_cal is not None and mp_cal is not None:
        analysis["global_paired_bootstrap"]["calibrated_vs_mp_calibrated"] = _global_paired_bootstrap(
            target, ours_cal, mp_cal, prior, cancer_types, blocks,
            replicates=args.bootstrap_replicates, seed=args.bootstrap_seed, evaluator_module=evaluator_module,
        )

    (out_dir / "final_analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=False, default=str) + "\n")
    print(f"Wrote {out_dir / 'final_analysis.json'}")
    print(f"Wrote {out_dir / 'mse_contribution_by_cpg.csv'}")


if __name__ == "__main__":
    main()
