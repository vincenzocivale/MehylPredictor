from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "rna_branch" / "evaluate_chr1_biological_fidelity.py"
spec = importlib.util.spec_from_file_location("evaluate_chr1_bio", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_methylprophet_alignment_is_exact_and_order_independent(tmp_path: Path) -> None:
    sample_ids = np.array(["S1", "S2"])
    cpg_ids = np.array(["chr1_10", "chr1_20"])
    target = np.array([[0.1, 0.2], [0.3, np.nan]])
    rows = pd.DataFrame(
        {
            "group_idx": [7, 7, 7],
            "sample_idx": [1, 0, 0],
            "cpg_idx": [0, 1, 0],
            "pred_methyl": [0.31, 0.21, 0.11],
            "gt_methyl": [0.3, 0.2, 0.1],
        }
    )
    pred_path = tmp_path / "mp.csv"
    rows.to_csv(pred_path, index=False)
    sample_map = {"0": "S1", "1": "S2"}
    cpg_map = {"0": "chr1_10", "1": "chr1_20"}
    matrix, audit = module._load_methylprophet_matrix(
        pred_path,
        canonical_sample_ids=sample_ids,
        canonical_cpg_ids=cpg_ids,
        canonical_target=target,
        group_column="group_idx",
        group_idx=7,
        sample_column="sample_idx",
        cpg_column="cpg_idx",
        prediction_column="pred_methyl",
        target_column="gt_methyl",
        sample_mapping=sample_map,
        cpg_mapping=cpg_map,
        allow_partial_overlap=False,
        target_tolerance=1e-6,
    )
    np.testing.assert_allclose(matrix[0], [0.11, 0.21])
    assert matrix[1, 0] == pytest.approx(0.31)
    assert np.isnan(matrix[1, 1])
    assert audit["coverage_fraction_observed"] == 1.0


def test_strict_alignment_rejects_missing_observed_cell(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "sample_idx": ["S1"],
            "cpg_idx": ["C1"],
            "pred_methyl": [0.1],
            "gt_methyl": [0.1],
        }
    )
    path = tmp_path / "mp.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="lacks 1 observed cells"):
        module._load_methylprophet_matrix(
            path,
            canonical_sample_ids=np.array(["S1"]),
            canonical_cpg_ids=np.array(["C1", "C2"]),
            canonical_target=np.array([[0.1, 0.2]]),
            group_column="group_idx",
            group_idx=None,
            sample_column="sample_idx",
            cpg_column="cpg_idx",
            prediction_column="pred_methyl",
            target_column="gt_methyl",
            sample_mapping=None,
            cpg_mapping=None,
            allow_partial_overlap=False,
            target_tolerance=1e-6,
        )


def test_hierarchical_bootstrap_favors_exact_model() -> None:
    rng = np.random.default_rng(4)
    target = rng.uniform(0.1, 0.9, size=(12, 8))
    prior = target.mean(axis=0)
    ours = target.copy()
    methylprophet = prior[None, :] + 0.1 * (target - prior[None, :])
    cancer = np.array(["A"] * 6 + ["B"] * 6)
    blocks = np.array(["chr1:0"] * 4 + ["chr1:1"] * 4)
    result = module.paired_hierarchical_bootstrap(
        target,
        ours,
        methylprophet,
        prior,
        cancer,
        blocks,
        np.ones(8, dtype=bool),
        replicates=20,
        seed=9,
    )
    assert result["mas_skill_vs_prior_variable"]["probability_difference_gt_zero"] == 1.0
    assert result["mas_ccc_variable"]["probability_difference_gt_zero"] == 1.0


def test_region_annotation_alignment_is_strict(tmp_path: Path) -> None:
    annotation = pd.DataFrame({"cpg_id": ["C2", "C1"], "region_id": ["R2", "R1"]})
    path = tmp_path / "regions.tsv"
    annotation.to_csv(path, sep="\t", index=False)
    regions, audit = module._region_ids(
        path, np.array(["C1", "C2", "C3"]),
        id_column="cpg_id", region_column="region_id",
    )
    assert regions.tolist() == ["R1", "R2", ""]
    assert audit == {"annotated_cpgs": 2, "unannotated_cpgs": 1, "unique_regions": 2}
