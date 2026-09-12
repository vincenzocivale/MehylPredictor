"""Registry for the paper-facing functional mean-proxy ablation.

All three arms use the exact J0 functional-locus architecture and official chr1
protocol. The only causal factors are mean-proxy supervision and branch
presence. Fresh study/run IDs prevent reuse of historical shared-backbone
mean-contribution results.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY = "mean_contribution_functional_2026_09"
ENGINE = "matched_chr1_shared_backbone"
SEEDS = (17, 29, 43)


def engine_for(scope: str) -> str:
    if scope != "chr1":
        raise SystemExit(
            "functional mean-proxy ablation is currently frozen for chr1 only"
        )
    return ENGINE


def seeds_for(scope: str) -> tuple[int, ...]:
    if scope != "chr1":
        raise SystemExit(
            "functional mean-proxy ablation is currently frozen for chr1 only"
        )
    return SEEDS


@dataclass(frozen=True)
class Arm:
    name: str
    recipe: str
    question: str
    run_id_template: str

    def run_id(self, seed: int, scope: str = "chr1") -> str:
        if scope != "chr1":
            raise ValueError("functional mean-proxy ablation is chr1-only")
        return self.run_id_template.format(seed=seed)


ARMS = (
    Arm(
        "full_reference",
        "configs/models/main.yaml",
        "Full functional-locus J0 with mean-proxy supervision.",
        "functional-mean-full-seed{seed}",
    ),
    Arm(
        "no_mean_supervision",
        "configs/models/mean_contribution/no_mean_supervision.yaml",
        "Identical J0 architecture/capacity with aux_weight=0.",
        "functional-mean-no-supervision-seed{seed}",
    ),
    Arm(
        "no_mean_branch",
        "configs/models/mean_contribution/no_mean_branch.yaml",
        "J0 with the training-only mean-proxy head removed.",
        "functional-mean-no-branch-seed{seed}",
    ),
)
ARMS_BY_NAME = {arm.name: arm for arm in ARMS}


def select(names: str | None) -> tuple[Arm, ...]:
    if not names:
        return ARMS
    wanted = [x.strip() for x in names.split(",") if x.strip()]
    unknown = [x for x in wanted if x not in ARMS_BY_NAME]
    if unknown:
        raise SystemExit(
            f"unknown arm(s): {', '.join(unknown)}; "
            f"known: {', '.join(ARMS_BY_NAME)}"
        )
    return tuple(ARMS_BY_NAME[x] for x in wanted)


def data_paths(
    data_root: str | None = None,
    scope: str = "chr1",
) -> dict[str, str]:
    if scope != "chr1":
        raise SystemExit(
            "functional mean-proxy ablation is currently frozen for chr1 only"
        )

    root = Path(
        data_root
        or os.environ.get(
            "METHYL_DATA_ROOT",
            "/dune/DATASETS/MethylPredictionData",
        )
    )
    canonical = root / "datasets" / "methylprophet_repro_v1"
    prepared = root / "derived" / "methylprophet_table5_tcga_chr1"

    return {
        "data_root": str(root),
        "scope": "chr1",
        "canonical_root": str(canonical),
        "prepared_root": str(prepared),
        "prior_cache": str(prepared / "features"),
        "rna_cache": str(prepared / "rna"),
        "registry": str(
            canonical / "cpg" / "registries" / "array_cpg_map.parquet"
        ),
        "cpg_targets_dir": str(
            root / "derived" / "cpg_statistics" / "chr1"
        ),
        "functional_atlas": str(
            root / "derived" / "ntv3_functional_peak_atlas_chr1_all_sources"
        ),
        "annotation_cache": str(
            root / "derived" / "ntv3_probe_targets"
            / "chr1_annotation_features_all_sources"
        ),
        "output_root": str(root / "experiments"),
        "analysis_root": str(
            root / "experiments" / "analysis" / STUDY
        ),
    }
