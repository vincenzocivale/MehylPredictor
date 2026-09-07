"""Registry for the paper-facing mean-contribution experiment.

Primary scope is chr1 (3 seeds: 17, 29, 43), matched to MethylProphet's own
protocol via ``matched_chr1_shared_backbone``. A single-seed (17) chr123
follow-up reuses the exact same arms/recipes with the generic
``shared_backbone`` engine and the chr123 compact cache
(``scripts/prepare_chr123_compact.py``) -- see docs/MEAN_CONTRIBUTION_EXPERIMENTS.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY = "mean_contribution_2026_09"

ENGINE_BY_SCOPE = {
    "chr1": "matched_chr1_shared_backbone",
    "chr123": "shared_backbone",
}
SEEDS_BY_SCOPE = {
    "chr1": (17, 29, 43),
    "chr123": (17,),
}
# Kept for backward compatibility with existing chr1-only call sites.
ENGINE = ENGINE_BY_SCOPE["chr1"]
SEEDS = SEEDS_BY_SCOPE["chr1"]


def engine_for(scope: str) -> str:
    if scope not in ENGINE_BY_SCOPE:
        raise SystemExit(f"unknown scope: {scope}; known: {', '.join(ENGINE_BY_SCOPE)}")
    return ENGINE_BY_SCOPE[scope]


def seeds_for(scope: str) -> tuple[int, ...]:
    if scope not in SEEDS_BY_SCOPE:
        raise SystemExit(f"unknown scope: {scope}; known: {', '.join(SEEDS_BY_SCOPE)}")
    return SEEDS_BY_SCOPE[scope]


@dataclass(frozen=True)
class Arm:
    name: str
    recipe: str
    question: str
    run_id_template: str

    def run_id(self, seed: int, scope: str = "chr1") -> str:
        base = self.run_id_template.format(seed=seed)
        # chr1 run-ids are unchanged (already collected); chr123 gets an explicit
        # suffix, matching the convention used by results/reference/ours/04_baselines.yaml's
        # chr123_arms.
        return base if scope == "chr1" else f"{base}-{scope}"


ARMS = (
    Arm(
        "full_reference",
        "configs/models/rna_methylation_locus_attention.yaml",
        "Full locus-attention reference with biological mean proxy supervision.",
        "ref-locus-attn-k64-noproduct-seed{seed}",
    ),
    Arm(
        "no_mean_supervision",
        "configs/models/mean_contribution/no_mean_supervision.yaml",
        "Same architecture/capacity as reference, aux_weight=0.",
        "mean-no-supervision-seed{seed}",
    ),
    Arm(
        "no_mean_branch",
        "configs/models/mean_contribution/no_mean_branch.yaml",
        "Remove h_mean entirely to isolate branch capacity.",
        "mean-no-branch-seed{seed}",
    ),
)
ARMS_BY_NAME = {arm.name: arm for arm in ARMS}


def select(names: str | None) -> tuple[Arm, ...]:
    if not names:
        return ARMS
    wanted = [x.strip() for x in names.split(",") if x.strip()]
    unknown = [x for x in wanted if x not in ARMS_BY_NAME]
    if unknown:
        raise SystemExit(f"unknown arm(s): {', '.join(unknown)}; known: {', '.join(ARMS_BY_NAME)}")
    return tuple(ARMS_BY_NAME[x] for x in wanted)


def data_paths(data_root: str | None = None, scope: str = "chr1") -> dict[str, str]:
    root = Path(data_root or os.environ.get("METHYL_DATA_ROOT", "/dune/DATASETS/MethylPredictionData"))
    canonical = root / "datasets" / "methylprophet_repro_v1"
    # rna-cache and registry are genome/scope-independent (see [[data-root-this-machine]]) and are
    # reused as-is across scopes; only prepared-root/feature-cache/cpg-targets-dir vary by scope.
    rna_cache = root / "derived" / "methylprophet_table5_tcga_chr1" / "rna"
    registry = canonical / "cpg" / "registries" / "array_cpg_map.parquet"

    if scope == "chr1":
        prepared = root / "derived" / "methylprophet_table5_tcga_chr1"
        feature_cache = prepared / "features"
        cpg_targets_dir = root / "derived" / "cpg_statistics" / "chr1"
    elif scope == "chr123":
        prepared = root / "derived" / "methylprophet_compact_chr123"
        feature_cache = root / "derived" / "rna_feature_cache" / "chr123"
        cpg_targets_dir = root / "derived" / "cpg_statistics" / "chr123"
    else:
        raise SystemExit(f"unknown scope: {scope}; known: chr1, chr123")

    return {
        "data_root": str(root),
        "scope": scope,
        "canonical_root": str(canonical),
        "prepared_root": str(prepared),
        "feature_cache": str(feature_cache),
        "rna_cache": str(rna_cache),
        "registry": str(registry),
        "cpg_targets_dir": str(cpg_targets_dir),
        "output_root": str(root / "experiments"),
        "analysis_root": str(root / "experiments" / "analysis" / f"{STUDY}_{scope}" if scope != "chr1"
                              else root / "experiments" / "analysis" / STUDY),
    }
