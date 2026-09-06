"""Registry for the paper-facing mean-contribution experiment."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY = "mean_contribution_2026_09"
ENGINE = "matched_chr1_shared_backbone"
SEEDS = (17, 29, 43)


@dataclass(frozen=True)
class Arm:
    name: str
    recipe: str
    question: str
    run_id_template: str

    def run_id(self, seed: int) -> str:
        return self.run_id_template.format(seed=seed)


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


def data_paths(data_root: str | None = None) -> dict[str, str]:
    root = Path(data_root or os.environ.get("METHYL_DATA_ROOT", "/dune/DATASETS/MethylPredictionData"))
    prepared = root / "derived" / "methylprophet_table5_tcga_chr1"
    canonical = root / "datasets" / "methylprophet_repro_v1"
    return {
        "data_root": str(root),
        "canonical_root": str(canonical),
        "prepared_root": str(prepared),
        "feature_cache": str(prepared / "features"),
        "rna_cache": str(prepared / "rna"),
        "registry": str(canonical / "cpg" / "registries" / "array_cpg_map.parquet"),
        "cpg_targets_dir": str(root / "derived" / "cpg_statistics" / "chr1"),
        "output_root": str(root / "experiments"),
        "analysis_root": str(root / "experiments" / "analysis" / STUDY),
    }
