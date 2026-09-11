"""Registry for functional-matched chr1 paper baselines."""
from __future__ import annotations

from dataclasses import dataclass

STUDY = "functional_baselines_2026_09"
SEEDS = (17, 29, 43)


@dataclass(frozen=True)
class Arm:
    name: str
    recipe: str
    run_id_template: str
    question: str

    def run_id(self, seed: int) -> str:
        return self.run_id_template.format(seed=seed)


ARMS = (
    Arm(
        "global_rna_shift",
        "configs/models/baselines/functional_global_rna_shift.yaml",
        "functional-baseline-global-shift-seed{seed}",
        "Can patient-global RNA shifts explain methylation without "
        "patient-by-locus RNA interaction?",
    ),
    Arm(
        "mlp_rna_cpg",
        "configs/models/baselines/functional_mlp_rna_cpg.yaml",
        "functional-baseline-mlp-seed{seed}",
        "Is generic pooled-RNA + functional-locus MLP fusion sufficient?",
    ),
    Arm(
        "bilinear_rna_cpg",
        "configs/models/baselines/functional_bilinear_rna_cpg.yaml",
        "functional-baseline-bilinear-seed{seed}",
        "Is a simple multiplicative interaction sufficient without "
        "locus-conditioned token retrieval?",
    ),
)
