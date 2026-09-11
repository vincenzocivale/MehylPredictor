"""Registry for the functional-matched RNA encoder comparison."""
from __future__ import annotations

from dataclasses import dataclass


STUDY = "functional_rna_encoder_comparison_2026_09"
SEEDS = (17, 29, 43)


@dataclass(frozen=True)
class Arm:
    name: str
    recipe: str
    cache_kind: str
    run_id_template: str

    def run_id(self, seed: int) -> str:
        return self.run_id_template.format(seed=seed)


ARMS = (
    Arm(
        "ours_program_tokens",
        "configs/models/rna_encoder_comparison/"
        "functional_ours_program_tokens.yaml",
        "canonical",
        "functional-rnaenc-ours-seed{seed}",
    ),
    Arm(
        "bottleneck_mlp",
        "configs/models/rna_encoder_comparison/"
        "functional_bottleneck_mlp.yaml",
        "canonical",
        "functional-rnaenc-bottleneck-mlp-seed{seed}",
    ),
    Arm(
        "gene_pathway",
        "configs/models/rna_encoder_comparison/"
        "functional_gene_pathway.yaml",
        "canonical",
        "functional-rnaenc-gene-pathway-seed{seed}",
    ),
    Arm(
        "bulkformer_147m",
        "configs/models/rna_encoder_comparison/"
        "functional_bulkformer_147m.yaml",
        "bulkformer_147m",
        "functional-rnaenc-bulkformer-147m-seed{seed}",
    ),
    Arm(
        "bulkrnabert",
        "configs/models/rna_encoder_comparison/"
        "functional_bulkrnabert.yaml",
        "bulkrnabert",
        "functional-rnaenc-bulkrnabert-seed{seed}",
    ),
)
