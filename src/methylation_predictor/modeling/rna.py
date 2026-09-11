"""RNA representation for the paper-facing methylation predictor.

The public candidate models need only one operation from RNA before locus
conditioning: convert the full transcriptome into a bank of learned program
tokens. Retrieval is implemented separately in ``modeling.retrieval``.
"""

from __future__ import annotations

import torch
from torch import nn


LEGACY_UNUSED_RNA_ATTENTION_PREFIXES = (
    "rna_encoder.query.",
    "rna_encoder.key.",
    "rna_encoder.value.",
    "rna_encoder.out.",
    "rna_encoder.mean_query.",
)


class ProgramTokenEncoder(nn.Module):
    """Factorized transcriptome -> learned program-token encoder."""

    def __init__(
        self,
        input_dim: int,
        n_programs: int,
        program_dim: int,
        bottleneck_dim: int = 256,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.n_programs = int(n_programs)
        self.program_dim = int(program_dim)

        self.norm = nn.LayerNorm(input_dim) if layer_norm else nn.Identity()
        self.bottleneck = nn.Linear(input_dim, bottleneck_dim)
        self.bottleneck_norm = nn.LayerNorm(bottleneck_dim)

        self.program_basis = nn.Parameter(
            torch.empty(n_programs, program_dim, bottleneck_dim)
        )
        nn.init.normal_(self.program_basis, std=bottleneck_dim ** -0.5)

        self.token_norm = nn.LayerNorm(program_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.bottleneck_norm(
            torch.nn.functional.gelu(self.bottleneck(self.norm(x)))
        )
        return self.token_norm(
            torch.einsum("br,kdr->bkd", shared, self.program_basis)
        )


def consume_legacy_attention_initialization(width: int) -> None:
    """Consume RNG draws used by the retired RNA-internal Q/K/V/out layers.

    This preserves same-seed initialization of every later live parameter
    during the behavior-preserving refactor, without registering dead weights.
    """
    for _ in range(4):
        nn.Linear(width, width)


def migrate_legacy_candidate_state_dict(state_dict):
    """Drop only RNA-attention keys proven unused by J0/J1."""
    migrated = state_dict.copy()
    for key in list(migrated.keys()):
        if key.startswith(LEGACY_UNUSED_RNA_ATTENTION_PREFIXES):
            migrated.pop(key)
    return migrated
