#!/usr/bin/env bash
set -euo pipefail

BRANCH_EXPECTED="refactor/repo-v2-2026-09"

if [[ "${1:-}" == "--dry-run" ]]; then
  cat <<'EOF'
Phase 2b will:
  + add src/methylation_predictor/modeling/retrieval.py
  ~ replace modeling/reference.py aliases with real J0/J1 implementations
  ~ route J0/J1 production trainer dispatch through modeling/
  ~ strengthen refactor tests with exact old<->new state_dict/forward equivalence
  ~ update docs/REFACTORING_CONTRACT.md

Historical implementations remain temporarily in models.py for compatibility.
No historical architecture code is deleted in this phase.
EOF
  exit 0
fi

branch="$(git branch --show-current)"
if [[ "${branch}" != "${BRANCH_EXPECTED}" ]]; then
  echo "ERROR: expected ${BRANCH_EXPECTED}, got ${branch}" >&2
  exit 1
fi

if [[ -e src/methylation_predictor/modeling/retrieval.py ]]; then
  echo "ERROR: src/methylation_predictor/modeling/retrieval.py already exists; refusing to overwrite." >&2
  exit 1
fi

mkdir -p src/methylation_predictor/modeling

cat > src/methylation_predictor/modeling/retrieval.py <<'PY'
"""Locus-conditioned retrieval primitives used by the paper candidates."""

from __future__ import annotations

import torch
from torch import nn


class FeedForwardResidual(nn.Module):
    """Pre-norm 4x-expansion FFN with an internal residual."""

    def __init__(self, width: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * width, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class LocusToRNAAttention(nn.Module):
    """Bare multi-head cross-attention from locus states to RNA program tokens."""

    def __init__(self, width: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if width % n_heads:
            raise ValueError("cross-attention width must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = width // n_heads
        self.q_norm = nn.LayerNorm(width)
        self.k_norm = nn.LayerNorm(width)
        self.v_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.out = nn.Linear(width, width)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, loci: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        batch, n_programs, width = tokens.shape
        n_loci = loci.shape[0]
        q = self.query(self.q_norm(loci)).view(
            n_loci, self.n_heads, self.head_dim
        )
        k = self.key(self.k_norm(tokens)).view(
            batch, n_programs, self.n_heads, self.head_dim
        )
        v = self.value(self.v_norm(tokens)).view(
            batch, n_programs, self.n_heads, self.head_dim
        )
        scores = torch.einsum("lhd,bkhd->bhlk", q, k) / (self.head_dim**0.5)
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        attended = torch.einsum("bhlk,bkhd->blhd", weights, v)
        return self.out(attended.reshape(batch, n_loci, width))


class BatchedLocusToRNAAttention(nn.Module):
    """Cross-attention for an already patient-conditioned locus query."""

    def __init__(self, width: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if width % n_heads:
            raise ValueError("cross-attention width must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = width // n_heads
        self.q_norm = nn.LayerNorm(width)
        self.k_norm = nn.LayerNorm(width)
        self.v_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.out = nn.Linear(width, width)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        batch, n_loci, width = query.shape
        n_programs = tokens.shape[1]
        q = self.query(self.q_norm(query)).view(
            batch, n_loci, self.n_heads, self.head_dim
        )
        k = self.key(self.k_norm(tokens)).view(
            batch, n_programs, self.n_heads, self.head_dim
        )
        v = self.value(self.v_norm(tokens)).view(
            batch, n_programs, self.n_heads, self.head_dim
        )
        scores = torch.einsum("blhd,bkhd->bhlk", q, k) / (self.head_dim**0.5)
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        attended = torch.einsum("bhlk,bkhd->blhd", weights, v)
        return self.out(attended.reshape(batch, n_loci, width))
PY

cat > src/methylation_predictor/modeling/reference.py <<'PY'
"""Paper-facing implementations of the current full-data candidates."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig
from ..models import build_rna_encoder
from .retrieval import (
    BatchedLocusToRNAAttention,
    FeedForwardResidual,
    LocusToRNAAttention,
)


class SingleRetrievalPredictor(nn.Module):
    """J0: functional locus + pure single-step RNA retrieval."""

    N_TRACKS = 4165
    DENSE_DIM = 23
    WIDTH = 256

    def __init__(
        self,
        input_dim: int,
        config: ModelConfig,
        *,
        final_regressor_dropout: float = 0.15,
    ):
        super().__init__()
        enc = config.encoder
        if enc.kind != "locus_attention" or enc.program_dim != self.WIDTH:
            raise ValueError(
                "SingleRetrievalPredictor requires locus_attention RNA tokens "
                "with program_dim=256"
            )

        self.rna_encoder = build_rna_encoder(
            enc, input_dim=input_dim, locus_dim=self.WIDTH
        )
        self.track_embedding = nn.EmbeddingBag(
            self.N_TRACKS,
            self.WIDTH,
            mode="mean",
            include_last_offset=True,
        )
        self.dense_encoder = nn.Sequential(
            nn.LayerNorm(self.DENSE_DIM),
            nn.Linear(self.DENSE_DIM, self.WIDTH),
            nn.GELU(),
        )
        self.locus_norm = nn.LayerNorm(self.WIDTH)
        self.cross_attention = LocusToRNAAttention(
            self.WIDTH, enc.n_heads, enc.dropout
        )
        self.mean_head = nn.Sequential(
            nn.LayerNorm(self.WIDTH),
            nn.Linear(self.WIDTH, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.h_c_concat_norm = nn.LayerNorm(self.WIDTH)
        self.r_pc_concat_norm = nn.LayerNorm(self.WIDTH)
        self.final_regressor_dropout = float(final_regressor_dropout)
        self.final_regressor = nn.Sequential(
            nn.Linear(2 * self.WIDTH, self.WIDTH),
            nn.GELU(),
            nn.Dropout(self.final_regressor_dropout),
            nn.Linear(self.WIDTH, 128),
            nn.GELU(),
            nn.Dropout(self.final_regressor_dropout),
            nn.Linear(128, 1),
        )

    @property
    def requires_cpg_positions(self) -> bool:
        return False

    def forward(
        self,
        rna: torch.Tensor,
        cpg_embedding: torch.Tensor | None = None,
        cpg_positions: torch.Tensor | None = None,
        *,
        functional_track_indices: torch.Tensor,
        functional_offsets: torch.Tensor,
        functional_dense: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del cpg_embedding, cpg_positions
        representation = self.rna_encoder(rna)
        if representation.program_tokens is None:
            raise RuntimeError("RNA encoder did not produce program tokens")
        tokens = representation.program_tokens

        peak = self.track_embedding(
            functional_track_indices, functional_offsets
        )
        dense = self.dense_encoder(functional_dense)
        h_c = self.locus_norm(peak + dense)
        mu_hat = torch.sigmoid(self.mean_head(h_c).squeeze(-1))

        r_pc = self.cross_attention(h_c, tokens)
        h_for_concat = self.h_c_concat_norm(h_c)
        r_for_concat = self.r_pc_concat_norm(r_pc)
        z_pc = torch.cat(
            [
                h_for_concat[None, :, :].expand(r_for_concat.shape[0], -1, -1),
                r_for_concat,
            ],
            dim=-1,
        )
        logit = self.final_regressor(z_pc).squeeze(-1)
        beta = torch.sigmoid(logit)
        return {
            "beta": beta,
            "delta_logit": logit,
            "raw_delta": logit,
            "prediction_logit": logit,
            "mu_logit": None,
            "mu_hat": mu_hat,
            "residual_logit": logit,
            "h_cpg": h_c,
        }


class IterativeRetrievalPredictor(nn.Module):
    """J1: four residual RNA-retrieval/refinement blocks."""

    N_TRACKS = 4165
    DENSE_DIM = 23
    WIDTH = 256
    N_BLOCKS = 4
    LOCUS_CHUNK = 1024

    def __init__(
        self,
        input_dim: int,
        config: ModelConfig,
        *,
        final_regressor_dropout: float = 0.15,
    ):
        super().__init__()
        enc = config.encoder
        if enc.kind != "locus_attention" or enc.program_dim != self.WIDTH:
            raise ValueError(
                "IterativeRetrievalPredictor requires locus_attention RNA "
                "tokens with program_dim=256"
            )

        self.rna_encoder = build_rna_encoder(
            enc, input_dim=input_dim, locus_dim=self.WIDTH
        )
        self.track_embedding = nn.EmbeddingBag(
            self.N_TRACKS,
            self.WIDTH,
            mode="mean",
            include_last_offset=True,
        )
        self.dense_encoder = nn.Sequential(
            nn.LayerNorm(self.DENSE_DIM),
            nn.Linear(self.DENSE_DIM, self.WIDTH),
            nn.GELU(),
        )
        self.locus_norm = nn.LayerNorm(self.WIDTH)
        self.mean_head = nn.Sequential(
            nn.LayerNorm(self.WIDTH),
            nn.Linear(self.WIDTH, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.retrieval_attn = nn.ModuleList(
            [
                BatchedLocusToRNAAttention(
                    self.WIDTH, enc.n_heads, enc.dropout
                )
                for _ in range(self.N_BLOCKS)
            ]
        )
        self.retrieval_ffn = nn.ModuleList(
            [
                FeedForwardResidual(self.WIDTH, enc.dropout)
                for _ in range(self.N_BLOCKS)
            ]
        )
        self.h_c_concat_norm = nn.LayerNorm(self.WIDTH)
        self.r_pc_concat_norm = nn.LayerNorm(self.WIDTH)
        self.final_regressor_dropout = float(final_regressor_dropout)
        self.final_regressor = nn.Sequential(
            nn.Linear(2 * self.WIDTH, self.WIDTH),
            nn.GELU(),
            nn.Dropout(self.final_regressor_dropout),
            nn.Linear(self.WIDTH, 128),
            nn.GELU(),
            nn.Dropout(self.final_regressor_dropout),
            nn.Linear(128, 1),
        )

    @property
    def requires_cpg_positions(self) -> bool:
        return False

    def _retrieve_and_predict(
        self,
        h_c_chunk: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch = tokens.shape[0]
        state = h_c_chunk[None, :, :].expand(batch, -1, -1)
        for attn, ffn in zip(self.retrieval_attn, self.retrieval_ffn):
            state = state + attn(state, tokens)
            state = ffn(state)

        h_for_concat = self.h_c_concat_norm(h_c_chunk)
        r_for_concat = self.r_pc_concat_norm(state)
        z_pc = torch.cat(
            [
                h_for_concat[None, :, :].expand(batch, -1, -1),
                r_for_concat,
            ],
            dim=-1,
        )
        return self.final_regressor(z_pc).squeeze(-1)

    def forward(
        self,
        rna: torch.Tensor,
        cpg_embedding: torch.Tensor | None = None,
        cpg_positions: torch.Tensor | None = None,
        *,
        functional_track_indices: torch.Tensor,
        functional_offsets: torch.Tensor,
        functional_dense: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del cpg_embedding, cpg_positions
        representation = self.rna_encoder(rna)
        if representation.program_tokens is None:
            raise RuntimeError("RNA encoder did not produce program tokens")
        tokens = representation.program_tokens

        peak = self.track_embedding(
            functional_track_indices, functional_offsets
        )
        dense = self.dense_encoder(functional_dense)
        h_c = self.locus_norm(peak + dense)
        mu_hat = torch.sigmoid(self.mean_head(h_c).squeeze(-1))

        n_loci = h_c.shape[0]
        if n_loci <= self.LOCUS_CHUNK:
            logit = self._retrieve_and_predict(h_c, tokens)
        else:
            chunks = []
            for start in range(0, n_loci, self.LOCUS_CHUNK):
                chunk = h_c[start : start + self.LOCUS_CHUNK]
                if self.training and torch.is_grad_enabled():
                    chunks.append(
                        checkpoint(
                            self._retrieve_and_predict,
                            chunk,
                            tokens,
                            use_reentrant=False,
                        )
                    )
                else:
                    chunks.append(self._retrieve_and_predict(chunk, tokens))
            logit = torch.cat(chunks, dim=1)

        beta = torch.sigmoid(logit)
        return {
            "beta": beta,
            "delta_logit": logit,
            "raw_delta": logit,
            "prediction_logit": logit,
            "mu_logit": None,
            "mu_hat": mu_hat,
            "residual_logit": logit,
            "h_cpg": h_c,
        }
PY

cat > src/methylation_predictor/modeling/__init__.py <<'PY'
"""Paper-facing model API."""

from .reference import IterativeRetrievalPredictor, SingleRetrievalPredictor

__all__ = [
    "SingleRetrievalPredictor",
    "IterativeRetrievalPredictor",
]
PY

python - <<'PY'
from pathlib import Path

path = Path("src/methylation_predictor/rna_training/locus_cls_trainer.py")
text = path.read_text()

old_import = """from ..models import (
    FeatureFusionArchitectureVariantModel,
    FeatureFusionLocusCLSModel,
    FunctionalConcatIterativeRNAModel,
    FunctionalConcatMASModel,
    FunctionalFusionModel,
    feature_fusion_variant_label,
    is_architecture_variant,
)
"""
new_import = """from ..models import (
    FeatureFusionArchitectureVariantModel,
    FeatureFusionLocusCLSModel,
    FunctionalConcatMASModel,
    FunctionalFusionModel,
    feature_fusion_variant_label,
    is_architecture_variant,
)
from ..modeling import IterativeRetrievalPredictor, SingleRetrievalPredictor
"""
if old_import not in text:
    raise SystemExit("trainer import block did not match expected phase-2a snapshot")
text = text.replace(old_import, new_import, 1)

old_dispatch = """            if self.functional_fusion_variant == "mas_concat_v4_iterative":
                # 2026-09-11 iterative-retrieval-depth candidate: everything
                # but the retrieval stack itself is byte-for-byte v3_purecontext
                # (own class, FunctionalConcatIterativeRNAModel, not a
                # FunctionalConcatMASModel constructor flag -- the two extra
                # retrieval blocks aren't expressible as one).
                self.model = FunctionalConcatIterativeRNAModel(
                    self.rna.values.shape[1], self.recipe.model,
                    final_regressor_dropout=final_regressor_dropout,
                ).to(self.device)
            else:
                self.model = FunctionalConcatMASModel(
                    self.rna.values.shape[1], self.recipe.model,
                    detach_h_c_main_path=self.functional_fusion_variant == "mas_concat_v2_detached",
                    separate_concat_norm=self.functional_fusion_variant == "mas_concat_v3_purecontext",
                    final_regressor_dropout=final_regressor_dropout,
                ).to(self.device)
"""
new_dispatch = """            if self.functional_fusion_variant == "mas_concat_v4_iterative":
                self.model = IterativeRetrievalPredictor(
                    self.rna.values.shape[1], self.recipe.model,
                    final_regressor_dropout=final_regressor_dropout,
                ).to(self.device)
            elif self.functional_fusion_variant == "mas_concat_v3_purecontext":
                self.model = SingleRetrievalPredictor(
                    self.rna.values.shape[1], self.recipe.model,
                    final_regressor_dropout=final_regressor_dropout,
                ).to(self.device)
            else:
                self.model = FunctionalConcatMASModel(
                    self.rna.values.shape[1], self.recipe.model,
                    detach_h_c_main_path=self.functional_fusion_variant == "mas_concat_v2_detached",
                    separate_concat_norm=False,
                    final_regressor_dropout=final_regressor_dropout,
                ).to(self.device)
"""
if old_dispatch not in text:
    raise SystemExit("trainer J0/J1 dispatch did not match expected phase-2a snapshot")
text = text.replace(old_dispatch, new_dispatch, 1)
path.write_text(text)
PY

python - <<'PY'
from pathlib import Path

path = Path("tests/test_refactor_candidate_contract.py")
text = path.read_text()

old = """def test_paper_facing_names_are_exact_compatibility_aliases_in_phase2a():
    # Phase 2a creates an import boundary only: zero production behavior change.
    assert SingleRetrievalPredictor is FunctionalConcatMASModel
    assert IterativeRetrievalPredictor is FunctionalConcatIterativeRNAModel
"""
new = """def test_phase2b_single_retrieval_is_exactly_equivalent_to_legacy_j0():
    cfg = _config()

    torch.manual_seed(123)
    legacy = FunctionalConcatMASModel(
        48,
        cfg,
        separate_concat_norm=True,
        final_regressor_dropout=0.15,
    ).eval()

    torch.manual_seed(123)
    refactored = SingleRetrievalPredictor(
        48,
        cfg,
        final_regressor_dropout=0.15,
    ).eval()

    legacy_state = legacy.state_dict()
    new_state = refactored.state_dict()
    assert legacy_state.keys() == new_state.keys()
    for key in legacy_state:
        torch.testing.assert_close(legacy_state[key], new_state[key], rtol=0, atol=0)

    torch.manual_seed(999)
    inputs = _functional_inputs()
    rna = torch.randn(3, 48)
    with torch.no_grad():
        old_out = legacy(rna, None, **inputs)
        new_out = refactored(rna, None, **inputs)

    for key in ("beta", "mu_hat", "prediction_logit", "h_cpg"):
        torch.testing.assert_close(old_out[key], new_out[key], rtol=0, atol=0)


def test_phase2b_iterative_is_exactly_equivalent_to_legacy_j1():
    cfg = _config()

    torch.manual_seed(321)
    legacy = FunctionalConcatIterativeRNAModel(
        48,
        cfg,
        final_regressor_dropout=0.15,
    ).eval()

    torch.manual_seed(321)
    refactored = IterativeRetrievalPredictor(
        48,
        cfg,
        final_regressor_dropout=0.15,
    ).eval()

    legacy_state = legacy.state_dict()
    new_state = refactored.state_dict()
    assert legacy_state.keys() == new_state.keys()
    for key in legacy_state:
        torch.testing.assert_close(legacy_state[key], new_state[key], rtol=0, atol=0)

    torch.manual_seed(1001)
    inputs = _functional_inputs()
    rna = torch.randn(3, 48)
    with torch.no_grad():
        old_out = legacy(rna, None, **inputs)
        new_out = refactored(rna, None, **inputs)

    for key in ("beta", "mu_hat", "prediction_logit", "h_cpg"):
        torch.testing.assert_close(old_out[key], new_out[key], rtol=0, atol=0)
"""
if old not in text:
    raise SystemExit("phase-2a alias test not found")
text = text.replace(old, new, 1)
path.write_text(text)
PY

cat >> docs/REFACTORING_CONTRACT.md <<'MD'

## Phase 2b status

J0 and J1 production dispatch now uses the paper-facing implementations in
`methylation_predictor.modeling`.

The historical classes remain temporarily in `models.py` so exact equivalence
and old-import compatibility can still be tested.

`tests/test_refactor_candidate_contract.py` verifies exact equality of the
initialized state dictionaries and deterministic forward outputs between the
historical and extracted J0/J1 implementations.

No RNA-token cleanup is included yet. The extracted candidates deliberately
still use the historical `build_rna_encoder`, preserving checkpoint keys and
numerical behavior. Removing the unused internal attention projections from
that encoder is phase 2c.
MD

echo
echo "Phase 2b applied."
echo
echo "Run:"
echo "  pytest -q tests/test_refactor_candidate_contract.py tests/test_mas_concat_model.py tests/test_iterative_retrieval_model.py"
echo "  pytest -q"
echo
echo "Then inspect:"
echo "  git diff --stat"
echo "  git diff"
