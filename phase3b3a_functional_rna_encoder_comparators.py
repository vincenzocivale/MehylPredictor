#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


ROOT = Path.cwd()
EXPECTED_BRANCH = "refactor/repo-v2-2026-09"


def require_branch() -> None:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], text=True
    ).strip()
    if branch != EXPECTED_BRANCH:
        raise SystemExit(
            f"expected branch {EXPECTED_BRANCH!r}, got {branch!r}"
        )


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{path}: expected exactly one match, found {count}\n"
            f"--- needle ---\n{old[:700]}"
        )
    path.write_text(text.replace(old, new, 1))


def write_new(path: Path, content: str) -> None:
    if path.exists():
        raise SystemExit(f"{path}: already exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    require_branch()

    if args.dry_run:
        print(
            """Phase 3b3a will:
  + add functional RNA-encoder comparator implementations that all emit the
    same 64x256 program-token contract consumed by J0 retrieval
  + add RNAEncoderComparisonPredictor, reusing J0 downstream architecture
  + keep the entire functional/retrieval/mean-proxy/final-regressor stack
    bit-identically initialized to J0 under the same seed
  + add fresh functional comparison recipes for:
      - ours ProgramTokenEncoder
      - MethylProphet-style BottleneckMLP
      - GenePathway
      - frozen BulkFormer-147M
      - frozen BulkRNABert
  + add comparison registry + regression tests
  ~ add trainer dispatch for functional_rna_encoder_comparison

Historical shared-backbone RNA-encoder configs/results/runners are retained
unchanged for provenance. No GPU jobs are launched by this patch.
"""
        )
        return 0

    # ------------------------------------------------------------------
    # 1. Comparator RNA encoders: global encoders are lifted to the same
    #    K x program_dim token interface used by J0.
    # ------------------------------------------------------------------
    write_new(
        ROOT / "src/methylation_predictor/modeling/rna_comparators.py",
        """\
\"\"\"RNA encoders for the functional, matched RNA-encoder comparison.

Every encoder returns a tensor of shape ``[batch, n_programs, program_dim]``.
This keeps J0's functional locus encoder, locus-conditioned retrieval,
mean-proxy task, and beta regressor fixed while varying only the patient RNA
representation upstream of retrieval.
\"\"\"
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..config import EncoderConfig
from .rna import ProgramTokenEncoder


class GlobalToProgramTokens(nn.Module):
    \"\"\"Lift one patient-global latent into the common program-token space.\"\"\"

    def __init__(
        self,
        global_dim: int,
        n_programs: int,
        program_dim: int,
    ):
        super().__init__()
        self.global_norm = nn.LayerNorm(global_dim)
        self.program_basis = nn.Parameter(
            torch.empty(n_programs, program_dim, global_dim)
        )
        nn.init.normal_(self.program_basis, std=global_dim ** -0.5)
        self.token_norm = nn.LayerNorm(program_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.global_norm(x)
        tokens = torch.einsum("br,kdr->bkd", x, self.program_basis)
        return self.token_norm(tokens)


class _BottleneckMLPBlock(nn.Module):
    def __init__(self, width: int, inner_width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, inner_width)
        self.fc2 = nn.Linear(inner_width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(
            self.dropout(
                torch.nn.functional.gelu(self.fc1(self.norm(x)))
            )
        )
        return x + h


class BottleneckMLPProgramEncoder(nn.Module):
    \"\"\"MethylProphet-style B_6-Wi_1024 encoder -> common token bank.\"\"\"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        n_blocks: int,
        mlp_ratio: int,
        dropout: float,
        layer_norm: bool,
        n_programs: int,
        program_dim: int,
    ):
        super().__init__()
        self.norm = (
            nn.LayerNorm(input_dim) if layer_norm else nn.Identity()
        )
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            _BottleneckMLPBlock(
                hidden_dim,
                hidden_dim * mlp_ratio,
                dropout,
            )
            for _ in range(n_blocks)
        )
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        self.to_tokens = GlobalToProgramTokens(
            latent_dim, n_programs, program_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(self.norm(x))
        for block in self.blocks:
            h = block(h)
        return self.to_tokens(self.output_proj(h))


class FrozenEmbeddingProgramEncoder(nn.Module):
    \"\"\"Frozen sample embedding + thin adapter -> common token bank.\"\"\"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        layer_norm: bool,
        n_programs: int,
        program_dim: int,
    ):
        super().__init__()
        self.norm = (
            nn.LayerNorm(input_dim) if layer_norm else nn.Identity()
        )
        self.projection = nn.Linear(input_dim, latent_dim)
        self.to_tokens = GlobalToProgramTokens(
            latent_dim, n_programs, program_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_tokens(self.projection(self.norm(x)))


class GenePathwayProgramEncoder(nn.Module):
    \"\"\"SurvPath-style sparse gene->pathway encoder -> common token bank.\"\"\"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        membership_path: str,
        dim1: int,
        dim2: int,
        dropout: float,
        layer_norm: bool,
        n_programs: int,
        program_dim: int,
    ):
        super().__init__()
        if not membership_path:
            raise ValueError(
                "gene_pathway requires encoder.pathway_membership_path"
            )
        path = Path(membership_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"gene-pathway membership file not found: {path}"
            )

        data = np.load(path, allow_pickle=False)
        n_genes = int(np.asarray(data["n_genes"]).reshape(-1)[0])
        n_pathways = int(np.asarray(data["n_pathways"]).reshape(-1)[0])
        gene_idx = np.asarray(data["gene_idx"], dtype=np.int64)
        pathway_idx = np.asarray(data["pathway_idx"], dtype=np.int64)

        if n_genes != input_dim:
            raise ValueError(
                f"pathway membership was built for {n_genes} genes but "
                f"RNA cache width is {input_dim}"
            )
        if (
            gene_idx.ndim != 1
            or pathway_idx.ndim != 1
            or gene_idx.shape != pathway_idx.shape
        ):
            raise ValueError(
                "invalid gene-pathway membership: gene_idx/pathway_idx "
                "must be equal-length 1-D arrays"
            )
        if len(gene_idx) == 0:
            raise ValueError("gene-pathway membership has zero edges")

        self.input_dim = int(input_dim)
        self.n_pathways = int(n_pathways)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.norm = (
            nn.LayerNorm(input_dim) if layer_norm else nn.Identity()
        )

        indices = np.stack([pathway_idx, gene_idx], axis=0)
        self.register_buffer(
            "membership_indices",
            torch.from_numpy(indices).long(),
            persistent=True,
        )

        n_edges = int(gene_idx.size)
        per_path = np.bincount(pathway_idx, minlength=n_pathways)
        scale = float(
            1.0 / max(np.sqrt(max(float(per_path.mean()), 1.0)), 1.0)
        )
        self.edge_weight = nn.Parameter(torch.empty(self.dim1, n_edges))
        nn.init.normal_(self.edge_weight, std=scale)
        self.bias1 = nn.Parameter(
            torch.zeros(n_pathways, self.dim1)
        )

        self.weight2 = nn.Parameter(
            torch.empty(n_pathways, self.dim1, self.dim2)
        )
        nn.init.xavier_normal_(self.weight2)
        self.bias2 = nn.Parameter(
            torch.zeros(n_pathways, self.dim2)
        )

        flattened = n_pathways * self.dim2
        hidden = max(latent_dim, flattened // 4)
        self.to_latent = nn.Sequential(
            nn.LayerNorm(flattened),
            nn.Linear(flattened, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim),
        )
        self.to_tokens = GlobalToProgramTokens(
            latent_dim, n_programs, program_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x_t = x.float().transpose(0, 1).contiguous()

        first_channels = []
        with torch.autocast(device_type=x.device.type, enabled=False):
            for channel in range(self.dim1):
                weight = torch.sparse_coo_tensor(
                    self.membership_indices,
                    self.edge_weight[channel].float(),
                    size=(self.n_pathways, self.input_dim),
                    device=x.device,
                ).coalesce()
                first_channels.append(
                    torch.sparse.mm(weight, x_t).transpose(0, 1)
                )

        h1 = (
            torch.stack(first_channels, dim=-1)
            + self.bias1.float()[None, :, :]
        )
        h2 = (
            torch.einsum(
                "bpi,pij->bpj",
                h1,
                self.weight2.float(),
            )
            + self.bias2.float()[None, :, :]
        )
        latent = self.to_latent(h2.reshape(h2.shape[0], -1))
        return self.to_tokens(latent)


def build_comparison_program_encoder(
    config: EncoderConfig,
    input_dim: int,
) -> nn.Module:
    \"\"\"Build one RNA encoder under the common program-token contract.\"\"\"

    common = dict(
        n_programs=config.n_programs,
        program_dim=config.program_dim,
    )

    if config.kind == "locus_attention":
        return ProgramTokenEncoder(
            input_dim=input_dim,
            n_programs=config.n_programs,
            program_dim=config.program_dim,
            bottleneck_dim=config.latent_dim,
            layer_norm=config.layer_norm,
        )

    if config.kind == "bottleneck_mlp":
        return BottleneckMLPProgramEncoder(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            n_blocks=config.n_blocks,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            layer_norm=config.layer_norm,
            **common,
        )

    if config.kind == "gene_pathway":
        return GenePathwayProgramEncoder(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            membership_path=config.pathway_membership_path,
            dim1=config.pathway_dim1,
            dim2=config.pathway_dim2,
            dropout=config.dropout,
            layer_norm=config.layer_norm,
            **common,
        )

    if config.kind == "frozen_embedding":
        return FrozenEmbeddingProgramEncoder(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            layer_norm=config.layer_norm,
            **common,
        )

    raise ValueError(
        "functional RNA-encoder comparison supports "
        "locus_attention, bottleneck_mlp, gene_pathway, "
        f"or frozen_embedding; got {config.kind!r}"
    )
""",
    )

    # ------------------------------------------------------------------
    # 2. Comparison predictor: inherit J0 and replace only rna_encoder.
    #    Super-construction uses the canonical 25,017-gene input so every
    #    downstream parameter is initialized exactly as in main J0.
    # ------------------------------------------------------------------
    write_new(
        ROOT / "src/methylation_predictor/modeling/rna_comparison.py",
        """\
\"\"\"Functional-matched RNA encoder comparison model.\"\"\"
from __future__ import annotations

from dataclasses import replace

from ..config import ModelConfig
from .reference import SingleRetrievalPredictor
from .rna_comparators import build_comparison_program_encoder


class RNAEncoderComparisonPredictor(SingleRetrievalPredictor):
    \"\"\"J0 with only the upstream patient RNA encoder replaced.

    The functional CpG encoder, retrieval attention, mean-proxy head and final
    beta regressor are constructed by ``SingleRetrievalPredictor`` and retain
    its exact same-seed initialization.

    The canonical RNA axis used by ``main.yaml`` has 25,017 columns.  We build
    that reference J0 first even for frozen-embedding arms, then replace only
    ``rna_encoder``.  This decouples downstream initialization from comparator
    input width/capacity.
    \"\"\"

    REFERENCE_RNA_INPUT_DIM = 25017

    def __init__(
        self,
        input_dim: int,
        config: ModelConfig,
        *,
        final_regressor_dropout: float = 0.15,
        use_mean_proxy: bool = True,
    ):
        comparison_encoder = config.encoder

        reference_encoder = replace(
            comparison_encoder,
            kind="locus_attention",
        )
        reference_config = replace(
            config,
            encoder=reference_encoder,
        )

        super().__init__(
            self.REFERENCE_RNA_INPUT_DIM,
            reference_config,
            final_regressor_dropout=final_regressor_dropout,
            use_mean_proxy=use_mean_proxy,
        )

        # For the reference arm on the canonical 25,017-gene cache, the
        # encoder created by super() already is exactly the production J0
        # ProgramTokenEncoder, so retain it bit-for-bit.
        if not (
            comparison_encoder.kind == "locus_attention"
            and input_dim == self.REFERENCE_RNA_INPUT_DIM
        ):
            self.rna_encoder = build_comparison_program_encoder(
                comparison_encoder,
                input_dim,
            )

        self.comparison_encoder_kind = comparison_encoder.kind
        self.comparison_encoder_source = (
            comparison_encoder.frozen_embedding_source
        )
""",
    )

    # ------------------------------------------------------------------
    # 3. modeling namespace.
    # ------------------------------------------------------------------
    init_path = ROOT / "src/methylation_predictor/modeling/__init__.py"
    text = init_path.read_text()
    if "RNAEncoderComparisonPredictor" in text:
        raise SystemExit("modeling/__init__.py already has RNA comparison API")
    text = text.replace(
        "from .reference import IterativeRetrievalPredictor, SingleRetrievalPredictor\n",
        "from .reference import IterativeRetrievalPredictor, SingleRetrievalPredictor\n"
        "from .rna_comparison import RNAEncoderComparisonPredictor\n",
        1,
    )
    text = text.replace(
        '    "IterativeRetrievalPredictor",\n',
        '    "IterativeRetrievalPredictor",\n'
        '    "RNAEncoderComparisonPredictor",\n',
        1,
    )
    init_path.write_text(text)

    # ------------------------------------------------------------------
    # 4. Trainer dispatch.
    # ------------------------------------------------------------------
    trainer = ROOT / "src/methylation_predictor/rna_training/locus_cls_trainer.py"

    replace_once(
        trainer,
        """from ..modeling import (
    FunctionalBaselinePredictor,
    IterativeRetrievalPredictor,
    SingleRetrievalPredictor,
)
""",
        """from ..modeling import (
    FunctionalBaselinePredictor,
    IterativeRetrievalPredictor,
    RNAEncoderComparisonPredictor,
    SingleRetrievalPredictor,
)
""",
    )

    replace_once(
        trainer,
        """            or self.functional_fusion_variant in BASELINE_VARIANTS
        )
""",
        """            or self.functional_fusion_variant in BASELINE_VARIANTS
            or self.functional_fusion_variant
            == "functional_rna_encoder_comparison"
        )
""",
    )

    replace_once(
        trainer,
        """            if self.functional_fusion_variant in BASELINE_VARIANTS:
                self.architecture_label = self.functional_fusion_variant
                self.model = FunctionalBaselinePredictor(
                    self.rna.values.shape[1],
                    self.recipe.model,
                    variant=self.functional_fusion_variant,
                    final_regressor_dropout=final_regressor_dropout,
                    use_mean_proxy=use_mean_branch,
                ).to(self.device)
            else:
                candidate_cls = (
""",
        """            if self.functional_fusion_variant in BASELINE_VARIANTS:
                self.architecture_label = self.functional_fusion_variant
                self.model = FunctionalBaselinePredictor(
                    self.rna.values.shape[1],
                    self.recipe.model,
                    variant=self.functional_fusion_variant,
                    final_regressor_dropout=final_regressor_dropout,
                    use_mean_proxy=use_mean_branch,
                ).to(self.device)
            elif (
                self.functional_fusion_variant
                == "functional_rna_encoder_comparison"
            ):
                source = self.recipe.model.encoder.frozen_embedding_source
                suffix = (
                    f"-{source}"
                    if source
                    else f"-{self.recipe.model.encoder.kind}"
                )
                self.architecture_label = (
                    "functional_rna_encoder_comparison" + suffix
                )
                self.model = RNAEncoderComparisonPredictor(
                    self.rna.values.shape[1],
                    self.recipe.model,
                    final_regressor_dropout=final_regressor_dropout,
                    use_mean_proxy=use_mean_branch,
                ).to(self.device)
            else:
                candidate_cls = (
""",
    )

    replace_once(
        trainer,
        """                "'mas_concat_v3_purecontext', 'mas_concat_v4_iterative', "
                "and the functional_baseline_* comparison variants"
""",
        """                "'mas_concat_v3_purecontext', 'mas_concat_v4_iterative', "
                "'functional_rna_encoder_comparison', and the "
                "functional_baseline_* comparison variants"
""",
    )

    # ------------------------------------------------------------------
    # 5. Fresh functional-matched recipes. Existing historical configs stay
    #    untouched because their recorded metrics depend on them.
    # ------------------------------------------------------------------
    cfg_dir = ROOT / "configs/models/rna_encoder_comparison"

    configs = {
        "functional_ours_program_tokens.yaml": """\
# Functional-matched RNA encoder comparison: production J0 RNA token encoder.
extends: ../main.yaml

model:
  functional_fusion_variant: functional_rna_encoder_comparison

tracking:
  group: functional_rna_encoder_comparison_2026_09
  tags: [functional-only, rna-encoder, ours-program-tokens]
""",
        "functional_bottleneck_mlp.yaml": """\
# Functional-matched RNA encoder comparison: MethylProphet-style B_6-Wi_1024.
# The global 256-D output is lifted to the same 64x256 token contract before
# the unchanged J0 locus-conditioned retrieval.
extends: ../main.yaml

model:
  functional_fusion_variant: functional_rna_encoder_comparison
  encoder:
    kind: bottleneck_mlp
    hidden_dim: 1024
    n_blocks: 6
    mlp_ratio: 4
    dropout: 0.1
    layer_norm: true

tracking:
  group: functional_rna_encoder_comparison_2026_09
  tags: [functional-only, rna-encoder, methylprophet-bottleneck-mlp]
""",
        "functional_gene_pathway.yaml": """\
# Functional-matched RNA encoder comparison: SurvPath-style gene->pathway.
extends: ../main.yaml

model:
  functional_fusion_variant: functional_rna_encoder_comparison
  encoder:
    kind: gene_pathway
    dropout: 0.1
    layer_norm: false
    pathway_membership_path: resources/pathways/survpath_combine_tcga.npz
    pathway_dim1: 8
    pathway_dim2: 16

tracking:
  group: functional_rna_encoder_comparison_2026_09
  tags: [functional-only, rna-encoder, gene-pathway]
""",
        "functional_bulkformer_147m.yaml": """\
# Functional-matched RNA encoder comparison: frozen BulkFormer-147M embedding.
# Use an RNACache prepared by scripts/prepare_bulkformer_embeddings.py.
extends: ../main.yaml

model:
  functional_fusion_variant: functional_rna_encoder_comparison
  encoder:
    kind: frozen_embedding
    layer_norm: true
    frozen_embedding_source: bulkformer_147m

tracking:
  group: functional_rna_encoder_comparison_2026_09
  tags: [functional-only, rna-encoder, frozen, bulkformer-147m]
""",
        "functional_bulkrnabert.yaml": """\
# Functional-matched RNA encoder comparison: frozen BulkRNABert embedding.
# Use an RNACache prepared by scripts/prepare_bulkrnabert_embeddings.py.
extends: ../main.yaml

model:
  functional_fusion_variant: functional_rna_encoder_comparison
  encoder:
    kind: frozen_embedding
    layer_norm: true
    frozen_embedding_source: bulkrnabert_tcga

tracking:
  group: functional_rna_encoder_comparison_2026_09
  tags: [functional-only, rna-encoder, frozen, bulkrnabert]
""",
    }
    for name, content in configs.items():
        write_new(cfg_dir / name, content)

    # ------------------------------------------------------------------
    # 6. Registry only. Runner/collector migration is intentionally a
    #    separate phase after the model contract is frozen by tests.
    # ------------------------------------------------------------------
    write_new(
        ROOT / "scripts/experiments/functional_rna_encoder_suite.py",
        """\
\"\"\"Registry for the functional-matched RNA encoder comparison.\"\"\"
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
""",
    )

    # ------------------------------------------------------------------
    # 7. Regression tests.
    # ------------------------------------------------------------------
    write_new(
        ROOT / "tests/test_functional_rna_encoder_comparison.py",
        """\
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from methylation_predictor.config import EncoderConfig, ModelConfig
from methylation_predictor.modeling import (
    RNAEncoderComparisonPredictor,
    SingleRetrievalPredictor,
)
from methylation_predictor.modeling.rna_comparators import (
    BottleneckMLPProgramEncoder,
    FrozenEmbeddingProgramEncoder,
    GenePathwayProgramEncoder,
)
from methylation_predictor.rna_training.config import load_rna_recipe


ROOT = Path(__file__).resolve().parents[1]


def _functional_inputs():
    return {
        "functional_track_indices": torch.tensor(
            [1, 2, 5, 9], dtype=torch.int64
        ),
        "functional_offsets": torch.tensor(
            [0, 2, 2, 3, 4], dtype=torch.int64
        ),
        "functional_dense": torch.randn(4, 23),
    }


def _reference_config():
    return ModelConfig(
        encoder=EncoderConfig(
            kind="locus_attention",
            latent_dim=8,
            n_programs=6,
            program_dim=256,
            n_heads=4,
            dropout=0.0,
            layer_norm=True,
        )
    )


def _membership(path: Path):
    np.savez_compressed(
        path,
        n_genes=np.asarray([6], dtype=np.int64),
        n_pathways=np.asarray([3], dtype=np.int64),
        gene_idx=np.asarray([0, 1, 1, 2, 3, 4], dtype=np.int64),
        pathway_idx=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
        pathway_names=np.asarray(["p0", "p1", "p2"]),
        matched_genes_per_pathway=np.asarray([2, 2, 2], dtype=np.int64),
    )


def test_functional_rna_encoder_recipes_keep_non_encoder_protocol_fixed():
    main = load_rna_recipe(ROOT / "configs/models/main.yaml").raw

    files = (
        "functional_ours_program_tokens.yaml",
        "functional_bottleneck_mlp.yaml",
        "functional_gene_pathway.yaml",
        "functional_bulkformer_147m.yaml",
        "functional_bulkrnabert.yaml",
    )

    for filename in files:
        raw = load_rna_recipe(
            ROOT / "configs/models/rna_encoder_comparison" / filename
        ).raw

        for key in (
            "loss",
            "training",
            "schedule_policy",
            "exclude_official_val_from_auxiliary",
            "structured_loss_sources",
            "locus_cls",
            "batching",
            "development",
        ):
            assert raw[key] == main[key]

        lhs = dict(raw["model"])
        rhs = dict(main["model"])
        assert lhs.pop("functional_fusion_variant") == (
            "functional_rna_encoder_comparison"
        )
        assert rhs.pop("functional_fusion_variant") == (
            "mas_concat_v3_purecontext"
        )
        lhs.pop("encoder")
        rhs.pop("encoder")
        assert lhs == rhs


def test_ours_comparison_arm_is_bit_exact_j0_at_initialization_and_forward():
    cfg = _reference_config()

    torch.manual_seed(17)
    j0 = SingleRetrievalPredictor(
        25017,
        cfg,
        final_regressor_dropout=0.15,
    ).eval()

    torch.manual_seed(17)
    comp = RNAEncoderComparisonPredictor(
        25017,
        cfg,
        final_regressor_dropout=0.15,
    ).eval()

    assert j0.state_dict().keys() == comp.state_dict().keys()
    for key, value in j0.state_dict().items():
        torch.testing.assert_close(
            value,
            comp.state_dict()[key],
            rtol=0,
            atol=0,
        )

    rna = torch.randn(2, 25017)
    inputs = _functional_inputs()
    with torch.no_grad():
        a = j0(rna, None, **inputs)
        b = comp(rna, None, **inputs)
    torch.testing.assert_close(a["beta"], b["beta"], rtol=0, atol=0)
    torch.testing.assert_close(
        a["mu_hat"], b["mu_hat"], rtol=0, atol=0
    )


def test_bottleneck_and_frozen_encoders_emit_common_token_contract():
    bottleneck = BottleneckMLPProgramEncoder(
        input_dim=12,
        latent_dim=8,
        hidden_dim=16,
        n_blocks=2,
        mlp_ratio=2,
        dropout=0.0,
        layer_norm=True,
        n_programs=6,
        program_dim=256,
    )
    frozen = FrozenEmbeddingProgramEncoder(
        input_dim=10,
        latent_dim=8,
        layer_norm=True,
        n_programs=6,
        program_dim=256,
    )

    assert bottleneck(torch.randn(3, 12)).shape == (3, 6, 256)
    assert frozen(torch.randn(3, 10)).shape == (3, 6, 256)


def test_gene_pathway_program_encoder_forward_and_gradient(tmp_path):
    membership = tmp_path / "membership.npz"
    _membership(membership)

    model = GenePathwayProgramEncoder(
        input_dim=6,
        latent_dim=8,
        membership_path=str(membership),
        dim1=2,
        dim2=3,
        dropout=0.0,
        layer_norm=False,
        n_programs=6,
        program_dim=256,
    )
    x = torch.randn(4, 6, requires_grad=True)
    tokens = model(x)

    assert tokens.shape == (4, 6, 256)
    assert torch.isfinite(tokens).all()
    tokens.square().mean().backward()
    assert model.edge_weight.grad is not None
    assert torch.isfinite(model.edge_weight.grad).all()


def test_downstream_j0_parameters_do_not_depend_on_comparator_encoder(tmp_path):
    ref_cfg = _reference_config()

    torch.manual_seed(23)
    reference = SingleRetrievalPredictor(
        25017,
        ref_cfg,
        final_regressor_dropout=0.15,
    )
    reference_state = reference.state_dict()

    variants = []

    variants.append(
        (
            12,
            ModelConfig(
                encoder=EncoderConfig(
                    kind="bottleneck_mlp",
                    latent_dim=8,
                    hidden_dim=16,
                    n_blocks=2,
                    mlp_ratio=2,
                    dropout=0.0,
                    layer_norm=True,
                    n_programs=6,
                    program_dim=256,
                    n_heads=4,
                )
            ),
        )
    )

    variants.append(
        (
            10,
            ModelConfig(
                encoder=EncoderConfig(
                    kind="frozen_embedding",
                    latent_dim=8,
                    layer_norm=True,
                    n_programs=6,
                    program_dim=256,
                    n_heads=4,
                    frozen_embedding_source="unit-test",
                )
            ),
        )
    )

    membership = tmp_path / "membership.npz"
    _membership(membership)
    variants.append(
        (
            6,
            ModelConfig(
                encoder=EncoderConfig(
                    kind="gene_pathway",
                    latent_dim=8,
                    dropout=0.0,
                    layer_norm=False,
                    n_programs=6,
                    program_dim=256,
                    n_heads=4,
                    pathway_membership_path=str(membership),
                    pathway_dim1=2,
                    pathway_dim2=3,
                )
            ),
        )
    )

    for input_dim, cfg in variants:
        torch.manual_seed(23)
        comparator = RNAEncoderComparisonPredictor(
            input_dim,
            cfg,
            final_regressor_dropout=0.15,
        )
        state = comparator.state_dict()

        for key, value in reference_state.items():
            if key.startswith("rna_encoder."):
                continue
            torch.testing.assert_close(
                value,
                state[key],
                rtol=0,
                atol=0,
            )
""",
    )

    # ------------------------------------------------------------------
    # 8. Documentation.
    # ------------------------------------------------------------------
    scope_doc = ROOT / "docs/REPOSITORY_SCOPE.md"
    with scope_doc.open("a") as handle:
        handle.write(
            """
## Phase 3b3a: functional-matched RNA encoder comparison

RNA encoder comparisons now have a paper-facing implementation that keeps the
functional locus representation and J0 retrieval stack fixed.

Every arm emits the same `[patient, 64, 256]` program-token contract before
the unchanged locus-conditioned retrieval:

```text
ours             canonical RNA -> ProgramTokenEncoder -> 64x256 tokens
BottleneckMLP    canonical RNA -> MP-style global encoder -> token lift
GenePathway      canonical RNA -> sparse pathway encoder -> token lift
BulkFormer       frozen sample embedding -> thin adapter -> token lift
BulkRNABert      frozen sample embedding -> thin adapter -> token lift
```

`RNAEncoderComparisonPredictor` constructs the production J0 downstream stack
with the canonical 25,017-gene reference input before replacing only the RNA
encoder. Therefore the functional CpG encoder, retrieval attention, mean-proxy
head and final regressor retain bit-identical same-seed initialization across
comparison arms.

Historical shared-backbone RNA-encoder configs and recorded result ledgers are
retained unchanged for provenance. Fresh functional-matched results must use
the new `functional_*` recipes and fresh run IDs.
"""
        )

    print("Phase 3b3a applied.")
    print()
    print("Focused tests:")
    print(
        "  pytest -q tests/test_functional_rna_encoder_comparison.py "
        "tests/test_functional_baselines.py "
        "tests/test_refactor_candidate_contract.py"
    )
    print()
    print("Then full suite:")
    print("  pytest -q")
    print()
    print("Inspect:")
    print("  git diff --stat")
    print("  git diff")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
