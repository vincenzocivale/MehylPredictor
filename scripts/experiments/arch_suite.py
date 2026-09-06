"""Shared registry for the architecture-novelty suite (architecture_novelty_2026_09).

Retargeted 2026-09-04: the suite originally sat on ``VarianceNormalizedResidualModel``
(the two-stage frozen-prior architecture), superseded by ``FeatureFusionLocusCLSModel``
via the ``shared_backbone_locus_cls_2026_09`` ablation ladder. Every arm here now
extends ``models.py::FeatureFusionArchitectureVariantModel`` instead, on the
``matched_chr1_shared_backbone`` engine.

Update (2026-09-06 refactor): the two-stage architecture and its whole generation
(``RNAMethylationPredictor``/``VarianceNormalizedResidualModel``/``RNA2DNAmModel``/
``ArchitectureVariantModel``/``DirectPredictionModel``, the ``MethylProphetTrainer``/
``ScopedRNATrainer`` engines, and its ``configs/models/arch/`` recipes) have since
been **removed from the codebase entirely, including old-checkpoint compatibility**
-- see CLAUDE.md's "Model compatibility note" and docs/RNA_METHYLATION.md's "Retired
architecture" section. The suite's earlier arms are not part of this registry and are
no longer runnable at all, not merely frozen.

Not part of the stable CLI, but no longer purely throwaway either: since 2026-09-05
``models.py::FeatureFusionArchitectureVariantModel`` (via ``encoder.kind=locus_attention``)
is the live primary RNA-methylation model -- see CLAUDE.md and docs/RNA_METHYLATION.md --
so it must NOT be deleted once this study concludes. ``configs/models/arch_shared_backbone/``
is likewise the ongoing home for the RNA-encoder-comparison harness (docs/RNA_METHYLATION.md's
"Forward direction" section), reused across studies, not a directory to delete either. Once
the ``architecture_novelty_2026_09`` study itself concludes and a decision is recorded in
results/reference/ablations.yaml, only that study's own queue-runner scaffolding --
scripts/experiments/{run_arch_suite,collect_arch_results}.py -- and its now-closed arm
entries/configs are candidates for removal.

Every arm runs on ``matched_chr1_shared_backbone`` at ``mode=final``, evaluated on
the official Table-5 chr1 Array views via ``evaluate_official_split`` -- the same
protocol that produced ``rung_b_official_final`` -- so results are directly
comparable to it. There is no development-mode screening tier for the same reason
as before: a second metric convention would make this suite's numbers
non-comparable to the number it is judged against.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY = "architecture_novelty_2026_09"
# Raw experiment outputs belong to the repository's results area.  Keep this
# independent from MethylPredictionData: that tree is reserved for canonical
# inputs and derived caches, not paper runs/checkpoints.
DEFAULT_EXPERIMENT_ROOT = REPO_ROOT / "results" / "experiments"

# PROVISIONAL reference: rung_b_official_final (results/reference/ablations.yaml)
# was stopped by user request at epoch 47/80, still improving 0.3-0.8%/epoch --
# this is a LOWER BOUND, not a converged number. seed_variance_reference (the
# noise-floor arm below) runs the same recipe to the full 80-epoch budget and
# supplies the real reference this suite should actually be judged against;
# update these two constants once it completes.
CANONICAL_MAS_PCC = 0.5627
CANONICAL_MSE = 0.01702
CANONICAL_RECIPE = "configs/models/rna_methylation_shared_backbone.yaml"
CANONICAL_IS_CONVERGED = False

HEADLINE_VIEW = "official_val_cpg_x_val_sample"
ENGINE = "matched_chr1_shared_backbone"


@dataclass(frozen=True)
class Arm:
    name: str
    recipe: str
    stage: str
    question: str
    seeds: tuple[int, ...] = (17,)
    control: str | None = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)


def _arch(name: str) -> str:
    return f"configs/models/arch_shared_backbone/{name}.yaml"


ARMS: tuple[Arm, ...] = (
    # -- Stage 0: the noise floor AND the missing convergence run for rung_b.
    # Run this FIRST -- nothing else is interpretable without it, and the
    # project is separately missing a converged reference number for its own
    # primary architecture until this exists.
    Arm("seed_variance_reference", CANONICAL_RECIPE, "0-noise-floor",
        "How large is a null delta on this protocol, and what does rung_b converge to?",
        seeds=(17, 29, 43)),

    # -- Stage 1a: is the raw branch's RNA encoder simply capacity-starved?
    Arm("enc_mlp", _arch("enc_mlp"), "1a-encoder-capacity",
        "Does the raw branch's RNA encoder just need a nonlinearity and a hidden layer?"),

    # -- Stage 1b: the flagship encoder arm.
    Arm("enc_locus_attention_k64", _arch("enc_locus_attention_k64"), "1b-locus-conditioned-rna",
        "Does letting each CpG query the transcriptome beat one locus-invariant RNA vector?",
        control="enc_mlp"),
    Arm("enc_locus_attention_k128", _arch("enc_locus_attention_k128"), "1b-locus-conditioned-rna",
        "Is the token count the binding constraint?", control="enc_locus_attention_k64"),

    # -- Stage 1f: RNA-encoder-comparison harness (docs/RNA_METHYLATION.md's
    # "Forward direction") -- alternative RNA encodings from the literature,
    # evaluated against the same locus-conditioned-attention baseline
    # (enc_locus_attention_k64 / row B).
    Arm("enc_bottleneck_mlp", _arch("enc_bottleneck_mlp"), "1f-encoder-comparison",
        "Does MethylProphet's published from-scratch RNA-encoder architecture (a 6-block "
        "pre-norm residual bottleneck MLP, trained here in place of the current encoder) "
        "match or beat locus-conditioned attention?",
        control="enc_locus_attention_k64"),
    Arm("enc_frozen_embedding_bulkrnabert", _arch("enc_frozen_embedding_bulkrnabert"),
        "1f-encoder-comparison",
        "Does a frozen pretrained transcriptome foundation-model embedding (BulkRNABert, "
        "TCGA-pretrained checkpoint, no fine-tuning) beat a from-scratch-trained encoder?",
        control="enc_locus_attention_k64",
        # Generated 2026-09-05 by scripts/prepare_bulkrnabert_embeddings.py
        # (10916, 256) real embeddings, bfloat16, all samples -- adjust this
        # path if that cache is ever regenerated somewhere else.
        extra_args=("--rna-cache", "MethylPredictionData/derived/bulkrnabert_embeddings/tcga")),

    # -- Stage 1c: depth on the concatenated [h_mean, h_raw] joint -- isolates
    # whether depth alone helps beyond the reference's single fusion Linear.
    Arm("trunk_plain_d2", _arch("trunk_plain_d2"), "1c-trunk-depth",
        "Does the reference architecture's single fusion Linear lose anything to depth at all?"),
    Arm("trunk_plain_d4", _arch("trunk_plain_d4"), "1c-trunk-depth",
        "Where does depth saturate?", control="trunk_plain_d2"),
    Arm("trunk_plain_d8", _arch("trunk_plain_d8"), "1c-trunk-depth",
        "Where does depth saturate?", control="trunk_plain_d4"),

    # -- Stage 1d/1e: orthogonal structural and likelihood changes.
    Arm("axial_cpg_d4", _arch("axial_cpg_d4"), "1d-axial-comethylation",
        "Does attending to genomic neighbours along the CpG axis add signal?",
        control="trunk_plain_d4"),
    Arm("beta_likelihood_head", _arch("beta_likelihood_head"), "1e-output-likelihood",
        "Does a bounded, heteroscedastic likelihood beat MSE on beta values?"),

    # Stage 2 ("how the two branch embeddings are combined", via a
    # Hyper-Connections/Manifold-Constrained-HC multi-stream trunk) was tried
    # and removed -- see docs/RNA_METHYLATION.md and CLAUDE.md's "Model
    # compatibility note" for why (not worth the added complexity for the
    # measured gain). Stage 3 (combining it with locus-conditioned attention)
    # was removed alongside it, having no remaining component to combine.
)

ARMS_BY_NAME = {arm.name: arm for arm in ARMS}
STAGES = tuple(dict.fromkeys(arm.stage for arm in ARMS))


def jobs(arms: tuple[Arm, ...] = ARMS) -> list[tuple[Arm, int]]:
    return [(arm, seed) for arm in arms for seed in arm.seeds]


def run_id(arm: Arm, seed: int) -> str:
    """Deterministic per (arm, seed): re-running the same unit on any machine
    reopens the same run directory instead of creating a duplicate.
    ``run_arch_suite.py`` auto-resumes a killed run from its
    ``checkpoints/last.pt`` when one exists."""
    return f"arch-{STUDY}-shared-{arm.name}-seed{seed}"


def select(names: str | None, stages: str | None, shard: str | None) -> tuple[Arm, ...]:
    chosen = ARMS
    if names:
        wanted = [n.strip() for n in names.split(",") if n.strip()]
        unknown = [n for n in wanted if n not in ARMS_BY_NAME]
        if unknown:
            raise SystemExit(f"unknown arm(s): {', '.join(unknown)}\nknown: {', '.join(ARMS_BY_NAME)}")
        chosen = tuple(ARMS_BY_NAME[n] for n in wanted)
    if stages:
        wanted_stages = {s.strip() for s in stages.split(",") if s.strip()}
        unknown = wanted_stages - set(STAGES)
        if unknown:
            raise SystemExit(f"unknown stage(s): {', '.join(sorted(unknown))}\nknown: {', '.join(STAGES)}")
        chosen = tuple(a for a in chosen if a.stage in wanted_stages)
    if shard:
        index, total = (int(x) for x in shard.split("/"))
        if not (1 <= index <= total):
            raise SystemExit(f"--shard must be i/n with 1 <= i <= n; got {shard!r}")
        chosen = tuple(a for i, a in enumerate(chosen) if i % total == index - 1)
    return chosen


def data_paths(
    data_root: str | None = None,
    experiment_root: str | None = None,
) -> dict[str, str]:
    root = Path(data_root or os.environ.get("METHYL_DATA_ROOT", "/dune/DATASETS/MethylPredictionData"))
    output_root = Path(
        experiment_root
        or os.environ.get("METHYL_EXPERIMENT_ROOT", str(DEFAULT_EXPERIMENT_ROOT))
    )
    prepared = root / "derived" / "methylprophet_table5_tcga_chr1"
    canonical = root / "datasets" / "methylprophet_repro_v1"
    return {
        "canonical_root": str(canonical),
        "prepared_root": str(prepared),
        "feature_cache": str(prepared / "features"),
        "rna_cache": str(prepared / "rna"),
        "registry": str(canonical / "cpg" / "registries" / "array_cpg_map.parquet"),
        "cpg_targets_dir": str(root / "derived" / "cpg_statistics" / "chr1"),
        "output_root": str(output_root),
    }
