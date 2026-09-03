"""Shared registry for the architecture-novelty suite (architecture_novelty_2026_09).

Retargeted 2026-09-04: the suite originally sat on ``VarianceNormalizedResidualModel``
(the two-stage frozen-prior architecture); that architecture is now frozen for
old-checkpoint compatibility only (CLAUDE.md's "Model compatibility note"),
superseded by ``FeatureFusionLocusCLSModel`` via the ``shared_backbone_locus_cls_2026_09``
ablation ladder. Every arm here now extends
``models.py::FeatureFusionArchitectureVariantModel`` instead, on the
``matched_chr1_shared_backbone`` engine. The suite's earlier arms and their
``configs/models/arch/`` recipes stay in the repo (frozen, not deleted) as a
compatibility-only measurement of the retired architecture; they are not part of
this registry.

Not part of the stable CLI -- delete this module together with
scripts/experiments/{run_arch_suite,collect_arch_results}.py,
configs/models/arch_shared_backbone/ and
``models.py::FeatureFusionArchitectureVariantModel`` once the study concludes and
a decision is recorded in results/reference/ablations.yaml.

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

    # -- Stage 1c: depth on the concatenated [h_mean, h_raw] joint -- the
    # mandatory control for the HC/mHC arms below.
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

    # -- Stage 2: HOW THE TWO BRANCH EMBEDDINGS ARE COMBINED -- the postdoc's
    # original question, now asked of the architecture that actually has two
    # named branch embeddings to combine.
    Arm("trunk_hc_n2_d4", _arch("trunk_hc_n2_d4"), "2-hyper-connections",
        "Do two unconstrained residual streams over the joint beat one?", control="trunk_plain_d4"),
    Arm("trunk_mhc_n2_d4", _arch("trunk_mhc_n2_d4"), "2-hyper-connections",
        "Does the Birkhoff/doubly-stochastic constraint recover what HC loses?",
        control="trunk_hc_n2_d4"),
    Arm("trunk_mhc_stream_semantics_d4", _arch("trunk_mhc_stream_semantics_d4"), "2-hyper-connections",
        "Do the mean/raw embeddings themselves, as the two mHC streams, beat a single fusion Linear? "
        "(the literal answer to 'improve how the two branch embeddings are combined')",
        control="trunk_mhc_n2_d4"),

    # -- Stage 3: only after both components clear the promotion threshold.
    Arm("combined_locus_attention_mhc_semantic", _arch("combined_locus_attention_mhc_semantic"),
        "3-combination", "Do the two novelty components compose?",
        control="trunk_mhc_stream_semantics_d4"),
)

ARMS_BY_NAME = {arm.name: arm for arm in ARMS}
STAGES = tuple(dict.fromkeys(arm.stage for arm in ARMS))


def jobs(arms: tuple[Arm, ...] = ARMS) -> list[tuple[Arm, int]]:
    return [(arm, seed) for arm in arms for seed in arm.seeds]


def run_id(arm: Arm, seed: int) -> str:
    """Deterministic per (arm, seed): re-running the same unit on any machine
    reopens the same run directory instead of creating a duplicate. Unlike the
    two-stage suite, this engine has NO resume support (LocusCLSJointTrainer's
    RunStore.create is never called with resume=True) -- a killed run's
    directory is a dead end, not something a re-invocation can continue."""
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


def data_paths(data_root: str | None = None) -> dict[str, str]:
    root = Path(data_root or os.environ.get("METHYL_DATA_ROOT", "/dune/DATASETS/MethylPredictionData"))
    prepared = root / "derived" / "methylprophet_table5_tcga_chr1"
    canonical = root / "datasets" / "methylprophet_repro_v1"
    return {
        "canonical_root": str(canonical),
        "prepared_root": str(prepared),
        "feature_cache": str(prepared / "features"),
        "rna_cache": str(prepared / "rna"),
        "registry": str(canonical / "cpg" / "registries" / "array_cpg_map.parquet"),
        "cpg_targets_dir": str(root / "derived" / "cpg_statistics" / "chr1"),
        "output_root": str(root / "experiments"),
    }
