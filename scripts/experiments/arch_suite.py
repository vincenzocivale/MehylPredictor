"""Shared registry for the architecture-novelty suite (architecture_novelty_2026_09).

Not part of the stable CLI -- delete this module together with
scripts/experiments/{run_arch_suite,collect_arch_results}.py, configs/models/arch/
and the ArchitectureVariantModel machinery in src/methylation_predictor/models.py
once the study concludes and a decision is recorded in
results/reference/ablations.yaml.

Every arm runs on the frozen matched_chr1 engine at mode=final, so its MAS-PCC on
the official Table-5 chr1 views is directly comparable both to the canonical
0.5613 and to the earlier fusion_mechanism_2026_08 arms, which used the same
engine, split, seed and budget. There is deliberately no development-mode
screening tier: MethylProphetTrainer has no refit cycle by design, and inventing a
second metric convention (inner_double_ood_mas_pcc) for this suite would make its
numbers non-comparable to the very baseline it is being judged against.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY = "architecture_novelty_2026_09"

# Canonical reference this suite is judged against
# (results/reference/rna_methylation/chr1.yaml, mode=final, matched_chr1).
CANONICAL_MAS_PCC = 0.5613
CANONICAL_MSE = 0.01941
CANONICAL_RECIPE = "configs/models/rna_methylation.yaml"
HEADLINE_VIEW = "val_cpg_x_val_sample"


@dataclass(frozen=True)
class Arm:
    name: str
    recipe: str
    stage: str
    question: str
    seeds: tuple[int, ...] = (17,)
    # Arms whose result is only interpretable next to another arm's.
    control: str | None = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)


def _arch(name: str) -> str:
    return f"configs/models/arch/{name}.yaml"


ARMS: tuple[Arm, ...] = (
    # -- Stage 0: the noise floor. Run this FIRST; nothing else is interpretable
    # without it. Three seeds of the unmodified canonical recipe give the SD that
    # every other arm's delta is judged against. No previous ablation in this repo
    # ever measured it, which is why sigma_normalization_2026_08's +0.0007 could
    # only be called "in noise" by assertion rather than by measurement.
    Arm("seed_variance_canonical", CANONICAL_RECIPE, "0-noise-floor",
        "How large is a null delta on this protocol?", seeds=(17, 29, 43)),

    # -- Stage 1a: is the RNA branch simply capacity-starved?
    Arm("enc_mlp", _arch("enc_mlp"), "1a-encoder-capacity",
        "Does the RNA branch just need a nonlinearity and a hidden layer?"),
    Arm("enc_program_bottleneck", _arch("enc_program_bottleneck"), "1a-encoder-capacity",
        "Does an interpretable normalized gene-program bottleneck match a plain MLP?",
        control="enc_mlp"),

    # -- Stage 1b: the flagship. Make the RNA representation locus-specific.
    Arm("enc_locus_attention_k64", _arch("enc_locus_attention_k64"), "1b-locus-conditioned-rna",
        "Does letting each CpG query the transcriptome beat one locus-invariant vector?",
        control="enc_mlp"),
    Arm("enc_locus_attention_k128", _arch("enc_locus_attention_k128"), "1b-locus-conditioned-rna",
        "Is the token count the binding constraint?", control="enc_locus_attention_k64"),

    # -- Stage 1c: depth. The mandatory control for anything HC/mHC claims.
    Arm("trunk_plain_d2", _arch("trunk_plain_d2"), "1c-trunk-depth",
        "Does the depth-1 fusion head lose anything to depth at all?"),
    Arm("trunk_plain_d4", _arch("trunk_plain_d4"), "1c-trunk-depth",
        "Where does trunk depth saturate?", control="trunk_plain_d2"),
    Arm("trunk_plain_d8", _arch("trunk_plain_d8"), "1c-trunk-depth",
        "Where does trunk depth saturate?", control="trunk_plain_d4"),

    # -- Stage 1d/1e: orthogonal structural and likelihood changes.
    Arm("axial_cpg_d4", _arch("axial_cpg_d4"), "1d-axial-comethylation",
        "Does attending to genomic neighbours along the CpG axis add signal?",
        control="trunk_plain_d4"),
    Arm("beta_likelihood_head", _arch("beta_likelihood_head"), "1e-output-likelihood",
        "Does a bounded, heteroscedastic likelihood beat MSE on beta values?"),

    # -- Stage 1f: make the 2026-08 fusion verdict defensible at matched capacity.
    Arm("fusion_bilinear_rank512", _arch("fusion_bilinear_rank512"), "1f-fusion-retest",
        "Does interaction rank ABOVE the canonical 256 help? (rank 256 is provably "
        "identical to canonical -- see tests/test_architecture_variants.py)"),
    Arm("fusion_cross_attention_h4", _arch("fusion_cross_attention_h4"), "1f-fusion-retest",
        "Was cross_attention's 0.5329 a single-head artefact?"),

    # -- Stage 2: HC vs mHC vs the depth control, at matched depth/width.
    Arm("trunk_hc_n4_d4", _arch("trunk_hc_n4_d4"), "2-hyper-connections",
        "Do four unconstrained residual streams beat one?", control="trunk_plain_d4"),
    Arm("trunk_mhc_n4_d4", _arch("trunk_mhc_n4_d4"), "2-hyper-connections",
        "Does the Birkhoff/doubly-stochastic constraint recover what HC loses?",
        control="trunk_hc_n4_d4"),
    Arm("trunk_mhc_semantic_d4", _arch("trunk_mhc_semantic_d4"), "2-hyper-connections",
        "Do modality-identified streams make the conserved exchange operator useful?",
        control="trunk_mhc_n4_d4"),

    # -- Stage 3: only after both components clear the promotion threshold.
    Arm("combined_locus_attention_mhc_semantic", _arch("combined_locus_attention_mhc_semantic"),
        "3-combination", "Do the two novelty components compose?",
        control="trunk_mhc_semantic_d4"),
)

ARMS_BY_NAME = {arm.name: arm for arm in ARMS}
STAGES = tuple(dict.fromkeys(arm.stage for arm in ARMS))


def jobs(arms: tuple[Arm, ...] = ARMS) -> list[tuple[Arm, int]]:
    """Flatten arms into (arm, seed) units of work -- the schedulable granularity."""
    return [(arm, seed) for arm in arms for seed in arm.seeds]


def run_id(arm: Arm, seed: int) -> str:
    """Stable, path-safe identity for one unit of work.

    Deterministic on purpose: re-running the same arm/seed on any machine reopens
    the same run directory instead of creating a duplicate, so a suite spread
    across machines that share the output root merges by construction.
    """
    return f"arch-{STUDY}-{arm.name}-seed{seed}"


def select(names: str | None, stages: str | None, shard: str | None) -> tuple[Arm, ...]:
    """Resolve --arms/--stages/--shard into the arms this machine should run."""
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
        # Round-robin over the *arm* list so each machine gets a mix of cheap and
        # expensive arms rather than one machine drawing every deep-trunk run.
        chosen = tuple(a for i, a in enumerate(chosen) if i % total == index - 1)
    return chosen


def data_paths(data_root: str | None = None) -> dict[str, str]:
    """Machine-portable data locations.

    Defaults target this machine; ``METHYL_DATA_ROOT`` (or --data-root) retargets
    the whole set on another host, since the derived cache layout underneath is
    identical wherever prepare.py built it.
    """
    root = Path(data_root or os.environ.get("METHYL_DATA_ROOT", "/dune/DATASETS/MethylPredictionData"))
    prepared = root / "derived" / "methylprophet_table5_tcga_chr1"
    canonical = root / "datasets" / "methylprophet_repro_v1"
    return {
        "canonical_root": str(canonical),
        "prepared_root": str(prepared),
        "feature_cache": str(prepared / "features"),
        "rna_cache": str(prepared / "rna"),
        "registry": str(canonical / "cpg" / "registries" / "array_cpg_map.parquet"),
        "output_root": str(root / "experiments"),
    }
