# Paper experiment settings

This repo's headline model versions and any future baseline comparisons are organized around
**three canonical MethylProphet-matched settings**, plus one further-out setting that is not a
paper-comparison result. This document is the single place that states which is which and their
current status — read it before adding a new baseline or claiming a new "vs MethylProphet"
result, and update it (not just `results/reference/`) when a setting's status changes.

Any new baseline model must be trained and hyperparameter-optimized **independently in each of
the three settings below** (not tuned once and reused across settings) — the point of "matched
MP" is a like-for-like comparison against MethylProphet's own published/released per-setting
numbers, not a single number applied everywhere.

## Status: what's missing right now, and how to run it

**See [`EXPERIMENT_ROADMAP.md`](EXPERIMENT_ROADMAP.md) for the full, launchable list** — one
short ID per pending experiment (usable directly as `--run-id`), exact commands against the
**current** codebase (no `--engine generic`/`matched_chr1`, `configs/models/rna_methylation.yaml`,
or `InteractionConfig.kind` baseline switches remain), priority, dependencies, and where each
result lands. Summary: the 3 non-CpG-Prior paper baselines need a fresh chr1 run each (their
existing numbers are from the removed engine — see the "Baseline models" section below and
`results/reference/methylprophet_comparison/baselines_chr1.md`'s provenance note);
`architecture_novelty_2026_09` has arms finished-but-uncollected, stalled mid-training, and one
outside the queue entirely (including the missing chr1 full-budget convergence run); chr123 has
no isolated matched cache yet.

**Not "resume training"** — longer-term/blocked work, tracked separately, not on the roadmap
above: ENCODE atlas expansion (setting 3 below, blocked on NTv3 atlas coverage) and the
foundation-model baselines (blocked on integration work — position lookups, ID crosswalks — see
that section below), neither of which is a matter of just launching a training run.

## Baseline models

Four simplified baselines isolate why the reference architecture
(`configs/models/rna_methylation_locus_attention.yaml`) works, each trained/tuned independently
per setting above and reported alongside the headline MethylProphet comparison:

| Baseline | What it tests | Recipe | Architecture |
| --- | --- | --- | --- |
| CpG Prior | how much performance is explained by the intrinsic methylation tendency of each CpG alone, with no RNA input at all | `configs/models/baselines/baseline_cpg_prior.yaml` (documentation-only stub, zero parameters) | prediction = cached prior `mu` directly, via `CpGPriorEvaluator` (`rna_training/evaluator.py`) — no model, no training |
| Global RNA Shift | whether a single per-patient correction is sufficient, without any CpG-specific modeling of the RNA effect | `configs/models/baselines/baseline_global_rna_shift.yaml` | `FeatureFusionArchitectureVariantModel` with `use_mean_branch=False, include_raw_cpg=False, use_raw_product=False` (raw branch sees RNA only, ignores the CpG embedding entirely) |
| Bilinear RNA–CpG | a simple low-rank model of patient-locus interactions (shared latent space, dot product) | `configs/models/baselines/baseline_bilinear_rna_cpg.yaml` | `FeatureFusionArchitectureVariantModel` with `use_mean_branch=False, include_raw_rna=False, include_raw_cpg=False` (raw branch sees only the RNA×CpG product term) |
| MLP RNA–CpG | whether a standard nonlinear fusion model can substitute for the proposed explicit interaction term | `configs/models/baselines/baseline_mlp_rna_cpg.yaml` | `FeatureFusionArchitectureVariantModel` with `use_mean_branch=False, use_raw_product=False` (raw branch sees concatenated RNA+CpG, no product term) |

Results land under `results/reference/baselines/<name>/<scope>.yaml`, one file per
baseline per setting, schema-compatible with `results/reference/rna_methylation/<scope>.yaml` so
they drop into the same comparison tables. The narrative head-to-head table at
`results/reference/methylprophet_comparison/baselines_chr1.md` — **all four completed for chr1
2026-09-01** — reports numbers from the now-retired two-stage engine
(`VarianceNormalizedResidualModel` + `InteractionConfig.kind`), kept as frozen historical
provenance; reproducing them under the current engine (table above) requires a fresh training run
per baseline, not yet done as of this document's last update. None of these reuse numbers from
`results/reference/ablations.yaml`'s `fusion_mechanism_2026_08`/
`interaction_concat_and_latent_dim_2026_08` studies — those used different splits/engines/flag
combinations and cover chr1 only; the baselines above get fresh, independently-tuned runs.

## 1. TCGA chr1, matched MP — DONE, verified

- **Reference architecture (2026-09-03): shared-backbone.** `--engine
  matched_chr1_shared_backbone` (`rna_training/locus_cls_trainer.py`, `models.py::
  FeatureFusionLocusCLSModel` — see `docs/RNA_METHYLATION.md`), MAS-PCC 0.5627 on the true official
  split (`results/reference/rna_methylation/chr1.yaml`) — single seed, not yet run to convergence
  (stopped epoch 47/80 by user request), superseding the two-stage architecture's 0.5613 (kept
  under that file's `legacy_two_stage` key). Selected via the `shared_backbone_locus_cls_2026_09`
  ablation ladder (`results/reference/ablations.yaml`).
- The earlier two-stage architecture's own `--engine matched_chr1` (`benchmark/methylprophet/`'s
  `MethylProphetTrainer`, a frozen/isolated Cartesian-block trainer) has been **removed**; its
  0.5613 chr1 number remains frozen under `legacy_two_stage` above and in
  `results/reference/methylprophet_comparison/`, not reproducible by current code.
- Data: `derived/methylprophet_table5_tcga_chr1/` — a pre-extracted, well-chunked (per-source
  small HDF5 files, `chunks=(128, 2048)` for Array/EPIC) slice of the canonical bundle, built by
  `scripts/benchmark_methylprophet/prepare.py`. This is *not* the same data path as the
  scope-agnostic `shared_backbone` engine's own `chr1` scope reads from
  (`derived/rna_feature_cache/chr1` + the full genome-wide canonical bundle) — the two happen to
  share the same official Array split (see next bullet) but are otherwise separate, and the
  matched_chr1 files are dramatically faster to read (small, training-friendly chunking vs. the
  genome-wide bundle's per-row chunking).
- Split independently verified **exact** (ID-set match, not just counts) against MethylProphet's
  actual released chr1 evaluation artifact — see `docs/BENCHMARK_METHYLPROPHET.md` and
  `results/reference/methylprophet_comparison/chr1_official_split_verification.md`.
- Reference numbers: `results/reference/methylprophet_comparison/`, headline MAS-PCC in
  `docs/BENCHMARKS.md`.

## 2. TCGA chr1-3 (chr123), matched MP — CpG axis verified 2026-09-02, sample axis a reconstruction

- The existing checkpoint (`results/reference/rna_methylation/chr123.yaml`, 2026-08-31) was
  trained via the now-retired two-stage architecture's generic engine at `scope=chr123`, reading
  the full genome-wide canonical bundle — pending a rerun under the current reference architecture
  (`--engine shared_backbone`, no isolated `matched_chr123` cache exists yet either, unlike chr1;
  see `docs/CHR123_TRAINING_OPTIMIZATIONS.md` for the input-pipeline work already done for this
  scope).
- Checked against the released MethylProphet chr123 evaluation artifact
  (`MethylProphet/eval-tcga-mix-chr123-bs_512-32xl40s-aws-eval_on_tcga_chr123`, access approved
  and downloaded 2026-09-02). The CpG axis (`note1 ∪ note4`) matched exactly as already
  reconstructed — resolved. The **sample axis could not be verified the same way**: 306 of the
  release's sample_idx values don't exist anywhere in this repo's canonical Array HDF5 at all
  (release sample_idx range extends to 10,915, our bundle's only to 10,702) — a genuine content
  difference between the release's underlying snapshot and this repo's `241231` bundle, not an
  ID-extraction bug (the CpG axis, extracted the same way from the same rows, matched exactly).
  Applying the release's sample IDs broke `scripts/prepare.py --model cpg_statistics` outright, so
  `protocols/tcga_mix_chr123`'s sample idx files were reverted to the chr1-reused 8,260/918 split
  — the only one internally consistent with this repo's own data. See
  `docs/BENCHMARK_METHYLPROPHET.md`'s "chr123: verified" section and
  `results/reference/methylprophet_comparison/chr1_official_split_verification.md` for the full
  record.
- **Consequence**: none for existing checkpoints — `cpg_statistics/chr123.yaml` and
  `rna_methylation/chr123.yaml` (2026-08-31 runs) already used the chr1-reused split, which is
  unchanged by this investigation; no retrain needed. chr123's CpG-axis provenance is now on the
  same footing as chr1's (exact ID match); its sample axis remains a reconstruction, not a literal
  release-ID match, and that gap looks unresolvable without access to whatever raw snapshot
  produced the release's larger Array sample universe.
- Building an isolated `matched_chr123` cache mirroring chr1's (pre-extracted,
  `(128, 2048)`-chunked per-source files instead of routing through the genome-wide bundle) —
  this is what makes chr1 training/ablation runs fast; the generic engine reading the full
  genome-wide bundle directly is measurably much slower for the same total data volume (see the
  session notes behind the 2026-09-01 joint-training-ablation experiment for the measured
  before/after).

## 3. ENCODE, all chromosomes, matched MP — NOT YET BUILT

- Blocked on NTv3 embedding coverage: ENCODE's CpG universe is ~28.3M CpGs genome-wide, and only
  ~20.2% of it is in the current atlas (`ntv3_cpg_atlas_v1.h5`, built for TCGA's chr1-3 dense
  coverage, not ENCODE) — see `docs/PLANNING_NTV3_GENOMEWIDE_EXPANSION.md` for the exact gap
  analysis.
- Extraction pipeline exists and extension is in progress: `src/methylation_predictor/ntv3_atlas.py`
  (`prepare_missing_universe_from_positions`, generalized for external chrom/position lists like
  ENCODE's) + `scripts/build_ntv3_atlas.py`; live extraction artifacts under
  `derived/ntv3_expansion/` (shards, logs) as of 2026-09-01.
- MethylProphet's own released ENCODE evaluation artifact is already downloaded locally
  (`methylprophet_official/eval-encode_wgbs-bs_512-64xl40s-aws/`, raw ENCODE data in
  `methylprophet_official/raw-encode/`) — once atlas coverage is sufficient, this setting needs
  the same split-verification treatment chr1 got (`docs/BENCHMARK_METHYLPROPHET.md`) before any
  comparison number from it is reported.

## Foundation-model masked-CpG comparison (CpGPT / MethylGPT)

A separate, isolated reference point alongside the baselines above: feed the exact
same `val_cpg_x_train_sample` view (unseen CpGs, seen patients) to external, already-
pretrained masked-methylation foundation models -- no training on this repo's data at
all -- to see how much of the masked signal an off-the-shelf model recovers. Code
lives in `src/methylation_predictor/benchmark/foundation_models/` (isolated per the
same "not central" principle as the MethylProphet benchmark, `CLAUDE.md`), driven by
`scripts/benchmark_foundation_models/`. Vendored repos + checkpoints live in
`external/` (gitignored, `setup.sh` recreates it).

Models: **CpGPT** (`lucascamillomd/CpGPT`, DNA-sequence + position embeddings, no
fixed CpG vocabulary -- both released pretrained-masked sizes, `small`/`large`),
**MethylGPT** (`albert-ying/MethylGPT`, fixed 49,156-site Illumina-probe-ID
vocabulary -- all three released sizes, `base`/`medium`/`large`), and **DeepCpG**
(`cangermueller/deepcpg`, DNA-only submodel -- `hou2016_hcc_dna`/`hou2016_hepg2_dna`,
see below).

**On DeepCpG specifically -- correcting an earlier claim in this document (and one
made to the user in-session):** it was initially assumed both MethylProphet's and
CpGPT's papers use DeepCpG as an experimental baseline on TCGA/ENCODE bulk data, the
same setting as this pipeline. **Verified false** by reading both papers' full text
directly (2026-09-02): MethylProphet's paper cites DeepCpG only in related work, no
experimental comparison or reported numbers anywhere in its tables. CpGPT's paper
does compare against DeepCpG, but on a **single-cell** dataset (sciMETv3) with
**AUPRC/AUROC** classification metrics -- a different data regime (single-cell,
binary calls) than this pipeline's bulk, continuous `val_cpg_x_train_sample` task,
and it doesn't state whether DeepCpG was retrained or used off-the-shelf there.
**There is no paper precedent for DeepCpG in this exact setting** -- it is included
zero-shot, patient-agnostic (see below), purely as an additional reference point,
not as a reproduction of any published number. DeepCpG's own CpG-neighbor and Joint
modules were considered and excluded: they're hard-wired to the exact number/identity
of cells in their original single-cell training dataset, so they don't transfer to
arbitrary new bulk samples the way the DNA-only module (sequence window in,
methylation probability out) does.

**Status as of 2026-09-02 (readiness only, no full-split GPU inference run yet --
GPU was busy this session):**

- **chr1**: ready to run once GPU is free. `val_cpg_x_train_sample` for chr1 pulls
  directly from the existing split-verified `matched_chr1` cache (6,742 val CpGs x
  8,260 train-sample context, confirmed by `protocol.py` against the real cache).
- **CpGPT weight loading: verified clean.** Both `small` and `large` checkpoints
  pass `check_readiness.py` -- every tensor's fingerprint changes after loading
  (not randomly initialized) and a tiny sanity forward pass produces
  checkpoint-dependent output. The full evaluator (`CpGPTEvaluator`) still needs a
  genomic-position -> DNA-embedding lookup wired against
  `external/checkpoints/cpgpt_human_dependencies/` before it can actually predict
  (currently `NotImplementedError`, see `evaluator.py`).
- **MethylGPT weight loading: fixed, verified (`PASS_WITH_BENIGN_CAVEAT`).** All
  three released checkpoints (`base`/`medium`/`large`) were trained with
  `fast_transformer=True` (`FlashTransformerEncoderLayer`'s fused `Wqkv` attention
  projection); without flash-attn installed the model falls back to plain
  `nn.MultiheadAttention` (`in_proj_weight`/`in_proj_bias`). An earlier version of
  this check misdiagnosed that as a different parameterization requiring
  flash-attn -- direct shape comparison showed both are the identical
  `(3*d_model, d_model)`/`(3*d_model,)` fused QKV projection, just under a
  different key name. `methylgpt_adapter._remap_flash_attn_qkv_keys` now renames
  `Wqkv.{weight,bias}` -> `in_proj_{weight,bias}` before loading (assumes the
  standard Q/K/V-contiguous-blocks convention both implementations follow --
  not yet cross-checked numerically against a real flash-attn forward pass, do
  that once flash-attn is available on a GPU host). After the fix, 96/100 tensors
  per variant change on load; the remaining 4 (`cls_decoder._decoder.2/.5`) are
  present in the checkpoint under the same name/shape with a value that already
  equals a fresh model's default init -- confirmed by direct comparison
  (`check_readiness.py`'s `mismatch_diagnosis`), very likely an
  untrained/passthrough LayerNorm, not a skipped tensor.
  Separately, scoring arbitrary chr1/chr123 val-CpGs needs an Illumina
  probe-ID -> hg38-position crosswalk this repo doesn't have yet (our own CpG
  registry only carries (chrom, pos), not Illumina probe IDs) --
  `verify.vocabulary_overlap()` is ready for it, but has nothing to compute against.
- **DeepCpG weight loading: verified clean, both variants.** DeepCpG's own download
  host (and its own `dcpg_download.py`) is dead
  (`http://www.ebi.ac.uk/~angermue/deepcpg/alias/...`, confirmed HTTP 500); the live
  mirror used instead is Kipoi (`kipoi.org`) -> a stable Zenodo record (1466079).
  Both `hou2016_hcc_dna` and `hou2016_hepg2_dna` (Keras 1.2.2/TensorFlow 1.13.1,
  isolated `deepcpg-env` **conda** env -- no python3.7 interpreter available for a
  venv) load with MD5-verified weights, every weight array's fingerprint changes
  after loading, and a tiny forward pass on random one-hot DNA windows produces
  checkpoint-dependent output. Their original CpG positions were already lifted
  GRCh37 -> GRCh38 by the authors, matching this repo's own hg38 canonical bundle --
  no liftover needed. **Still needed before a real chr1 check**: an absolute path to
  an hg38 FASTA on this machine (not yet located/provided) to sanity-check a real
  DNA window via `check_readiness.py --model deepcpg --fasta-path ...` (the
  dummy-window check above already ran and passed without it). Because the model
  only sees a CpG's DNA sequence -- never any per-patient input -- its prediction is
  identical across every one of this repo's samples at a given CpG; report results
  as patient-agnostic, closer in spirit to this repo's own `CpG Prior` baseline than
  to CpGPT/MethylGPT's per-patient masked recovery. It was also trained on
  single-cell data, so zero-shot transfer to bulk TCGA methylation may simply score
  poorly -- a legitimate result to report, not a pipeline bug.
- **chr123**: usable once chr1 is running, inherits the same split-verification
  caveat as the repo's own chr123 numbers (see setting 2 above).
- **ENCODE**: blocked, same atlas-coverage gap as everything else (setting 3 above).
- **Published-number comparison**: `published_reference.py` is a structured but
  currently empty template (`CPGPT_PUBLISHED`/`METHYLGPT_PUBLISHED`, same shape as
  `TABLE5_PUBLISHED_METHYLPROPHET`) -- not pre-filled with numbers seen only in a
  secondhand search summary; fill in once read directly off each paper's own table.

## Methodology note: `mode=development` proxy split vs. the official MethylProphet split

Several trainers in this repo (`rna_training/joint_trainer.py`, `rna_training/locus_cls_trainer.py`,
the generic `rna_training/trainer.py`) support `mode="development"`, which carves an **inner**
train/val split *out of* the official training pool (`array_train_sample_idx`/`array_train_cpg_idx`)
via `stratified_sample_split`/`blocked_cpg_split`, so architecture/hyperparameter search never
touches the true held-out test split while iterating. Its three resulting views are named
`train_cpg_x_val_sample`/`val_cpg_x_train_sample`/`val_cpg_x_val_sample` — **the same names** the
official evaluation (`mode="final"`, evaluated against the true `array_val_sample_idx`/
`array_val_cpg_idx` from `benchmark/methylprophet/protocol.py`) uses for its own views. These are
**not the same data** despite the shared name: the official 0.5613 canonical number
(`results/reference/rna_methylation/chr1.yaml`) is always a `mode=final` number on the real
MethylProphet Table5 split; a development-mode `val_cpg_x_val_sample` is always a same-chromosome,
same-population proxy on a different (training-pool-internal) held-out subset. `results/reference/
ablations.yaml` disambiguates the two by using `inner_double_ood_mas_pcc` for the development-mode
proxy and `val_cpg_x_val_sample_mas_pcc` for the true official-split number — follow that
convention in any new ablation entry, and never report a development-mode number as if it were the
official benchmark figure. A promising architecture found via `mode=development` search needs a
follow-up `mode=final` run (trained on the full official training pool, evaluated on the true
official held-out split) before its number is comparable to 0.5613 in the paper.

## Internal architecture ablations (not paper-comparison settings)

Design/hyperparameter ablations that inform which architecture choices make it into the paper's
final model live in `results/reference/ablations.yaml` (see CLAUDE.md's results/reference
taxonomy) — status/conclusions are recorded there per study, not duplicated here. Notable ongoing
one as of 2026-09-02:

- **`shared_backbone_locus_cls_2026_09`** — **PROMOTED, no longer just an internal ablation**: this
  ladder's winner (rung B) is now the repo's primary/reference RNA-methylation architecture (see
  setting 1 above and `docs/RNA_METHYLATION.md`), kept here for the ladder methodology/history that
  produced it. A systematic, one-change-at-a-time ladder (rungs A-F) tested a postdoc-proposed
  single-stage shared-backbone architecture (`models.py::FeatureFusionLocusCLSModel`,
  `rna_training/locus_cls_trainer.py`) as a more elegant alternative to the two-stage frozen-prior
  pipeline (`RNAMethylationPredictor`, since removed -- see CLAUDE.md's "Model compatibility note") — mean-prediction proxy
  task and RNA-conditioned prediction late-fused into one head, instead of that model's explicit
  `logit(mu) + sigma*residual` composition. The ladder (rungs A-F, `mode=development`, see
  methodology note above) found only one real jump — adding the mean branch (rung A -> B) — with
  every further addition (residual probe, LR multiplier, init tweak, fusion product) statistically
  indistinguishable from rung B, so none earned a place in the final config. Rung B was then
  re-run in `mode=final` and evaluated on the true official split: **MAS-PCC 0.5627** on all three
  official views, slightly *above* the two-stage model's 0.5613 — and a lower bound, since that run
  was stopped by user request at epoch 47/80 while still improving. Single-seed, chr1 only, not yet
  run to convergence — a full-budget rerun (or a second seed) is worth doing before citing 0.5627
  as final; see `ablations.yaml`'s entry for full numbers/caveats.

- **`architecture_novelty_2026_09`** (in progress, retargeted 2026-09-04): an architecture-novelty
  suite prompted by a postdoc review (2026-09-03) that judged the architecture too simple for the
  paper's novelty claim despite its numbers. Originally built against the two-stage
  `RNAMethylationPredictor` (`models.py::ArchitectureVariantModel`); rebuilt on
  `FeatureFusionLocusCLSModel` once `shared_backbone_locus_cls_2026_09` (above) concluded and that
  model became primary. The two-stage generation and `ArchitectureVariantModel` have since been
  removed entirely (see CLAUDE.md's "Model compatibility note") -- the retired arms' numbers
  remain frozen in `results/reference/ablations.yaml`, not reproducible by current code. Targets
  the two axes never ablated on the primary
  architecture -- the raw branch's RNA encoder (still a single `Linear(25017 -> 256)`) and whether
  extra trunk depth helps at all (still one `Linear`) -- plus a bounded Beta likelihood head and
  windowed attention along the CpG axis. (A Hyper-Connections/mHC multi-stream trunk kind was also
  tried as part of this suite and removed entirely once measured -- see
  `results/reference/ablations/architecture_novelty_2026_09/README.md`'s "What was tried and
  removed" section.) Every arm runs at `mode=final` on the
  `matched_chr1_shared_backbone` engine, evaluated via `evaluate_official_split`, so its MAS-PCC is
  directly comparable to rung B's. Its noise-floor arm (3 seeds, full 80-epoch budget) doubles as
  the full-budget rerun `shared_backbone_locus_cls_2026_09`'s own entry above already flagged as
  needed before 0.5627 can be cited as final -- so this suite is also how that number gets
  resolved. Ablation-only/throwaway code (see file docstrings). **No number from this study is a
  paper result**, and nothing is adopted as canonical without clearing `2 x` the seed SD; if an arm
  ever is adopted, the baseline rule at the top of this document applies and it must be retrained
  independently in all three settings. See
  `results/reference/ablations/architecture_novelty_2026_09/README.md`.

## Future, not a paper-comparison setting: TCGA whole-genome + ENCODE merged

Work in progress, deliberately **not** one of the three settings above — MethylProphet itself
does not publish this combination, so it is this repo's own general benchmark rather than a
"matched MP" result. It is meant to eventually **replace** the current Array-only `genomewide`
scope (`results/reference/{cpg_statistics,rna_methylation}/genomewide.yaml`), not sit alongside it
as a fourth parallel scope — see `docs/PLANNING_NTV3_GENOMEWIDE_EXPANSION.md` ("goal 2") for the
redefinition plan. Depends on the same NTv3 atlas expansion as setting 3.
