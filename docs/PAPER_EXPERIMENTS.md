# Paper experiment plan — status and mapping

The normative source for the paper's experiment catalog, claims, and
storage/manifest conventions is
[`MethylPredictor_Paper_Experiments_and_Codebase_Plan.md`](../MethylPredictor_Paper_Experiments_and_Codebase_Plan.md)
(repo root). This document maps that plan onto what already exists in the
codebase, so the two do not drift apart.

## Split vocabulary mapping

The experiment plan uses the official MethylProphet split vocabulary. The
runtime implements the same four-way split but names it after the two axes
being resolved (`{cpg_scope}_cpg_x_{sample_scope}_sample`). No renaming has
been done — this table is the single point of translation:

| Paper plan (`MethylProphet` vocabulary) | Repository view name                | Meaning                         |
|------------------------------------------|--------------------------------------|----------------------------------|
| `id`                                      | `train_cpg_x_train_sample`           | in-distribution                  |
| `sample_ood`                              | `train_cpg_x_val_sample`             | new patients                     |
| `locus_ood`                               | `val_cpg_x_train_sample`             | new CpGs                         |
| `double_ood`                              | `val_cpg_x_val_sample`               | new patients + new CpGs (headline) |

The three officially evaluated views are declared in
`configs/data/storage_v2.yaml` under `run_contract.official_views`; the
headline view is `val_cpg_x_val_sample` (`double_ood`).

## Experiment catalog status

Tiering (`main` / `supplementary` / `architecture`) and IDs follow the paper
plan's sections 3–5. Status values: `implemented` (code path exists and is
exercised), `partial` (some code exists but a required axis/config is
missing), `missing` (no code path yet).

| ID  | Title                                   | Tier          | Status      | Notes |
|-----|------------------------------------------|---------------|-------------|-------|
| E01 | Official MethylProphet benchmark         | main          | implemented | `benchmark/methylprophet/protocol.py`, `configs/benchmark_methylprophet/` |
| E02 | Methylation imputation models benchmark  | main          | partial     | per-model status detailed below; substantially more built out than earlier assessed |
| E03 | Functional annotation contribution       | main          | implemented | `configs/paper_studies.yaml: functional_representation` (`full_functional`/`basic_context`/`minimal`), see note below |
| E04 | Functional vs generic genomic FM         | main          | implemented | `configs/paper_studies.yaml: locus_representation_genomic_fm` (`functional_full`/`genomic_fm_ntv3_pre`), see note below. GENA baseline still needs its own atlas + `GENOMIC_FM_VARIANTS` entry (only NTv3-pre wired so far) |
| E05 | CpG stability analysis                   | main          | missing     | analysis-only, no training; not yet implemented |
| E06 | Direct mean-proxy evaluation              | main          | partial     | mean-proxy head exists (`use_mean_proxy`); standalone eval-only script producing the E06 diagnostic schema not yet implemented |
| E07 | Mean-proxy supervision ablation           | main          | implemented | `configs/models/mean_contribution/{no_mean_branch,no_mean_supervision}.yaml`, `paper_studies.yaml: mean_proxy` |
| E08 | Structured residual vs direct prediction  | main          | implemented | config-only arm (`paper_studies.yaml: prediction_head`); see note below on what "structured" means here |
| E09 | CpG-only / correct RNA / shuffled RNA     | main          | missing     | eval-only controls not implemented |
| E10 | RNA gain vs CpG variability               | main          | missing     | eval-only, depends on E05/E09 |
| E11 | External validation                       | main          | missing     | no external dataset adapter; out of scope for this pass |
| E12 | Efficiency benchmark                      | main          | missing     | no compute/memory/throughput profiling module; out of scope for this pass |
| E13 | Biological/functional stratification      | main          | missing     | eval-only, depends on functional feature groups |
| E14 | Statistical robustness (multi-seed, CI)   | main          | partial     | multi-seed runs supported via `paper_studies.yaml`; bootstrap CI not implemented |
| S01 | Functional annotation group ablation      | supplementary | missing     | leave-one-group-out feature configs not yet defined |
| S02 | Context-specific vs aggregate annotations | supplementary | missing     | |
| S03–S08 | Stratified/error analyses            | supplementary | missing     | eval-only |
| S09 | Compute scaling                           | supplementary | missing     | depends on E12 infra |
| S10 | Oracle mean analysis                      | supplementary | missing     | diagnostic-only |
| A01 | Global RNA vs locus-conditioned RNA       | architecture  | implemented | `paper_studies.yaml: rna_interaction_scope` (`main.yaml` vs `baselines/functional_global_rna_shift.yaml`, already functional-matched) |
| A02 | Fusion block ablation                     | architecture  | partial     | prior J-series fusion variants exist as protected/legacy, not a clean 2–3 arm ablation |
| A03 | RNA program token count sweep             | architecture  | missing     | |
| A04 | Interaction/attention causal controls     | architecture  | missing     | |
| A05 | Existing architecture variants            | architecture  | partial     | historical J-series results exist in `configs/experiment_surface.yaml` provenance, not yet collected into a single table |
| A06 | Temperature/top-k sweeps                  | architecture  | missing     | |

P0 experiments per the plan (section 35) — **E01, E03, E04, E07, E08, E14** —
are the priority for making the repo launch-ready; see the implementation
plan tracked alongside this document for the concrete gaps closed in each
pass.

### E02 — methylation FM benchmark status (per model)

`benchmark/foundation_models/` (`adapters/`, `evaluator.py`, `protocol.py`,
`verify.py`, `published_reference.py`) plus `scripts/benchmark_foundation_models/
{setup.sh,check_readiness.py}` and `configs/benchmark_foundation_models/*.yaml`
already implement checkpoint vendoring, isolated per-model environments,
independent (fingerprint-based) checkpoint-loaded verification, and the
`val_cpg_x_train_sample` masked-recovery evaluation scaffolding, schema-compatible
with every other evaluator in this repo. This is materially more built out than a
readiness check; only the concrete blockers below remain, each requiring either
external data/environment access this sandbox does not have, or a scientific
judgment call:

- **DeepCpG** — `status: ready`. The only remaining gap was the hg38 FASTA path;
  `configs/benchmark_foundation_models/deepcpg.yaml` now points at
  `${METHYL_DATA_ROOT}/reference/hg38/hg38.fa` (the same convention used by the
  locus-features build), matching the FASTA already available on this machine's
  storage. Still needs `scripts/benchmark_foundation_models/setup.sh` run on a
  host with network access to vendor `external/deepcpg` and the legacy
  `deepcpg-env` conda environment (python=3.7/tensorflow==1.13.1/keras==1.2.2).
- **MethylGPT** — decision made 2026-09-13: **wait for a GPU host with flash-attn**
  before running masked-recovery evaluation. The non-flash-attn compatible-load
  remap (`_remap_flash_attn_qkv_keys`) lets the checkpoint load and pass
  `check_readiness.py`'s fingerprint check, but has not been verified bit-exact
  against the real flash-attn forward pass; per the paper's evidentiary priority
  (correctness over speed of results), no MethylGPT numbers should be reported
  until that verification is done on a GPU host.
- **CpGPT** — blocked on implementing the genomic-position -> DNA-embedding-mmap-row
  lookup (`_predict_cpgpt` in `evaluator.py`), which requires inspecting the
  schema of `ensembl_metadata.db`/`illumina_metadata.db` inside
  `external/checkpoints/cpgpt_human_dependencies` (a ~6.2GB download via
  `setup.sh`, not present in this sandbox). Needs either that download run
  somewhere reachable, or the DB schema pasted in, before this can be
  implemented and verified.

None of the three require new architectural work in this repo's own training
code — E02 is entirely an external-model integration effort, gated on
network/GPU access this session does not have.

### E03/E04 — implementation notes (2026-09-13)

**E03 (`functional_representation` study)**: `storage.MaskedFunctionalLocusCache`
wraps the existing functional locus cache and zeroes parts of its output
rather than duplicating the store or changing the model:
- `minimal`: only dense annotation indices 0-3 (CpG-island context) and 17
  (TSS distance) are kept; all sparse regulatory tracks, genomic-region/cCRE
  class, and breadth are zeroed.
- `basic_context`: all 18 static reference annotations kept (indices 0-17);
  sparse tracks and breadth (18-22) zeroed.
- `full_functional`: passthrough (`configs/models/main.yaml`, unchanged).

Selected via `locus_cls.feature_set` in the recipe (e.g.
`configs/models/functional_representation/minimal.yaml`), threaded through
`RNAMethylationTrainer`/`scripts/train.py --feature-set` is not a CLI flag —
it comes from the recipe, matching how `use_mean_branch` already works.
Model architecture, RNA branch, loss, and evaluator are byte-identical to
`main.yaml` across arms — only which locus-representation entries are
nonzero changes.

**E04 (`locus_representation_genomic_fm` study)**:
`modeling.genomic_fm.GenomicFMLocusPredictor` subclasses the frozen
`EfficientSingleAttentionPredictor`, overriding only `N_TRACKS=1`/
`DENSE_DIM=1536` so the same architecture (RNA branch, retrieval attention,
mean-proxy head, final regressor — all inherited unchanged) consumes a
dense NTv3-pre embedding instead of the sparse+dense functional
representation. `storage.GenomicFMLocusCache` reads the relocated
`derived/ntv3_pre_chr1_atlas/chr1_ntv3_pretrain_atlas_v1.h5` (see this
document's external storage reorganization note) and returns
`FunctionalLocusCache`-shaped output (always-empty `track_indices`/
`offsets`, `dense=embedding`). Launch with
`configs/models/genomic_fm/ntv3_pre_chr1.yaml` and `--genomic-fm-cache`
(new flag on `scripts/train.py`/`scripts/evaluate.py`, mutually exclusive
with `--locus-store`/`--functional-atlas`/`--annotation-cache`); the
standard `scripts/paper_experiment.py` wrapper auto-selects this flag from
`paths.genomic_fm_cache` when the recipe's `functional_fusion_variant` is a
genomic-FM variant. `modeling.factory.GENOMIC_FM_VARIANTS` currently has
only `genomic_fm_ntv3_pre`; a GENA baseline needs its own atlas + variant
entry before it can be added the same way. NTv3-post is not, and will never
be, a member of `GENOMIC_FM_VARIANTS`.

Both were verified end-to-end (forward pass shape checks, cache round-trip,
recipe loading, `_locus_cache_args` branching) with the real `torch`/`h5py`
environment (`methyl-predictor` conda env) in this session — see the
session transcript for the exact checks; no `pytest` run was possible
(no `pytest` installed in the environments this session could reach).

### Note on E08's "structured residual" arm

`main.yaml` does not literally sum a locus component and a patient-residual
component into the final logit (the mean-proxy head is a training-only
auxiliary target, per [`MODEL.md`](MODEL.md)). This repository's closest
faithful implementation of the plan's structured-vs-direct axis (sec. C3) is
therefore defined as: **structured** = both locus-shaping signals present
(mean-proxy auxiliary supervision `locus_cls.use_mean_branch/aux_weight`,
and the per-locus Pearson structuring term
`loss.locus_pearson_weight` applied to the final beta prediction) vs.
**direct** = both removed
(`configs/models/architecture_ablation/direct_prediction.yaml`), leaving
only plain beta MSE. This is a config-only ablation of the existing
architecture, not a new algebraic decomposition of the prediction head. If a
literal `final_logit = locus_logit + residual_logit` architecture is
required for the paper's C3 claim, that is new model code, not yet
implemented.

## External storage reorganization (`/dune/DATASETS/MethylPredictionData`)

A dry-run audit (`scripts/inventory_external_artifacts.py`, never destructive)
was run against the shared data root on 2026-09-13. Findings:

- **Fixed a code bug**: the script's profile-reference check did not know
  about `paths.locus_store` (the new genome-wide locus feature store path in
  `configs/data/paper_chr1.yaml`), so `derived/locus_features_v1` (20GB,
  actively used) was spuriously flagged `REVIEW`/unreferenced. Fixed to also
  read `legacy_frozen` and include `locus_store` in the checked keys.
- **Hard blocker on any physical deletion**: `derived/bulkrnabert_embeddings/tcga`
  (required by the `rna_encoder` paper study's BulkRNABert arm) does not
  exist on disk yet. Per `configs/external_cleanup_policy.yaml`
  (`destructive_actions_allowed: false`) and the prior `CLEANUP_AGENT_HANDOFF.md`,
  no destructive cleanup may be authorized while this dependency is missing.
  Regenerate with `scripts/prepare_bulkrnabert_embeddings.py` before any
  physical deletion.
- A forensic pass on the largest/most ambiguous REVIEW item,
  `derived/tcga_canonical` (generic name), found it actually contained
  `chr1_ntv3_pretrain_atlas_v1.h5` — a **chr1-only NTv3-pre embedding atlas**
  (`InstaDeepAI/NTv3_650M_pre`, 2,004,436 loci = full chr1 CpG universe,
  1536-d, window 32768bp) — exactly the artifact **E04's `genomic_fm_ntv3_pre`
  baseline needs**, just filed under a misleading directory name. It was
  **relocated (not copied)** to `derived/ntv3_pre_chr1_atlas/` before any
  deletion, verified byte-identical (same size, same HDF5 `model`/`n_total`
  attributes) in the new location, and registered as a `paper_runtime_keep`
  entry in `configs/external_cleanup_policy.yaml` and
  `configs/data/storage_v2.yaml`'s `keep_derived_roots`.
- A separate forensic pass on `derived/ntv3_expansion` (the largest single
  REVIEW item, ~52.6GB) confirmed by direct HDF5 checkpoint-attribute
  inspection (4 shards sampled across the file range) that it is
  **genome-wide, NTv3-***post*** (`InstaDeepAI/NTv3_650M_post`)** — explicitly
  excluded from the paper regardless of scope, and (per an explicit scope
  clarification: all FM/architecture comparisons are chr1-only) not needed
  in genome-wide form even if it had been pre-trained. No chr1-only subset
  exists inside it (loci are scattered across shards in coordinate order,
  not partitioned by chromosome), so nothing was extracted from it.
- **Physical deletion executed 2026-09-13**, under explicit user
  authorization (a backup exists on another machine), for the following
  confirmed-unreferenced items — verified via HDF5/checkpoint-provenance
  inspection and a full codebase reference scan (no active runtime, config,
  or paper-experiment code path referenced any of them; only
  cleanup-tracking documentation did):

  ```text
  derived/genomic_prior_v2                                    1.36 GB
  derived/hyenadna_expansion                                  1.04 GB
  derived/ntv3_functional_peak_atlas_chr1 (no _all_sources)   0.55 GB
  derived/ntv3_post_track_manifest                            2.64 GB
  derived/ntv3_pretrain_expansion                            12.40 GB
  derived/methylprophet_table5_tcga_chr1_pretrain_ablation    9.05 GB
  derived/rna_feature_cache_ntv3_pre                          0.13 GB
  derived/rna_feature_cache (retired, CLAUDE.md forbids reintroducing) 18.17 GB
  model_weights_cache/ntv3_650m_post_hf_cache (confirmed NTv3-POST)     8.98 GB
  derived/ntv3_expansion (confirmed genome-wide NTv3-POST)    52.62 GB
  derived/tcga_canonical (post NTv3-pre-atlas relocation)     0.13 GB
  ────────────────────────────────────────────────────────────────────
  TOTAL RECLAIMED                                            107.08 GB
  ```

  Each deletion was preceded by a `du -sb` size check and followed by an
  existence check confirming removal; nothing else under
  `/dune/DATASETS/MethylPredictionData` was touched. Full record (paths,
  byte counts, authorization) kept in `configs/external_cleanup_policy.yaml`'s
  `deleted_2026_09_13` entry for provenance.
- `derived/methylprophet_compact_chr123` was kept (not deleted): it is
  referenced by `PROJECT_SPECS.md` and `docs/MEAN_CONTRIBUTION_EXPERIMENTS.md`
  as the live chr123-scope data preparation target; it only looked
  unreferenced because the chr1-only `paper_chr1.yaml` profile doesn't list
  it. Recorded as `keep_pending_chr123_scope` in `storage_v2.yaml`.
- **Still blocked**: `derived/bulkrnabert_embeddings/tcga` (required by the
  `rna_encoder` paper study's BulkRNABert arm) does not exist on disk yet.
  This is unrelated to the cleanup above (it's a missing generation, not a
  deletion target) — regenerate with `scripts/prepare_bulkrnabert_embeddings.py`
  before running that study arm.
- **Layout anomaly fixed 2026-09-13**: `experiments/runs/runs/{cpg_statistics,locus_cls_joint}/chr1`
  (double-nested `runs/runs/`, from a past run with a mistakenly nested
  `--output-root`) held 22 genuine, unique historical run directories (21
  J-series architecture-search runs — the `rung_a`-`rung_f` ladder,
  `ffn-fusion-*`, `functional-branch-depth-8ffn-*`, etc. — plus one
  `cpg_statistics` run) that were invisible to the standard 3-level
  `experiments/runs/<storage>/<scope>/<run>` scan and caused a
  `[cleanup-inventory] WARNING: unexpected run layout(s)`. Verified no name
  collisions against the correct-path siblings, then moved (not copied) each
  of the 22 up one level to its correct path; the emptied `experiments/runs/runs/`
  tree was removed. Re-running the inventory confirms the warning is gone
  and all 22 now correctly classify as `PROTECT` (historical J-series runs,
  useful for the plan's A05 "existing architecture variants" table) — no
  data was deleted, only relocated.
- **`experiments/runs/locus_cls_joint/` deleted 2026-09-13** (21GB, ~106
  historical pre-refactor runs — J-series architecture search, superseded
  mean-proxy/RNA-encoder dev runs), under explicit user authorization.
  Compact provenance (metadata, training summary, evaluation metrics) for
  the 20 runs with complete records was archived first via
  `scripts/plan_historical_archive.py --write-archive` into
  `experiments/archive/historical_run_metadata/` (verified non-empty,
  spot-checked) before deletion; the other ~86 runs were `INCOMPLETE`
  (never reached a full evaluation) or `MISSING_METADATA`, nothing useful
  lost. The live/current `experiments/runs/rna_methylation/` namespace
  (including the in-progress `main-seed17-r3`/`main-seed42-r3` chr1 final
  runs) was not touched.
- Every launched experiment run should now be logged in
  [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) (date, study, arm, seed, run ID,
  launching machine) — see the logging rule added to `CLAUDE.md`.
- E04's data-side blocker is now resolved: the chr1 NTv3-pre embedding
  exists at `derived/ntv3_pre_chr1_atlas/chr1_ntv3_pretrain_atlas_v1.h5`.
  Wiring it into `modeling/factory.py`/`configs/features/` as a selectable
  locus feature-set arm (replacing the functional annotations with this
  embedding, everything else held constant) is still open work — see this
  document's E04 status row above.

No other files under `/dune/DATASETS/MethylPredictionData` were moved,
renamed, or deleted beyond what is listed above. The target layout matches
`configs/data/storage_v2.yaml` (Data/Results V2 contract); no further
physical reorganization of the KEEP set is needed.

## Run/result storage

The plan's `run_manifest.json` / feature manifest / result-warehouse
conventions (sections 9–25) already have an equivalent, frozen implementation
in this repository:

- run/record layout, required files, URI scheme: `configs/data/storage_v2.yaml`
- run manifest + checkpoint hashing: `src/methylation_predictor/run_store.py`
- shared/dune-mounted artifact resolution: `src/methylation_predictor/artifact_uri.py`
- multi-machine run coordination: `src/methylation_predictor/experiment_coordination.py`
- arm/seed matrix per study: `configs/paper_studies.yaml`
- aggregate result collection: `scripts/collect_paper_runs.py`

New experiments from the paper plan should be added as new `studies:` entries
in `configs/paper_studies.yaml` rather than a parallel registry, and should
reuse the existing `methyl-data://` URI convention to keep heavy artifacts
(checkpoints, prediction matrices) on the shared `/dune/...` mount while only
lightweight manifests/metrics stay in the repo-tracked `results/` tree.
