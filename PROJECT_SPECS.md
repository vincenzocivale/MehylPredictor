# MethylPredictor publication specifications

**Status:** living specification for the cleaned public repository  
**Last updated:** 2026-09-06  
**Normative language:** **MUST**, **SHOULD**, and **MAY** describe release requirements.

## 1. Project objective

MethylPredictor predicts a patient's DNA-methylation profile from bulk RNA expression and
a patient-independent functional representation of each CpG locus. The scientific problem is
patient-by-locus prediction: the model must generalize both to unseen patients and to unseen
CpGs, rather than merely reconstructing a population-average methylation profile.

The paper-facing RNA model does not require a genomic foundation-model embedding as a model
input. CpG loci are represented from a sparse functional regulatory atlas and dense static /
regulatory-breadth annotations. Genomic-FM artifacts retained elsewhere in the repository are
historical, benchmark, metric-only, or compatibility assets rather than dependencies of the
current RNA predictor.

The repository must make three things independently reproducible:

1. the immutable input-data contract and released MethylProphet-compatible splits;
2. every transformation from canonical inputs to derived caches;
3. training, model selection, evaluation, and the reported tables.

Large datasets, model weights, caches, and run directories MUST NOT be committed to Git.
The public repository must instead contain download/build commands, schemas, checksums,
small manifests, and frozen aggregate results that are legally redistributable.

## 2. Public-codebase requirements

- Code, identifiers, CLI help, comments, docstrings, configuration keys, and maintained
  documentation MUST be in English and use scientifically precise names.
- Public entry points SHOULD be limited to data validation/preparation, training, tuning,
  evaluation, and explanation. One-off queue runners and abandoned experiment scripts must
  be removed after their commands and results have been captured in provenance records.
- There MUST be one canonical implementation per active model component. Compatibility code
  for released checkpoints may remain only if it is clearly labelled and regression-tested.
- Every paper result MUST resolve to a recipe, seed, data manifest, Git commit, checkpoint
  hash, evaluation manifest, and a machine-readable result row.
- Generated runs and searches belong under the external data root and remain gitignored.
  Only compact frozen reference tables belong under `results/reference/`.
- Tests MUST cover shapes, masks, split integrity, exact cache equivalence, checkpoint
  compatibility, determinism where promised, and CPU/CUDA forward-backward smoke tests.
- The release MUST include installation instructions, pinned dependency/lock information,
  supported Python/CUDA versions, a software license, citation metadata, a model card, a data
  card, and an explicit statement of data-access restrictions.
- Dead code is not defined as “code we currently dislike”. Before removal, references from
  recipes, checkpoints, tests, documentation, and result provenance MUST be audited.

## 3. Scientific architecture status

### 3.1 Current reference model

The paper-facing RNA runtime is functional-locus only. Its public trainer is
`RNAMethylationTrainer`; the reference recipe is `configs/models/main.yaml`.

The model has two information streams:

```text
CpG locus
  -> sparse regulatory-track atlas
  -> dense static + regulatory-breadth annotations
  -> functional locus representation h_c
  -> training-only mean-proxy head

patient RNA x_p
  -> RNA encoder
  -> K learned program tokens R_p

h_c
  -> locus-conditioned cross-attention over R_p
  -> patient/locus RNA context r_p,c

[h_c ; r_p,c]
  -> final regressor
  -> beta_hat_p,c
```

The functional locus representation is patient-independent. Patient specificity enters through
the RNA representation and locus-conditioned retrieval.

The mean-proxy auxiliary head reads only the functional locus representation. It supervises
locus-level methylation tendency during training but its scalar prediction is not inserted into
the final beta prediction.

The RNA workflow does not load genomic/FM embeddings. `--prior-cache`, when supplied, is
metric-only and is used for `prior_mse` and `skill_vs_prior`. The minimal prior-cache contract is
`cpg_idx.npy + prior.npy`.

The historical storage identifier `locus_cls_joint` and temporary trainer/evaluator aliases are
retained only for active-checkpoint resume compatibility until the architecture-search J-series
has finished. They are not part of the intended final public API.

### 3.2 Claims that must remain separate

- **Supported:** locus-conditioned RNA attention is the dominant improvement over the old
  global linear RNA encoder on the matched chr1 experiment.
- **Not yet supported:** learned attention weights are biologically causal explanations.
- **Not independent replication:** chr123 contains chr1. Results on chr123 cannot be described
  as confirmation on an independent chromosome set unless chr2+3 are reported separately.
- Architecture selection must use an inner development split. A fresh locked chromosome
  scope or external cohort is required for an unbiased final confirmation because the
  official chr1 evaluation was inspected during development.

## 4. Data-layer contract

The data root is external to Git and is passed explicitly with CLI arguments or environment
configuration. The intended layout is:

```text
MethylPredictionData/
  datasets/                 # immutable canonical inputs and frozen protocols
    methylprophet_repro_v1/
  derived/                  # reproducible caches; never treated as source data
  experiments/
    runs/                   # final and ablation runs
    searches/               # hyperparameter/model-selection runs
  model_weights_cache/      # downloadable third-party weights; disposable
  reference/                # external reference genome assets
```

### 4.1 Immutable TCGA canonical bundle

`datasets/methylprophet_repro_v1/` is the frozen source of truth. Training code must open it
read-only and must never alter, regenerate, deduplicate, or silently filter its contents.

| artifact | relative path | shape/dtype | role |
| --- | --- | --- | --- |
| RNA | `rna/tcga_rna_official_full.h5` | `10,916 x 25,017`, float32 | canonical gene matrix and ordering |
| Array beta | `methylation/tcga_array_official_full.h5` | `9,178 x 408,399`, float32 | primary train/evaluation source |
| EPIC beta | `methylation/epic_full.h5` | `1,706 x 740,296`, float32 | auxiliary training source |
| WGBS beta | `methylation/wgbs_full.h5` | `32 x 23,047,052`, float32 | auxiliary training source |
| NTv3 atlas | `cpg/ntv3/ntv3_cpg_atlas_v1.h5` | `5,723,092 x 1,536`, float16 | historical/benchmark FM asset; not an RNA-model input |
| registries | `cpg/registries/*_cpg_map.parquet` | source-specific | CpG ID/coordinate mapping |
| protocols | `protocols/<name>/` | JSON + NumPy ID arrays | immutable splits of record |

The retained NTv3 atlas was generated with `InstaDeepAI/NTv3_650M_post`, hg38, a
32,768-bp forward-orientation window, and central-C/G mean pooling. These choices remain part
of the provenance of experiments that used the atlas, but the paper-facing RNA runtime no
longer consumes these embeddings.

Identifiers MUST retain their exact meanings:

- `cpg_idx`: global MethylProphet CpG namespace;
- `sample_idx`: patient/sample identifier shared with RNA;
- `measurement_idx`: source-local measurement identifier.

WGBS contains 32 measurements from 31 unique patients. The repeated measurement is real and
MUST NOT be deduplicated. Missing RNA or CpG IDs must raise an error rather than being
silently dropped or imputed.

The bundle's `manifest.json`, artifact SHA-256 values, and `FROZEN_VALIDATED` status are the
authority for schema and integrity. Absolute paths embedded in old derived manifests are
historical provenance only; public builders should record portable bundle IDs plus hashes.

### 4.2 Frozen protocols and split semantics

Supported protocol definitions are:

- `tcga_array_chr1`;
- `tcga_array_epic_chr1`;
- `tcga_array_wgbs_chr1`;
- `tcga_mix_chr1`;
- `tcga_mix_chr123`;
- `array_genomewide`.

The official Array sample split is 8,260 train and 918 validation patients. Array CpG splits
are 33,885/6,742 on chr1 and 78,211/14,893 on chr123. Array splits are exact. EPIC and WGBS
auxiliary pools use the validated `241231` source revision and are
source-revision-compatible rather than bit-identical to every historical MethylProphet
snapshot; this distinction must remain visible in manifests and paper text.

Evaluation MUST report the three views separately:

- `train_cpg_x_val_sample`: patient generalization;
- `val_cpg_x_train_sample`: locus generalization;
- `val_cpg_x_val_sample`: joint/double holdout.

The repository must not call EPIC or WGBS “official held-out evaluation” data: in the
current TCGA protocols they are auxiliary training sources, while held-out evaluation is
defined on Array.

### 4.3 ENCODE status

ENCODE/WGBS is an intended external evaluation direction, but it is not interchangeable
with the TCGA WGBS auxiliary source and is not yet represented as a complete frozen training
protocol in the canonical TCGA bundle. The public specification must label ENCODE work as
planned until it has:

- a versioned acquisition manifest and accession list;
- sample/assay inclusion criteria and tissue mapping;
- hg38 coordinate and CpG-namespace mapping;
- RNA availability or an explicitly defined cross-domain evaluation task;
- leakage-safe splits and a documented relationship to MethylProphet's released setup;
- license/terms-of-use documentation and reproducible download/preparation scripts.

Do not state that the current model was trained on “TCGA and ENCODE” unless such a run and
its manifest actually exist.

## 5. Derived datasets and caches

Derived artifacts are immutable for a run but reproducible from a canonical bundle. Every
derived directory MUST contain a schema version, source hashes, creation command/version,
scope, axis hashes, shapes, dtypes, and any split-dependent fitting policy.

### 5.1 CpG statistics

`derived/cpg_statistics/{chr1,chr123,genomewide}/` contains:

```text
cpg_idx.npy
target_mu.npy
target_sigma.npy
official_train_mask.npy
official_val_mask.npy
source_counts.npz
manifest.json
```

For chr1/chr123, `target_mu` is the beta-space mean and `target_sigma` is the standard
deviation of clipped `logit(beta)`, aggregated with the declared source policy. Official
Array validation patients are excluded. The model's mean-proxy loss must use the official
scope (`cpg_statistics/chr1` or `/chr123`), never an expanded/imputed feature universe.

The expanded `chr1_rna_universe` and `chr123_rna_universe` caches, when present, are feature
lookup artifacts. Their imputed statistics MUST NOT supervise the auxiliary mean loss.

### 5.2 chr1 matched preparation

`derived/methylprophet_table5_tcga_chr1/` is the legacy-compatible matched chr1 preparation.
Its relevant live contents are:

- compact Array/EPIC methylation subsets;
- normalized RNA cache (`rna_zscore.f16.npy`) and fitted normalization statistics;
- metric-only prior information when prior-relative metrics are requested;
- protocol IDs and manifests;
- evaluation adapter only if a maintained evaluation path still consumes it.

The feature cache contains approximately 2.0M CpGs and is much larger than the 40,627 Array
train+validation loci because it includes auxiliary EPIC/WGBS chr1 loci. WGBS remains read
directly from the canonical column-major file because a second dense copy provides little
read-speed benefit.

Before publication, the active chr1 loader should be made to consume the canonical bundle
and clearly named derived caches through one path. Historical `table5_genomic_prior` and
evaluation-adapter artifacts may be removed only after confirming that no supported recipe,
checkpoint reproduction, or evaluation test references them.

### 5.3 chr123 protocol-ordered target cache

`derived/methylprophet_compact_chr123/` materializes only the immutable methylation targets
on axes repeatedly used by `tcga_mix_chr123`:

```text
array.h5     # 9,178 x 93,104
epic.h5      # 1,706 x 172,723
wgbs.h5      # 32 x 5,396,437; duplicate measurement retained
manifest.json
```

Files contain `beta`, `sample_idx`, `cpg_idx`, and `measurement_idx` where applicable.
`manifest.json` records sample/CpG axis hashes. The cache is approximately 5.1 GiB and is
derived—not a replacement for the canonical sources.

Build it with:

```bash
python scripts/prepare_chr123_compact.py \
  --canonical-root /path/to/MethylPredictionData/datasets/methylprophet_repro_v1 \
  --output /path/to/MethylPredictionData/derived/methylprophet_compact_chr123
```

The builder is restartable and validates completed source caches before reuse. A release
test must compare representative Array, EPIC, and WGBS blocks to canonical values with exact
equality, including NaN masks.

### 5.4 Legacy genomic/FM feature caches

Historical protocol caches may contain ordered `cpg_idx`, 1,536-D NTv3 embeddings,
`prior`, and `sigma` arrays.

They are not inputs to the paper-facing RNA predictor.

During the transition, an old feature-cache directory may still be passed as `--prior-cache`
when it contains the two metric-only arrays required by evaluation:

```text
cpg_idx.npy
prior.npy
```

Embedding and sigma arrays in that directory are not opened by the current RNA workflow.

These historical caches must not be deleted until active J-series checkpoints no longer need
resume/evaluation compatibility. After final architecture selection, external-cache cleanup is
performed from a dry-run dependency inventory rather than directory names alone.

### 5.5 Reproducible data-creation sequence

The release must expose the following dependency order. A later stage must consume only
manifests and artifacts produced by an earlier stage; it must not rediscover splits or fit
statistics implicitly during training.

```text
authorized TCGA inputs + frozen protocol definitions
  -> canonical bundle validation
  -> split-safe CpG mean targets
  -> normalized RNA cache
  -> functional regulatory atlas + annotation cache
  -> optional protocol-ordered compact methylation cache
  -> RNA training
```

#### Canonical bundle

The current frozen bundle is fully inventoried and hash-validated, but reconstructing it
from the original authorized downloads is **not yet a supported public top-level workflow**.
The recovered `provenance/build_protocols.py` inside the local data bundle is provenance,
not an adequate release entry point. Before publication, the repository must provide either:

1. a legally usable acquisition-and-build command from versioned source accessions; or
2. a validator/importer for user-obtained files, with exact expected filenames, schemas,
   checksums, and a clear statement that raw data cannot be redistributed.

Until then, the derived-data pipeline is reproducible from the frozen bundle, but the full
raw-to-canonical pipeline is not. This limitation must not be hidden by a README example.

#### NTv3 atlas

The existing atlas is frozen historical/benchmark infrastructure and should normally be
validated rather than regenerated. It is not required by the paper-facing RNA model. For
reproduction of experiments that explicitly use NTv3, extending it to a new protocol uses
three explicit phases:

```bash
python scripts/build_ntv3_atlas.py prepare-universe-tcga \
  --canonical-root "$CANONICAL_ROOT" \
  --base-embeddings "$CANONICAL_ROOT/cpg/ntv3/ntv3_cpg_atlas_v1.h5" \
  --protocol tcga_mix_chr123 \
  --output "$DERIVED_ROOT/ntv3_universe_chr123"

python scripts/build_ntv3_atlas.py extract-worker \
  --universe "$DERIVED_ROOT/ntv3_universe_chr123" \
  --fasta "$HG38_FASTA" --output "$DERIVED_ROOT/ntv3_shards_chr123" \
  --rank 0 --world-size 1 --device cuda --compile

python scripts/build_ntv3_atlas.py merge \
  --universe "$DERIVED_ROOT/ntv3_universe_chr123" \
  --shards "$DERIVED_ROOT/ntv3_shards_chr123" \
  --output "$DERIVED_ROOT/ntv3_merged_chr123"
```

Appending merged rows to the frozen atlas is intentionally not automated. A new atlas
version must be written and reviewed instead of mutating `ntv3_cpg_atlas_v1.h5` in place.

#### Split-safe CpG mean targets

For each scope, build targets from the canonical protocol while excluding official Array
validation patients:

```bash
python scripts/prepare.py --model cpg_statistics \
  --canonical-root "$CANONICAL_ROOT" \
  --registry "$CANONICAL_ROOT/cpg/registries/array_cpg_map.parquet" \
  --scope chr123 --sources array epic wgbs \
  --policy sample_weighted --aux-sample-policy exclude_array_validation \
  --output "$DERIVED_ROOT/cpg_statistics/chr123"
```

Use `--scope chr1` and a distinct output directory for chr1. The resulting `target_mu.npy`
is the supervision used by the current model's mean-proxy task. A trained
`CpGStatisticsPredictor` and exported predicted `mu/sigma` cache belong to older/static
workflows and are not prerequisites for the current functional-locus RNA model. The release
documentation must keep these workflows separate.

#### Matched chr1 caches

The exact chr1 preparation constructs frozen protocol IDs, compact Array/EPIC matrices,
split-fitted RNA normalization, protocol-aligned features, and an evaluation adapter:

```bash
python scripts/benchmark_methylprophet/prepare.py \
  --canonical-root "$CANONICAL_ROOT" \
  --atlas "$CANONICAL_ROOT/cpg/ntv3/ntv3_cpg_atlas_v1.h5" \
  --hg38-fasta "$HG38_FASTA" \
  --config configs/benchmark_methylprophet/reference.yaml \
  --output "$DERIVED_ROOT/methylprophet_table5_tcga_chr1" \
  --device cuda
```

RNA z-score mean and scale are fitted on the 8,260 Array training patients only. EPIC/WGBS
RNA rows are transformed with those frozen statistics; the 918 held-out Array patients must
not contribute to normalization.

#### chr123 compact targets

After the canonical bundle and protocol exist, build the cache described in section 5.3.
It changes physical storage/order only. Training must still use
`$DERIVED_ROOT/cpg_statistics/chr123` for the mean auxiliary target and a feature cache that
covers every protocol CpG.

Every command above must write its full invocation, code revision, input hashes, output
hashes, row/column axes, and completion status to a manifest. Shell variables shown here are
placeholders, not hidden defaults.

## 6. Optimized training pipeline

### 6.1 Original bottleneck

A pair-complete chr123 epoch contains 2,237 optimizer steps and about 1.11 billion
sample-CpG pair slots. Protocol CpGs were sparse in the physical columns of the genome-wide
HDF5 files, so a logical block triggered expensive fancy selections. Profiling showed about
886 seconds waiting for data versus 203 seconds of GPU compute per epoch. The bottleneck was
HDF5 selection/decode and CPU preparation, not raw disk bandwidth.

### 6.2 Optimizations

The optimized path combines:

1. **Protocol-ordered compact targets.** Repeated logical blocks become dense HDF5 slices.
2. **Physical-locality ordering.** With `schedule_layout: contiguous_blocks`, pools are
   ordered by physical HDF5 row/column positions before block construction.
3. **Bounded parallel prefetch.** Multiple workers overlap HDF5 reads, NaN scans, feature/RNA
   lookup, conversion, and pinned-memory preparation with GPU work.
4. **Non-blocking host-to-device copies.** Prepared tensors are pinned and transferred
   asynchronously where supported.
5. **Deferred scalar synchronization.** Training diagnostics remain on device until block
   aggregation instead of forcing repeated CUDA synchronization.
6. **Mixed-precision compute.** BF16 autocast, TF32 matmul, and fused AdamW are enabled on
   supported GPUs.

Recommended bounded settings on the reference server are:

```yaml
training:
  schedule_layout: contiguous_blocks
  prefetch_depth: 8
  prefetch_workers: 4
  hdf5_cache_mb: 256
  amp: true
  amp_dtype: bfloat16
  allow_tf32: true
  fused_adamw: true
```

`prefetch_workers <= prefetch_depth` is validated. The queue is bounded, so at most
`prefetch_depth` blocks are prepared or in flight; this is essential for predictable RAM
use with large WGBS blocks. Futures are consumed in submission order, preserving optimizer
step order. Peak RAM and VRAM must still be benchmarked per machine and recipe.

### 6.3 Measured effect and correctness boundary

For the chr123 simple model on the reference server:

- epoch time fell from 1,076 s to 146 s (7.4x);
- measured data wait fell from about 886 s to 0.28 s;
- throughput reached 7.62M pair slots/s;
- GPU utilization rose to approximately 89–93%;
- representative compact reads were about 1,290x faster for Array, 109x for EPIC, and
  1.1x for WGBS.

These are machine-specific measurements, not guaranteed model speedups. The optimization
preserves the same source cells, NaN masks, loss, pair-complete coverage, and number of
optimizer updates. Locality-based minibatch grouping is not bit-identical to the previous
sparse-I/O ordering, so “same mathematical training protocol” is accurate while
“bitwise-identical optimization trajectory” is not.

Every new scope must record `data_wait_seconds`, `cpu_prepare_seconds`, `h2d_seconds`,
`compute_seconds`, `pair_slots_per_second`, peak host RAM, and peak VRAM for at least one
full epoch. Cache and canonical blocks must be tested for exact equality before a production
run.

### 6.4 Reuse on other scopes

- **chr1:** the matched preparation already applies compact Array/EPIC storage. WGBS's
  32-row column-major canonical layout is already efficient. Prefetch can still be used,
  but another WGBS cache is unlikely to help.
- **Other fixed chromosome subsets:** the approach is appropriate when a stable sparse
  subset is revisited for many epochs. The current builder is chr123-specific and should be
  generalized with explicit scope/axis arguments before it becomes a public API.
- **Genome-wide:** compact copying may be wasteful. Benchmark it against canonical reads and
  prefer `axis_full_coverage` when enumerating the full Cartesian product is computationally
  inappropriate.
- **Sparse or one-pass datasets:** a dense compact copy is usually a poor trade-off.

## 7. Training, selection, and evaluation policy

- `scripts/train.py` is the canonical training entry point; `scripts/evaluate.py` is the
  canonical evaluation entry point.
- `mode=development` must create an inner split entirely inside the official training
  universe. Hyperparameters and architecture are selected on inner double-OOD MAS-PCC.
- `mode=final` trains on the full permitted training pool. Official held-out metrics must not
  be inspected for early stopping or checkpoint selection.
- The primary endpoint and selection rule must be declared before the final experiment.
  Recommended primary endpoint: `val_cpg_x_val_sample/MAS-PCC`.
- Every final evaluation must log all three views as separate W&B namespaces and save the
  same values in `evaluation/<scope>/metrics.json`.
- RNA normalization, CpG-statistic fitting, feature generation, and cache construction must
  fit only on data permitted by the relevant split. Automated leakage tests are required.
- At least three paired seeds are required for architectural claims; report mean, standard
  deviation, confidence intervals, and patient/locus bootstrap intervals.

The intended run contract is:

```text
results/experiments/runs/<model>/<train-scope>/<run-id>/
  config.resolved.yaml
  metadata.json
  checkpoints/{best.pt,last.pt}
  training/{history.json,summary.json}
  evaluation/<eval-scope>/{metrics.json,manifest.json,...}
  wandb/ or logs/
```

Metadata must include data/protocol IDs and hashes, feature and normalization provenance,
source checkpoint hashes, Git commit/dirty state, architecture label, seed, scheduler,
precision, hardware, and software versions.

### 7.1 Optimized reference training commands

The optimized settings in section 6 are recipe fields. The exact recipe passed to these
commands must therefore contain `contiguous_blocks`, bounded prefetch, BF16/TF32, and fused
AdamW; the CLI currently does not override prefetch settings.

Matched chr1:

```bash
python scripts/train.py --model rna_methylation --scope chr1 \
  --engine matched_chr1_shared_backbone --mode final \
  --recipe configs/models/rna_methylation_locus_attention.yaml \
  --canonical-root "$CANONICAL_ROOT" \
  --prepared-root "$DERIVED_ROOT/methylprophet_table5_tcga_chr1" \
  --feature-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/features" \
  --rna-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/rna" \
  --registry "$CANONICAL_ROOT/cpg/registries/array_cpg_map.parquet" \
  --cpg-targets-dir "$DERIVED_ROOT/cpg_statistics/chr1" \
  --output-root "$EXPERIMENT_ROOT" --run-id <declared-run-id>
```

Matched chr123 with the protocol-ordered target cache:

```bash
python scripts/train.py --model rna_methylation --scope chr123 \
  --engine shared_backbone --mode final \
  --recipe configs/models/rna_methylation_locus_attention.yaml \
  --canonical-root "$CANONICAL_ROOT" \
  --prepared-root "$DERIVED_ROOT/methylprophet_compact_chr123" \
  --feature-cache "$DERIVED_ROOT/rna_feature_cache/chr123" \
  --rna-cache "$DERIVED_ROOT/methylprophet_table5_tcga_chr1/rna" \
  --registry "$CANONICAL_ROOT/cpg/registries/array_cpg_map.parquet" \
  --cpg-targets-dir "$DERIVED_ROOT/cpg_statistics/chr123" \
  --output-root "$EXPERIMENT_ROOT" --run-id <declared-run-id>
```

The reference recipe explicitly enables `prefetch_depth: 8` and `prefetch_workers: 4`.
These settings preserve the training protocol but are machine-sensitive: smaller-memory or
different-storage systems should use a derived recipe with a lower bounded queue and record
the effective values in `config.resolved.yaml`.

After training, `scripts/evaluate.py` must be called against `checkpoints/best.pt` with the
same recipe and data paths. It writes all three official views to
`evaluation/<scope>/metrics.json` and logs them under separate W&B namespaces.

## 8. Required ablations

### 8.1 RNA encoder contribution

Use the same DNA branch, loss, batching, split, optimizer budget, and output head while
changing only the RNA encoder. Required controls include:

- global linear bottleneck;
- capacity-matched nonlinear/MLP bottleneck;
- locus-conditioned learned-token attention;
- frozen BulkRNABERT representation plus a matched trainable adapter;
- at least one competitive generic tabular/set encoder justified from literature.

Foundation-model comparisons must match information access and fitting policy. “Frozen”
versus fine-tuned and pretraining-data overlap must be reported. Parameter count, trainable
parameter count, FLOPs, wall time, peak memory, and input gene coverage are required.

### 8.2 Mean proxy-task contribution

Compare, under the same encoder and parameter/compute budget:

- no mean branch;
- CpG branch without auxiliary mean supervision;
- CpG branch with leakage-safe mean supervision;
- optional explicit additive mean-logit baseline.

Report both correlation and calibration metrics. Existing results show that removing the
mean branch can affect unseen-locus MSE much more than MAS-PCC; reporting only correlation
would hide the main benefit.

### 8.3 Data-source and technology ablations

Reproduce the MethylProphet-style source study with explicit recipes:

- Array only;
- Array + EPIC;
- Array + WGBS;
- Array + EPIC + WGBS.

For each arm, record observation counts, sampling/aggregation weights, missingness, patient
overlap, protocol status, and all three Array evaluation views. This isolates the benefit of
technology mixing without treating auxiliary sources as held-out tests.

### 8.4 Genomic encoder and sequence-window ablations

- Compare NTv3 with alternative genomic encoders on matched TCGA chr1.
- Test multiple sequence-window sizes while keeping CpG identity, pooling, orientation,
  downstream architecture, and training budget fixed.
- Cache embeddings under a key containing model revision, hg38 build, window length,
  orientation, pooling, dtype, and sequence checksum.
- Include a sequence-free or simple sequence baseline so gains cannot be attributed solely
  to locus identity memorization.

### 8.5 Biological analyses

Biological explanations must be hypotheses tested by interventions, not post-hoc stories.
Useful tests include:

- shuffle RNA across patients while retaining CpG features;
- zero/occlude RNA and DNA branches separately;
- stratify by inter-patient CpG variance, genomic context, tumor type, assay, and chromosome;
- test stability of RNA-token attention across seeds;
- perform corrected pathway enrichment on stable gene loadings;
- verify high-attention programs by perturbation/occlusion.

Attention weights must not be presented as causal explanations on their own.

## 9. Storage cleanup policy

No data should be deleted merely because it is large or appears duplicated. Cleanup must be
manifest-driven and recoverable.

### 9.1 Retain or make reproducibly obtainable

- the frozen canonical bundle schema, manifests, protocols, and checksums;
- legally distributable protocol IDs and registries;
- scripts required to obtain/build canonical inputs where licensing allows;
- the NTv3 atlas or a deterministic documented way to regenerate it;
- CpG-statistics manifests and builders;
- the compact-cache builder and exact-equivalence tests;
- final checkpoints and evaluation artifacts required for paper tables;
- third-party model names and immutable revisions, not necessarily their local download
  caches.

### 9.2 Rebuildable/local artifacts that may be removed after verification

- `model_weights_cache/`, if every dependency can be downloaded by immutable revision;
- failed-run logs and partial checkpoints that provide no unique scientific evidence;
- obsolete queue runners after commands and provenance are preserved;
- `experiments/_legacy_pre_refactor/` and superseded experiment trees after required result
  rows/checkpoints are archived;
- old raw/extracted MethylProphet copies after the canonical bundle has been hash-validated
  and the original data remain lawfully obtainable;
- duplicate NTv3/feature caches proven identical by CpG-axis and content hashes;
- old `prior.npy`/`sigma.npy` caches after all supported single-stage paths and historical
  checkpoint reproductions are tested without them;
- temporary provenance inventories and local absolute-path manifests after their useful
  content has been consolidated into portable release manifests.

### 9.3 Mandatory pre-deletion procedure

1. Produce a size inventory and classify every candidate as canonical, derived, run output,
   third-party cache, or unknown.
2. Search code, configs, documentation, checkpoints, and run metadata for references.
3. Verify that the canonical source or deterministic regeneration path exists.
4. Compare semantic identity using IDs, shapes, dtypes, NaN masks, and hashes—not directory
   names or byte size alone.
5. Archive unique final metrics, configuration, checkpoint hashes, and W&B identifiers.
6. Move candidates to a dated quarantine/trash area first; run the full reproduction smoke
   test before irreversible deletion.
7. Record what was removed, why, and how it can be recovered.

## 10. Publication checklist

### Data and ethics

- [ ] Data card with cohort composition, tumor types, assay technologies, missingness,
  preprocessing, known biases, and intended/forbidden uses.
- [ ] TCGA/GDC and ENCODE access/licensing reviewed; no controlled or non-redistributable
  patient-level data included in the Git release.
- [ ] Patient identifiers de-identified and mapped consistently; repeated measurements
  documented.
- [ ] Split and auxiliary-statistic leakage tests passing.
- [ ] Dataset/protocol version and checksums recorded for every reported run.

### Reproducibility

- [ ] Clean-environment installation test.
- [ ] Pinned Python, PyTorch, CUDA, h5py, NumPy, and W&B-compatible versions.
- [ ] One-command small CPU smoke test and one-command representative GPU reproduction.
- [ ] Deterministic seeds and nondeterministic CUDA operations documented.
- [ ] Hardware, wall time, energy/compute budget, parameter count, and peak memory reported.
- [ ] All paper tables generated from machine-readable result files.

### Scientific reporting

- [ ] Primary endpoint and model-selection rule declared before locked-test evaluation.
- [ ] Multi-seed confidence intervals and appropriate paired/bootstrap tests.
- [ ] Parameter/compute-matched controls for every architectural claim.
- [ ] chr2+3 or a genuinely fresh scope reported separately from chr1-informed development.
- [ ] Strong published baselines reproduced under the same split, or compared with explicit
  provenance limitations.
- [ ] Negative results and interrupted/invalid runs distinguished from completed evidence.
- [ ] Limitations cover cohort shift, ancestry/tissue representation, assay shift,
  reference-genome dependence, and non-causal interpretation.

### Repository release

- [ ] License, `CITATION.cff`, contribution policy, security/contact information.
- [ ] README quick start and exact paper reproduction guide.
- [ ] Model card and checkpoint license.
- [ ] Portable configuration: no hard-coded `/raid`, `/dune`, usernames, or local absolute
  paths in release-facing defaults/manifests.
- [ ] Secrets, W&B credentials, raw patient data, caches, logs, and large checkpoints excluded
  from Git history, not only from the current working tree.
- [ ] Broken links, spelling, stale architecture descriptions, and retired CLI examples
  removed by documentation tests.

## 11. Source-of-truth documents

This specification summarizes policy. Detailed live contracts remain in:

- `docs/DATA.md`: canonical bundle schema and loader invariants;
- `docs/data/METHYLPROPHET_PROTOCOLS.md`: protocol provenance and source-revision caveats;
- chr123 compact-cache and performance measurements retained in repository history and
  relevant provenance records;
- `docs/RNA_METHYLATION.md`: current architecture selection and open questions;
- `docs/CPG_STATISTICS.md`: mean/sigma target construction;
- `docs/BENCHMARK_METHYLPROPHET.md`: benchmark comparability.

If these documents disagree, code plus the resolved configuration and immutable data/run
manifests determine what actually happened; the documentation discrepancy must then be fixed
before publication.
