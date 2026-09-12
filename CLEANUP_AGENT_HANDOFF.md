MethylPredictor — Final cleanup / Data-Results V2 agent handoff

Work on repository:

vincenzocivale/MehylPredictor
branch: refactor/repo-v2-2026-09

Primary objective

Finish the repository cleanup/refactor and leave the project ready for final
multi-machine paper experiments.

This is an end-to-end engineering task. Make conservative changes, preserve
scientific behavior, run the full test suite, and commit/push coherent changes.

Do NOT delete, move, rename, truncate, or otherwise mutate external data under
METHYL_DATA_ROOT.

Before changing anything:

git status --short
git branch --show-current
git log -5 --oneline

Preserve useful uncommitted/local work. Do not blindly overwrite local changes.
Final scientific model contract — DO NOT CHANGE

Final model:

class: EfficientSingleAttentionPredictor
variant: efficient_single_attn_8ffn_residual_functional8_head2

Architecture:

functional branch:
  4165 sparse regulatory tracks via EmbeddingBag
  23 dense locus annotations
  width 256
  8 residual FFN blocks

RNA branch:
  bulk RNA
  64 latent program tokens
  token dim 256

retrieval:
  one locus-conditioned cross-attention
  4 heads
  deep_query = false
  8 residual FFN blocks after retrieval

prediction head:
  concat deep functional + RNA-conditioned state
  512 -> 256
  2 residual FFN blocks 256 -> 1024 -> 256
  256 -> 128 -> 1
  sigmoid

mean proxy:
  training-only
  reads deep functional state
  MUST NOT feed final beta prediction

Loss:

1.0 * beta MSE
0.15 * locus PCC
0.15 * mean proxy

Protocol:

epochs: 80
lr: 2e-4
weight_decay: 1e-4
scheduler: cosine warmup
warmup: 1 epoch
min_lr_ratio: 0.1
gradient_clip: 1.0
AMP: bfloat16
seeds: 17, 29, 43 for final study arms

Canonical recipe:

configs/models/main.yaml

No NTv3/genomic-FM embedding is an input to the final RNA model.

Do not redesign architecture, losses, splits, seeds, metrics, or paper protocol.
Cleanup already completed

Treat these as completed unless the current Git state proves otherwise:

5a  experiment surface freeze
5b  config audit
5c  repo/documentation cleanup
5d  standardized paper experiment API
5e  safe external cleanup inventory

6a  final architecture selection
6b  final architecture promotion
6c  J-series recipe/launcher removal
6d1 retired modeling architecture prune
6d2 runtime/config compatibility prune

The active public runtime should now center on:

RNAMethylationTrainer
evaluate_rna_checkpoint
EfficientSingleAttentionPredictor

Do NOT revive deleted J-series machinery.

Do NOT touch:

results/reference/**

except to read it when necessary for provenance.
Data / Results V2 target

The repo is moving to this durable storage contract:

METHYL_DATA_ROOT/
├── datasets/
│   └── methylprophet_repro_v1
├── reference/
│   └── hg38
├── derived/
│   ├── methylprophet_table5_tcga_chr1
│   ├── cpg_statistics
│   ├── ntv3_functional_peak_atlas_chr1_all_sources
│   ├── ntv3_probe_targets
│   ├── bulkformer_embeddings
│   └── bulkrnabert_embeddings
└── experiments/
    └── runs/
        └── rna_methylation/

Repository-side compact results target:

results/paper/
├── registry.json
├── runs.jsonl
├── runs.csv
├── main/
├── functional_baselines/
├── mean_proxy/
├── rna_encoder/
└── cpg_prior/

Heavy artifacts stay under METHYL_DATA_ROOT.

Curated result files must use portable artifact references:

methyl-data://experiments/runs/rna_methylation/...

Never commit machine-specific paths like:

/dune/...
/data2/...
/raid/...

into curated paper result JSON.
Current Data/Results V2 files

Inspect and finish/integrate these if present:

configs/data/storage_v2.yaml
configs/data/paper_dependencies_v2.yaml
scripts/inventory_paper_dependencies.py
docs/DATA_RESULTS_V2.md
docs/PAPER_DEPENDENCY_INVENTORY.md
tests/test_data_results_v2_contract.py
tests/test_paper_dependency_inventory.py

Some of these may currently be local/uncommitted. Preserve correct local work.
Frozen paper study surface

Centralize support for exactly these paper studies.
Main

main
seeds: 17, 29, 43
recipe: configs/models/main.yaml

Functional baselines

global_rna_shift
mlp_rna_cpg
bilinear_rna_cpg
seeds: 17, 29, 43

Mean-proxy contribution

full
no_supervision
no_branch
seeds: 17, 29, 43

RNA encoder comparison

ours_program_tokens
bottleneck_mlp
gene_pathway
bulkformer_147m
bulkrnabert
seeds: 17, 29, 43

CpG prior

cpg_prior
zero-parameter baseline

Do NOT silently remove BulkRNABert from the paper study.
Current real external dependency inventory

Observed under:

/dune/DATASETS/MethylPredictionData

Required runtime:

datasets/methylprophet_repro_v1                         ~39.48 GB
derived/methylprophet_table5_tcga_chr1                 ~12.29 GB
derived/ntv3_functional_peak_atlas_chr1_all_sources     ~4.63 GB
derived/ntv3_probe_targets                               ~0.84 GB
derived/cpg_statistics                                   ~0.13 GB
derived/bulkformer_embeddings                            ~0.01 GB
TOTAL                                                    ~57.38 GB

Required reproducibility:

reference/hg38                                          ~3.05 GB

Known unreferenced cleanup targets:

derived/ntv3_expansion                                  ~49.01 GB
derived/rna_feature_cache                               ~16.92 GB
derived/ntv3_pretrain_expansion                         ~11.55 GB
derived/methylprophet_table5_tcga_chr1_pretrain_ablation ~8.43 GB
derived/tcga_canonical                                   ~5.87 GB
derived/methylprophet_compact_chr123                     ~5.08 GB
derived/ntv3_post_track_manifest                         ~2.46 GB
derived/genomic_prior_v2                                 ~1.27 GB
derived/hyenadna_expansion                               ~0.97 GB
derived/ntv3_functional_peak_atlas_chr1                 ~0.52 GB
derived/rna_feature_cache_ntv3_pre                       ~0.12 GB
TOTAL                                                   ~102.20 GB

Review separately:

model_weights_cache/ntv3_650m_post_hf_cache              ~8.37 GB

Missing required runtime dependency:

derived/bulkrnabert_embeddings/tcga

The cleanup planner MUST remain blocked from destructive authorization while
this required dependency is missing.
Work to finish
1. Complete final run-storage migration

New final RNA runs must use:

experiments/runs/rna_methylation/<scope>/<run-id>

Existing historical:

experiments/runs/locus_cls_joint/**

must NOT be moved or rewritten.

Remove live/current use of locus_cls_joint where it exists only as storage
compatibility.

Update all active code/config/tests/docs consistently.
2. Implement portable artifact URI resolver

Add a small tested utility supporting:

Path under METHYL_DATA_ROOT
    -> methyl-data://relative/path

methyl-data://relative/path
    -> local machine path under METHYL_DATA_ROOT

Requirements:

reject paths outside data root
reject malformed scheme
reject path traversal
normalize safely
no dependency on machine-specific mount point

Use the URI form in curated paper records.
3. Centralize the paper job matrix

Create a single source of truth, preferably:

configs/paper_studies.yaml

Each job definition must resolve:

study
arm
recipe
seed
scope
RNA cache/dependency selection
required dependency names
run_id

Do not maintain duplicate hardcoded study definitions in several runner scripts.
4. Implement deterministic multi-machine execution

Create a runner such as:

python scripts/run_paper_jobs.py --shard 0/3 --gpu 0
python scripts/run_paper_jobs.py --shard 1/3 --gpu 0
python scripts/run_paper_jobs.py --shard 2/3 --gpu 0

Requirements:

deterministic stable job ordering
deterministic sharding
unique run id derived from study/arm/seed
explicit GPU selection
study/arm/seed filtering
dependency validation before launch
skip complete jobs by paper/record.json
never overwrite complete run by default
clear BLOCKED status for missing BulkRNABert
dry-run/list mode

No DB and no Slurm dependency.
5. Optional local scratch abstraction

Support:

METHYL_DATA_ROOT = shared durable storage
METHYL_WORK_ROOT = optional local SSD scratch

Shared-root execution should remain the default.

If local scratch is enabled:

train locally
publish only completed run
publish into temporary destination
finalize atomically where possible
paper/record.json must never appear for half-published runs

Do not overcomplicate this if it threatens robustness.
6. Curated result record schema

Each final per-run JSON should contain at least:

schema_version
kind
study
arm
seed
scope
run_id

git commit
git dirty state

recipe path
recipe sha256

data profile path/hash
dependency contract path/hash

best epoch
all official evaluation views
headline metrics

artifact URIs:
  run
  checkpoint
  resolved config
  training summary
  evaluation

Artifact values in curated repo results must be methyl-data://....

No checkpoint binary or other heavy data belongs in Git.

Update collector(s) to generate:

results/paper/runs.jsonl
results/paper/runs.csv
results/paper/registry.json

plus per-run JSON files.
7. Historical run slim-archive tooling

There are roughly 95 historical run directories.

Build a DRY-RUN-FIRST archiver/planner that extracts compact provenance from:

metadata.json
config.resolved.yaml
training/summary.json
evaluation/*/metrics.json
checkpoint metadata/hash when inexpensive

Target compact archive:

experiments/archive/historical_run_metadata/

The tool should report:

ARCHIVEABLE
INCOMPLETE
MISSING_METADATA
estimated reclaimable bytes
original relative path

Do NOT delete historical run directories automatically.
8. Deep cleanup planner

Build a final planner covering:

derived/*
experiments/runs/*
experiments/searches/*
model_weights_cache/*
reference/*
malformed experiments/runs/runs/**

Use explicit classes:

KEEP
BLOCKED_REQUIRED
ARCHIVE_THEN_DELETE
DELETE_AFTER_SMOKE
REVIEW
NEVER_TOUCH

It must be dry-run by default.

Normal inventory/planner code must contain no destructive calls.

Do not authorize deletion while bulkrnabert_rna is missing.

Quantify reclaimable GB.
9. Simplify CpG-prior dependency

Current RNA evaluation needs only:

features/cpg_idx.npy
features/prior.npy

Current CpGPriorEvaluator still uses LocusFeatureCache, therefore currently
also requires:

embeddings.f16.npy
sigma.npy

Refactor it to LocusPriorCache if metrics/predictions are behaviorally
identical.

Add regression tests.

Then update dependency manifests so the extra embedding/sigma files are no
longer required by the final paper surface.

Do not physically delete them in this task.
10. Remove stale active-repo references

Audit live source/config/tests/docs for stale current references to:

J0/J1/J-series
locus_cls_joint as current run namespace
NTv3 embeddings as final RNA input
deleted architecture classes
deleted functional_fusion recipe paths
old architecture-search scripts
old compatibility aliases

Historical provenance and results/reference/** stay untouched.
Safety constraints

Non-negotiable:

DO NOT mutate METHYL_DATA_ROOT
DO NOT run rm -rf on external data
DO NOT touch results/reference/**
DO NOT revive J-series architecture machinery
DO NOT change final scientific architecture
DO NOT change train/val splits
DO NOT change official metrics
DO NOT change seeds
DO NOT silently remove BulkRNABert
DO NOT rewrite historical runs
DO NOT commit machine-specific absolute data paths into curated results

Required validation

Run focused tests as needed, then:

python -m compileall -q src scripts
pytest -q

The full suite must be green.

Also test explicitly:

final model contract
paper experiment API
Data/Results V2 contract
dependency inventory
methyl-data URI resolver
paper study registry
job sharding determinism
collector/result schema
historical archive planner
deep cleanup planner
CpG-prior equivalence

Git completion

When tests are green:

git status --short
git diff --stat

Review for accidental historical/provenance changes.

Then commit coherent changes and push to:

refactor/repo-v2-2026-09

Do not force-push.
Final report required from the agent

At completion, report:

1. files added / modified / deleted
2. architectural/runtime changes made
3. tests executed and results
4. remaining blockers
5. BulkRNABert status
6. exact final storage contract
7. historical run cleanup summary
8. reclaimable external storage by category
9. exact commands to launch all final jobs across multiple machines
10. any manual action still required before physical data deletion

Do not perform the physical external-data deletion as part of this task.
