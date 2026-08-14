# Training technique

> **Legacy workflow notice.** This document describes the older
> `scripts/train.sh` / `configs/train.yaml` 21,792-gene training path. It is
> retained for checkpoint reproducibility. The current MethylProphet-compatible
> Array-chr1 benchmark is orchestrated by
> `scripts/run_overnight_current_model_vs_mp.sh`; mixed-source E2/E3/E4 runs use
> `scripts/run_full_e2_e4.sh`. See [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md)
> and [`scripts/README.md`](../scripts/README.md).

Two-stage protocol, run end-to-end by `scripts/train.sh`. Canonical config:
`configs/train.yaml`.

## Stage 1 — development training

Fits on a nested split strictly inside the official `train` label: 90% stays
`train` (the fit pool), 10% is relabeled `dev_heldout` — seed 17, stratified
by chromosome for CpGs / by cancer type for samples. Built once by
`scripts/build_dev_split.py` (writes
`rna_branch_inputs_dev_seed17/{cpg_split_manifest_dev,sample_metadata_dev}.parquet`
next to the main data files; `scripts/train.sh` skips rebuilding it if that
directory already has a `manifest.json`). Official `validation`/`test` are
never referenced at this stage.

Early stopping: `patience=10`, `min_delta=1e-5`, `checkpoint_metric=mse`,
`min_epochs=10`, `epochs=50` cap. `dev_heldout` is used for both the sample
and CpG axes of the validation panel (`validation_max_cpgs=1024`). The best
epoch by validation MSE is recorded as `best_epoch`.

## Stage 2 — final refit

`scripts/render_final_refit_config.py` derives a config from the development
one plus its measured `best_epoch`:

- same architecture, hyperparameters, optimizer, schedule, and seed;
- `data.sample_metadata`/`data.cpg_splits` point back at the **original,
  unmodified** manifests (100% of official `train_cpg`/`train_sample`, not
  the 90% dev-fit pool);
- `training.epochs`/`min_epochs` fixed to `best_epoch`,
  `checkpoint_selection: final` (every validation epoch overwrites `best.pt`
  unconditionally — early stopping is structurally unreachable);
- `validation_sample_split`/`validation_cpg_split` point at `train` itself,
  as an in-sample-only sanity curve — never official `validation`/`test`.

The rendered config is written to the run's `artifacts/` directory (not
tracked under `configs/`), e.g. `artifacts/train/seed17/final_config.yaml`.

## Fixed training choices (`configs/train.yaml`)

- **Sampling** — `training.cpg_sampling: full_coverage`
  (`full_coverage_sampler.py`): a deterministic per-epoch schedule that
  guarantees every CpG and every sample in the training pool is visited at
  least once per epoch, instead of independent random per-step draws.
  `steps_per_epoch` is derived from the pool sizes and
  `cpg_batch_size`/`sample_batch_size`, not configured directly.
- **Loss** (`losses.py`) — `beta_mse_weight=1.0` (MSE on the predicted beta)
  + `residual_huber_weight=0.1` (Huber on the raw pre-gate residual,
  `delta=1.0`) + `shrinkage_weight=1e-4` (L2 shrinkage toward the prior).
- **Optimizer** — fused AdamW, `lr=2e-5`, `weight_decay=1e-4`, gradient
  clipping at norm 1.0.
- **Precision** — bfloat16 autocast (`amp_dtype: bfloat16`), TF32 matmuls
  allowed, `matmul_precision: high`.
- **Batching** — `sample_batch_size=64`, `cpg_batch_size=2048`.
- **Tracking** — Weights & Biases (`project: MethylationPredictor`); scalar
  metrics only, checkpoints are **not** logged inline as W&B artifacts
  (`log_checkpoint: false` — an earlier run with it enabled stalled
  deterministically on the inline upload; upload checkpoints as a separate
  post-hoc step via `cli.py upload-artifact` if needed).

## Preflight

`scripts/train.sh` runs `scripts/preflight_genomewide_fullcoverage.py`
before every training phase it hasn't already completed:

- git state (HEAD, dirty status);
- the test suite (`pytest -q`), skippable with `--skip-tests`;
- split integrity: no duplicate `cpg_idx`/`sample_idx`, every ID maps to
  exactly one split label; optionally cross-checked against the raw
  MethylProphet official split source files
  (`--skip-official-split-check` to skip this cross-check when that source
  tree isn't present on the current machine — the internal integrity checks
  still run either way);
- data completeness: embedding/beta/RNA matrix shapes, no NaNs in
  `locus_features.parquet`;
- the nested dev split is present;
- disk/RAM/GPU headroom;
- a 1-epoch, 3-step smoke run of the real training path (uniform sampling,
  not full-coverage — kept fast), skippable with `--skip-smoke-run`.

Writes a manifest with a `sha256` of every input file checked and an overall
`ok: true/false`; exits non-zero on any hard failure.

## What `scripts/train.sh` does *not* do

Preflight → dev split → development training → final refit, and nothing
else — deliberately no test evaluation, no baseline comparison, no
bootstraps, no ablations, no report generation. Test evaluation is a
separate, explicit step — see [`evaluation.md`](evaluation.md).

## Idempotency

Each phase writes a `.done` marker file under
`$RUN_DIR/{manifests,development,final_refit}/`; re-running
`scripts/train.sh` skips any phase whose marker already exists. Override via
env vars: `RUN_DIR` (default `artifacts/train/seed17`), `DEV_CONFIG`
(default `configs/train.yaml`), `FINAL_CONFIG`, `DEV_SPLIT_DIR`.
