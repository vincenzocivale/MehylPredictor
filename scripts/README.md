# Script map

The repository contains two current research workflows plus one retained legacy
workflow. Use this file before launching a long job.

## Current entrypoints

### `run_overnight_current_model_vs_mp.sh`

**Role:** exact Array-chr1 current-model benchmark (E0).

Orchestrates:

1. canonical protocol/feature preflight;
2. nested-development training;
3. final refit on all official train IDs;
4. exact three-view evaluation;
5. optional released-MethylProphet paired scan when `MP_EVAL_DIR` is available.

The MethylProphet download is optional. Auto-download is disabled by default to
avoid treating gated-access failures as part of the scientific run.

### `run_full_e2_e4.sh`

**Role:** shared feature preparation plus E2/E3/E4 canonical suite.

It owns reusable NTv3 expansion, feature-extension inference, compact caches,
and the mixed/genome-wide training runs. Large reusable genomic features are
stored outside experiment directories.

To prepare features only, without starting E2/E3/E4:

```bash
RUN_E2=0 RUN_E3=0 RUN_E4=0 \
  HG38_FASTA=/path/to/hg38.fa \
  bash scripts/run_full_e2_e4.sh
```

### `full_suite.py`

**Role:** low-level CLI used by `run_full_e2_e4.sh`.

Prefer the shell launcher for full workflows; call individual subcommands only
for controlled resume/debug/feature-worker operations.

## Current helpers

- `prepare_current_model_mp_benchmark.py` — builds the canonical E0 adapter and
  renders its final-refit config.
- `evaluate_current_model_vs_methylprophet.py` — streaming exact three-view
  evaluator; optionally reads released MP prediction parquet files.
- `smoke_tcga_mix_chr1.py` — canonical data/protocol smoke test.

## Legacy workflow

The following files belong to the older 21,792-gene training path and are kept
for reproducibility of historical checkpoints:

- `train.sh`
- `build_dev_split.py`
- `render_final_refit_config.py`
- `preflight_genomewide_fullcoverage.py`
- `evaluate_official_test.py`

Their associated documentation is `docs/data.md`, `docs/training.md` and
`docs/evaluation.md`. Do not combine their manifests/splits with the canonical
25,017-gene E0/E2/E3/E4 workflow.

## Operational rules

- Canonical source data is read-only.
- Generated checkpoints/logs/caches/features stay outside git.
- Do not launch `run_full_e2_e4.sh` independently on multiple machines against
  the same NTv3 store unless all extraction workers share one global
  `world_size` with disjoint ranks.
- A completed `.done` marker is a resumability contract; do not delete it
  casually to force recomputation.
- Prefer fail-closed behavior when required IDs/features are missing.
