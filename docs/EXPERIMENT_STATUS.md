# Experiment status

This file is the compact source of truth for the **current** research workflow.
It records protocol status and stable result summaries, not transient PIDs,
GPU assignments or machine-specific progress.

Last updated: 2026-08-10.

## E0 — exact current-model Array-chr1 benchmark

**Status:** COMPLETE for OURS (seed 17).

Protocol:

- official MethylProphet Array-chr1 split;
- 8,260 train / 918 validation patients;
- 33,885 train / 6,742 held-out CpGs;
- all 25,017 canonical RNA genes;
- nested-development model selection, then final refit on all official train IDs;
- final checkpoint epoch: 45.

Local artifact root (default):

```text
artifacts/methylprophet_comparison/current_model_tcga_array_chr1_seed17/
```

OURS three-view result:

| view | MSE | MAE | MAS-PCC | MAC-PCC | prior MSE | skill vs prior |
|---|---:|---:|---:|---:|---:|---:|
| train-CpG x val-sample | 0.022541 | 0.096200 | 0.530027 | 0.918942 | 0.031190 | 0.277282 |
| val-CpG x train-sample | 0.020289 | 0.086770 | 0.513238 | 0.928712 | 0.026538 | 0.235500 |
| val-CpG x val-sample | 0.020968 | 0.088142 | 0.493754 | 0.926788 | 0.026998 | 0.223360 |

The current `headline.json` is OURS-only when `mp_label` is null. In that case
`cpg_win_fraction` and `sample_win_fraction` are wins against the frozen prior,
not against MethylProphet.

### MethylProphet paired comparison

**Status:** BLOCKED_BY_GATED_ACCESS when the released prediction dataset or
official checkpoint is unavailable to the active Hugging Face account.

The E0 training must **not** be repeated when access is granted. Set
`MP_EVAL_DIR` and rerun the benchmark launcher/evaluator; completed development
and final-refit stages are skipped.

## Shared NTv3 genomic feature expansion

**Status:** IN PROGRESS / reusable preprocessing resource.

The full-suite feature store is separate from experiment outputs. The target
representation is fixed to:

- `InstaDeepAI/NTv3_650M_post`;
- hg38;
- 32,768-bp forward window;
- central C/G mean;
- BF16 inference;
- FP16 storage for the current production extraction.

See [`data/GENOMIC_FEATURE_STORE.md`](data/GENOMIC_FEATURE_STORE.md).

## E2 — Array + EPIC + WGBS chr1

**Status:** NOT YET RECORDED AS COMPLETE.

Primary protocol: `tcga_mix_chr1`. The exact Array evaluation split remains the
same as E0; auxiliary-source holdout semantics are explicit (`mp_matched` vs
`strict_global`).

## E3 — Array + EPIC + WGBS chr1-3

**Status:** WAITING FOR COMPLETE NTv3 FEATURE EXPANSION.

Primary protocol: `tcga_mix_chr123`.

## E4 — Array genome-wide

**Status:** SUPPORTED BY CODE; no canonical result recorded here yet.

Protocol: 326,906 train vs 81,493 held-out Array CpGs genome-wide.

## Rule for updates

Update this file only when an experiment reaches a stable state. Detailed logs,
checkpoints and generated predictions remain outside git. Never copy transient
machine state into this document.
