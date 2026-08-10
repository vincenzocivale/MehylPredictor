# Evaluation

> **Legacy workflow notice.** This document describes evaluation for the older
> `scripts/train.sh` path. The current exact MethylProphet-compatible Array-chr1
> benchmark uses `scripts/evaluate_current_model_vs_methylprophet.py`, which
> evaluates the three official views and optionally scans released
> MethylProphet predictions when they are available. See
> [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md) and
> [`data/METHYLPROPHET_PROTOCOLS.md`](data/METHYLPROPHET_PROTOCOLS.md).

`scripts/train.sh` / `configs/train.yaml` deliberately do **not** evaluate on
test data — `configs/train.yaml`'s `evaluation.panels` is empty on purpose,
so a training run can never accidentally touch held-out data. Evaluation is
a separate, explicit step, run after a checkpoint exists.

## `scripts/evaluate_official_test.py`

Evaluates a given checkpoint against a given config's data on one panel
(default: `test x test`, the official double-OOD test set — see
[`data.md`](data.md#splits)). Reuses `ExperimentRunner.predict_panel` — the
same GPU-chunked forward pass `train()` uses internally for any configured
panel — rather than re-implementing inference; the only new logic is loading
a standalone checkpoint into a fresh runner outside of `train()`.

```bash
python scripts/evaluate_official_test.py \
  --config artifacts/train/seed17/final_config.yaml \
  --checkpoint artifacts/train/seed17/final_refit/best.pt \
  --output artifacts/train/seed17/final_refit/test_metrics.json \
  [--sample-split test] [--cpg-split test] \
  [--save-predictions artifacts/train/seed17/final_refit/test_predictions.npz]
```

- `--config` supplies the `data.*` paths (must match the checkpoint's
  training data layout) — use the **final-refit** config, since only its
  `data.sample_metadata`/`data.cpg_splits` point at the original, full
  manifests that actually contain a `test` label (the development config's
  `_dev` split files only have `train`/`dev_heldout`).
- `--checkpoint` must match the config's architecture exactly (the script
  loads `state_dict` with `strict=True`); it does not do the partial/remapped
  warm-start loading `trainer.py::_apply_warm_start` supports for
  architecture changes.
- `--save-predictions` writes a compressed `.npz` with per-cell
  `target`/`prediction`, `prior`, `sample_idx`, `cpg_idx`, `cancer_type` —
  omit it to only compute aggregate metrics (much less memory/disk).

Other useful panels, by passing `--sample-split`/`--cpg-split`:
`sample_ood` (`test`/`train`), `locus_ood` (`train`/`test`),
`in_distribution` (`train`/`train`).

## Metrics (`metrics.py::evaluate_predictions`)

Every panel reports, at minimum:
- `mse` / `prior_mse` and `skill_vs_prior = 1 - mse/prior_mse` — the
  headline number: how much the model improves on the frozen per-CpG prior.
- `dynamic_pearson`/`dynamic_spearman` — per-cell correlation across the
  full panel.
- `within_cancer_pearson`/`within_cancer_spearman`/`within_cancer_skill` —
  same, computed within each cancer type and pooled (so a model can't win
  purely by separating cancer types that already have very different
  baseline methylation levels).
- `sample_win_fraction` — fraction of samples where the model beats the
  prior.
- `per_cancer_type` and `per_variability_tertile` breakdowns (tertiles
  computed on train CpGs only, from the same variability proxy the gate
  uses, then applied genome-wide).

See [`README.md`](../README.md#performance) for the current model's numbers
on the official test panel, and
`artifacts/train/seed17/final_refit/test_metrics.json` for the full
breakdown.
