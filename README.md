# MethylPredictor — RNA-to-DNAm training

Trains a single canonical model, `RNA2DNAmModel`, that predicts per-CpG DNA
methylation (beta value) for a TCGA patient from that patient's bulk RNA-seq
expression profile, as a residual correction on top of a frozen, externally
computed per-CpG prior. The historical research code (architecture search,
baselines, ablations, diagnostics, bootstrap analyses, paper reports) has
been removed; git history is the archive if any of it is needed again.

Documentation:
- [`docs/architecture.md`](docs/architecture.md) — the model, its
  components, and the frozen prior it corrects.
- [`docs/training.md`](docs/training.md) — the two-stage training protocol,
  loss/optimizer, sampling strategy, and preflight checks.
- [`docs/data.md`](docs/data.md) — input files, splits, and how the nested
  development split is built.
- [`docs/evaluation.md`](docs/evaluation.md) — how to evaluate a checkpoint
  on held-out data and what the metrics mean.

## Performance

Current final-refit checkpoint (`artifacts/train/seed17/final_refit/best.pt`,
epoch 45), evaluated on the **official test set** — 414 test samples ×
40,689 test CpGs, both axes held out from all training/dev data (see
[`docs/data.md`](docs/data.md#splits)):

| metric | model | frozen prior | skill |
|---|---|---|---|
| MSE | 0.02241 | 0.02823 | **+20.6%** |
| Pearson (dynamic, per-cell) | 0.545 | — | — |
| Pearson (within cancer type) | 0.456 | — | +20.6% skill |
| fraction of samples beating the prior | 98.1% | — | — |

Full breakdown (per cancer type, per CpG-variability tertile):
`artifacts/train/seed17/final_refit/test_metrics.json`. See
[`docs/evaluation.md`](docs/evaluation.md) for what each metric means and
[`docs/data.md`](docs/data.md#splits) for exactly which data this was
trained on vs. tested on.

## Running

```bash
python -m pip install -r requirements.txt
python -m pip install -e .

bash scripts/train.sh          # preflight -> dev split -> dev training -> final refit

python -m methylation_predictor train --config configs/train.yaml     # single stage, direct CLI
python -m methylation_predictor validate --config configs/train.yaml  # summarize aligned inputs, no training

python scripts/evaluate_official_test.py \
  --config artifacts/train/seed17/final_config.yaml \
  --checkpoint artifacts/train/seed17/final_refit/best.pt \
  --output artifacts/train/seed17/final_refit/test_metrics.json
```

Generated checkpoints and runtime artifacts belong under `artifacts/`, which
must remain untracked.
