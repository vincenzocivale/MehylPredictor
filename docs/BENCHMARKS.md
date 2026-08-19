# Benchmarks

Machine-readable numbers live in `results/reference/`; this document is the
narrative index into them.

## Models and scopes

Two trainable models, three genomic scopes:

- `CpGStatisticsPredictor` → `results/reference/cpg_statistics/{chr1,chr123,genomewide}.yaml`
- `RNAMethylationPredictor` → `results/reference/rna_methylation/{chr1,chr123,genomewide}.yaml`

`chr1` and `chr123` (`chr1 ∪ chr2 ∪ chr3`) are the MethylProphet-matched
comparison scopes (see [`BENCHMARK_METHYLPROPHET.md`](BENCHMARK_METHYLPROPHET.md));
`genomewide` is the primary general benchmark.

## Current reference results

### `rna_methylation`

| scope | train-CpG × val-sample | val-CpG × train-sample | val-CpG × val-sample |
|---|---:|---:|---:|
| chr1 | 0.6327 / 0.01278 | 0.5984 / 0.01874 | 0.5613 / 0.01941 |
| chr123 | *pending — see `results/reference/rna_methylation/chr123.yaml`* | | |
| genomewide | 0.5889 / 0.01482 | 0.5800 / 0.02099 | 0.5244 / 0.02223 |

Cells are `MAS-PCC / MSE`. Genome-wide per-chromosome mean MAS-PCC: 0.5196
(observed range 0.459–0.579). LR 5e-5, constant scheduler, 80 epochs, seed 17
for both frozen chr1 and genome-wide runs.

### `cpg_statistics`

Both chr1 and genomewide references below are the validated mean-predictor
component only (pre joint mu/sigma refactor); retrain the joint model before
promoting a new reference. chr123 has no frozen reference yet.

| scope | heldout beta MSE | heldout PCC | heldout R² |
|---|---:|---:|---:|
| chr1 | 0.00768 | 0.9671 | 0.9348 |
| genomewide | 0.00790 | 0.9671 | 0.9347 |

## MethylProphet chr1 benchmark: how the reference architecture was reached

Historical progression on the exact MethylProphet Table-5 chr1 benchmark
(Array + EPIC + WGBS), frozen seed 17, `MAS-PCC / MSE` per view:

| run | train-CpG × val-sample | val-CpG × train-sample | val-CpG × val-sample |
|---|---:|---:|---:|
| V0, 4 epochs | 0.5055 / 0.0251 | 0.4705 / 0.0217 | 0.4695 / 0.0217 |
| V0, 25 epochs | 0.5505 / 0.0235 | 0.5295 / 0.0204 | 0.5169 / 0.0207 |
| V3 (prior fix) | 0.5609 / 0.0150 | 0.5276 / 0.0204 | 0.5135 / 0.0207 |
| V2 (+ locus PCC) | 0.5773 / 0.0147 | 0.5647 / 0.0199 | 0.5342 / 0.0204 |
| V1 (+ variance normalization) | 0.5811 / 0.0144 | 0.5708 / 0.0197 | 0.5401 / 0.0201 |
| MethylProphet paper | 0.5455 / 0.0199 | 0.4194 / 0.0266 | 0.3904 / 0.0271 |

V1 (variance-normalized residual) was selected as the canonical
`RNAMethylationPredictor` architecture. The current unified-scopes pipeline's
frozen chr1 reference (table above, 0.6327/0.5984/0.5613) supersedes this
table numerically — it was re-run end-to-end under the current pipeline — but
the table is kept as the record of *why* the variance-normalized architecture
was chosen over the V0/V2/V3 predecessors.

**Known protocol caveat:** the canonical bundle reproduces an 8,260/918 Array
train/validation split; the MethylProphet paper reports 8,258/920 after
excluding Array/WGBS patient overlap that this repo's bundle carries no
crosswalk for. See [`BENCHMARK_METHYLPROPHET.md`](BENCHMARK_METHYLPROPHET.md)
for the full explanation and the exact finite-pair counts this affects.

## Ablation summary

`results/reference/ablations.yaml` holds the machine-readable outcomes of
three completed ablations against the chr1 `rna_methylation` reference
(val-CpG × val-sample MAS-PCC 0.5613 baseline):

- **`prior_headroom`** — an oracle mu (true per-locus mean instead of the
  predicted prior) reaches MAS-PCC 0.5675, a delta of only 0.0062. Conclusion:
  the static mean prior is not the main MAS-PCC bottleneck.
- **`prior_replacement`** — a marginally-better standalone mean predictor
  (MAS-PCC 0.5598 vs the canonical prior's 0.5613 on its own held-out metric)
  did not improve downstream reconstruction when substituted in. Conclusion:
  better standalone mean prediction alone does not transfer to better joint
  reconstruction.
- **`training_search`** — LR/scheduler search on chr123 (see
  [`WORKFLOWS.md`](WORKFLOWS.md#hyperparameter-search) for the search
  mechanics): LR 5e-5
  constant, 80 epochs was selected (inner double-OOD MAS-PCC 0.5712 at epoch
  80, vs 0.5704 at epoch 57 with the same LR and 0.5639 at epoch 39 with
  LR 4e-5).

Three further chr1 development ablations were designed but not retained as
runnable configs after this repo's consolidation to a minimal pipeline
(`large_sample_pcc`: wider Array Cartesian blocks for more patients per
per-CpG Pearson estimate; `tail_aware_pcc`: split the locus-Pearson weight
between the mean and the lower-60%-correlation tail; `array_only_structured`:
restrict Pearson-family structured objectives to Array only, since WGBS's 32
measurements are too few for a stable patient-dynamic gradient). Their exact
configurations remain in git history (`configs/tcga_chr1/experiments/` before
the unified-scopes cleanup) if any needs to be reproduced.

## Reproduce

- [`WORKFLOWS.md`](WORKFLOWS.md) — the four generic entrypoints
  (`prepare.py`/`train.py`/`tune.py`/`evaluate.py`) across any model × scope.
- [`BENCHMARK_METHYLPROPHET.md`](BENCHMARK_METHYLPROPHET.md) — the frozen,
  exact MethylProphet chr1 reproduction path.
