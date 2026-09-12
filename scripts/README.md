# Script map

The public workflow has four entrypoints:

- `prepare.py` — build data for either model (`--model cpg_statistics`:
  multi-technology `mu`/`sigma` labels; `--model rna_methylation`: the
  required prepared caches);
- `train.py` — train either `cpg_statistics` or `rna_methylation`;
- `tune.py` — leakage-safe LR/scheduler/epoch search on inner development data;
- `evaluate.py` — evaluate either model on `chr1`, `chr123` or `genomewide`.

RNA-methylation training/evaluation/tuning use one functional-locus runtime.
For the exact MethylProphet-matched chr1 protocol, pass the prepared compact
cache through `--prepared-root`; functional atlas and annotation caches are
mandatory. See [`../docs/MODEL.md`](../docs/MODEL.md).

## Generated data

Training runs are written under `runs/<model>/<scope>/...`; hyperparameter
searches are written under `searches/<model>/<scope>/...`. Both roots are
ignored by git. Small frozen reference metrics are kept in `results/reference/`.

## `benchmark_methylprophet/`

Isolated MethylProphet-benchmark data preparation (`prepare.py`) that backs
the exact matched-chr1 prepared cache but isn't part of the
generic four-entrypoint workflow — see
[`../docs/BENCHMARK_METHYLPROPHET.md`](../docs/BENCHMARK_METHYLPROPHET.md).
The earlier exact two-stage-architecture reproduction path
(`run_experiment.py`/`analyze_context.py`/`resolve_final_epoch_budget.py`/
`run.sh`) has been retired along with that architecture generation; its
frozen numbers remain under `results/reference/methylprophet_comparison/`.
