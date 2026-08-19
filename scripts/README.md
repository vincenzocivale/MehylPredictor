# Script map

The public workflow has four entrypoints:

- `prepare.py` — build data for either model (`--model cpg_statistics`:
  multi-technology `mu`/`sigma` labels; `--model rna_methylation`: the
  `cpg_idx + NTv3 embeddings + prior + sigma` cache exported from a trained
  `cpg_statistics` checkpoint);
- `train.py` — train either `cpg_statistics` or `rna_methylation`;
- `tune.py` — leakage-safe LR/scheduler/epoch search on inner development data;
- `evaluate.py` — evaluate either model on `chr1`, `chr123` or `genomewide`.

The exact validated chr1 compatibility path remains available through:

```bash
python scripts/train.py --model rna_methylation --scope chr1 --engine matched_chr1 ...
```

It reuses the prepared chr1 caches (`scripts/benchmark_methylprophet/prepare.py`)
and the frozen pair-complete `MethylProphetTrainer`. This is the path for
reproducing the MethylProphet-matched Table-5 chr1 reference — see
[`../docs/BENCHMARK_METHYLPROPHET.md`](../docs/BENCHMARK_METHYLPROPHET.md).

## Generated data

Training runs are written under `runs/<model>/<scope>/...`; hyperparameter
searches are written under `searches/<model>/<scope>/...`. Both roots are
ignored by git. Small frozen reference metrics are kept in `results/reference/`.

## `benchmark_methylprophet/`

Isolated MethylProphet-benchmark-exclusive scripts (data preparation,
per-context analysis, reporting) that back the `--engine matched_chr1` path
above but aren't part of the generic four-entrypoint workflow. Not intended
as a second architecture — see
[`../docs/BENCHMARK_METHYLPROPHET.md`](../docs/BENCHMARK_METHYLPROPHET.md).
