# Script map

The public workflow has four generic entrypoints:

- `prepare_statistics.py` — build multi-technology `mu`/`sigma` labels;
- `train.py` — train either `cpg_statistics` or `rna_methylation`;
- `tune.py` — leakage-safe LR/scheduler/epoch search on inner development data;
- `evaluate.py` — evaluate either model on `chr1`, `chr123` or `genomewide`.

`export_statistics_cache.py` converts a trained CpG statistics checkpoint to the
`cpg_idx + NTv3 embeddings + prior + sigma` cache consumed by the RNA model.

The exact validated chr1 compatibility path remains available through:

```bash
python scripts/train.py --model rna_methylation --scope chr1 --engine matched_chr1 ...
```

It reuses the existing prepared TCGA-chr1 caches and pair-complete trainer.
This is the path for reproducing the MethylProphet-matched E1 reference.

## Generated data

Training runs are written under `runs/<model>/<scope>/...`; hyperparameter
searches are written under `searches/<model>/<scope>/...`. Both roots are
ignored by git. Small frozen reference metrics are kept in `results/reference/`.

## Legacy/closed experiments

Historical specialized launchers remain only until the refactored real-data
smoke tests pass. Then run:

```bash
bash scripts/cleanup_closed_experiments.sh
```

Review the staged deletions before committing. Git history is the archive for
closed one-off experiments; the working tree should expose only reusable model,
protocol, preparation, training and evaluation code.
