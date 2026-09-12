# Final paper training

The frozen paper training surface contains 37 jobs:

- 3 final-model runs;
- 9 functional-baseline runs;
- 9 mean-proxy runs;
- 15 RNA-encoder runs;
- 1 zero-parameter CpG-prior evaluation.

All new trainable runs are written under:

```text
$METHYL_DATA_ROOT/experiments/runs/rna_methylation/chr1/
```

## Preflight

On every machine:

```bash
export METHYL_DATA_ROOT=<machine-specific mount>/MethylPredictionData

python scripts/preflight_paper_training.py   --allow-blocked-bulkrnabert
```

Before launching the complete campaign, remove
`--allow-blocked-bulkrnabert`; the preflight must then be fully green.

Inspect the deterministic job table with:

```bash
python scripts/run_paper_jobs.py --list
```

BulkFormer must show its cache under:

```text
derived/bulkformer_embeddings/tcga_147m
```

BulkRNABert must show:

```text
derived/bulkrnabert_embeddings/tcga
```

## Multi-machine launch

For three machines, each exposing the shared data root:

```bash
python scripts/run_paper_jobs.py --shard 0/3 --gpu 0
python scripts/run_paper_jobs.py --shard 1/3 --gpu 0
python scripts/run_paper_jobs.py --shard 2/3 --gpu 0
```

Run one command per machine.

The runner skips a job only when its final
`paper/record.json` exists. Interrupted runs are therefore not mistaken for
completed paper runs.

## Collect results

After jobs complete:

```bash
python scripts/collect_paper_runs.py
```

This materializes compact, versionable records below:

```text
results/paper/
```

Heavy artifacts remain in `METHYL_DATA_ROOT` and are referenced through
`methyl-data://` URIs.
