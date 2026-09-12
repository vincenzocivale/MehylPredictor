# Paper experiment API

The final paper-facing experiment interface is deliberately smaller than the
historical research interface.

For a retained recipe, one command owns the sequence:

```text
train -> official evaluation -> standardized run record
```

The entry point is:

```text
scripts/paper_experiment.py
```

It is intended for the fresh paper reruns performed after the final
architecture has been selected. Protected J-series launchers remain separate
and resumable until that decision is locked.

## Safety contract

The wrapper:

1. accepts only model recipes classified as `paper_facing` in
   `configs/experiment_surface.yaml`;
2. verifies the recipe SHA-256 against that frozen manifest;
3. refuses final RNA training from a dirty Git tree unless `--allow-dirty` is
   explicitly supplied for a non-final smoke run;
4. keeps the historical runtime storage key behind the data profile rather
   than exposing it as public API;
5. always evaluates the three official views;
6. writes one normalized `paper/record.json` containing recipe, Git,
   checkpoint, resolved-config and evaluation provenance.

Therefore a protected J-series recipe cannot accidentally be launched through
the final-paper interface.

## Data profile

The current matched-chr1 profile is:

```text
configs/data/paper_chr1.yaml
```

Paths are relative to `--data-root` (or `METHYL_DATA_ROOT`). The profile owns
the data/cache layout and temporary storage compatibility key.

Machine-specific absolute paths do not belong in the profile.

## Standard final RNA run

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/paper_experiment.py \
  --profile configs/data/paper_chr1.yaml \
  --data-root "$METHYL_DATA_ROOT" \
  --recipe configs/models/main.yaml \
  --run-id main-seed17 \
  --seed 17 \
  --study main_model \
  --arm main \
  --stage all
```

Before executing a run, inspect the exact commands with:

```bash
python scripts/paper_experiment.py \
  --profile configs/data/paper_chr1.yaml \
  --data-root "$METHYL_DATA_ROOT" \
  --recipe configs/models/main.yaml \
  --run-id main-seed17 \
  --seed 17 \
  --stage all \
  --dry-run
```

For RNA-encoder comparators whose precomputed RNA representation lives in a
different cache, pass `--rna-cache /path/to/cache`. All other data inputs stay
fixed by the profile.

## Resume

A completed training run is never silently overwritten. If an incomplete run
directory exists, inspect it first and resume explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/paper_experiment.py \
  ... \
  --resume \
  --stage all
```

## Zero-parameter CpG-prior baseline

The CpG-prior recipe is paper-facing but has no training stage:

```bash
python scripts/paper_experiment.py \
  --profile configs/data/paper_chr1.yaml \
  --data-root "$METHYL_DATA_ROOT" \
  --recipe configs/models/baselines/baseline_cpg_prior.yaml \
  --run-id cpg-prior \
  --study functional_baselines \
  --arm cpg_prior \
  --stage all
```

For this recipe, `all` means evaluation plus record creation.

## Standardized run record

Each run produces:

```text
<runtime-run-dir>/
  training/
    summary.json                 # trainable models
  checkpoints/
    best.pt                      # trainable models
  evaluation/
    <scope>/
      metrics.json
  paper/
    record.json
```

`record.json` is the canonical input to paper result collection. It contains:

- study / arm / seed / run ID;
- Git commit and dirty flag;
- recipe path and SHA-256;
- data-profile path and SHA-256;
- resolved data paths used on that machine;
- runtime run directory / compatibility storage key;
- training summary and SHA-256, when applicable;
- checkpoint path, SHA-256 and epoch, when applicable;
- resolved-config SHA-256, when available;
- all three official evaluation views;
- the double-OOD headline metrics.

The output format is intentionally independent of study-specific historical
collectors.

## Collection

After final runs exist:

```bash
python scripts/collect_paper_runs.py \
  --profile configs/data/paper_chr1.yaml \
  --data-root "$METHYL_DATA_ROOT" \
  --output-dir results/paper
```

This scans only standardized `paper/record.json` files and writes:

```text
results/paper/
  runs.jsonl
  runs.csv
  manifest.json
```

Study-specific tables can be generated from this uniform ledger during the
final paper-table phase. The collector does not scrape historical
`results/reference/` layouts.

## Architecture-selection boundary

Until the J-series finishes:

- do not invoke protected J recipes through this API;
- do not change their recipe or launcher hashes;
- do not migrate the historical runtime storage key.

After architecture selection, the selected implementation is promoted to
`configs/models/main.yaml` and its manifest hash is updated. The same paper API
then launches all fresh final experiments without any J-series-specific
command.
