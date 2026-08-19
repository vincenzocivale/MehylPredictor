# Refactored workflows

The repository has two trainable model families and one genomic-scope axis.
Experiment history is not an API.

## Models

### `cpg_statistics`

Predicts static locus statistics from frozen NTv3 embeddings:

- `mu`: beta-space locus mean;
- `sigma`: logit-space inter-sample scale used by the RNA residual model.

Targets can use Array, EPIC and WGBS simultaneously.  The target builder records
per-technology counts and requires an explicit aggregation policy:
`sample_weighted` or `technology_balanced`.  Official Array validation patients
are excluded from target construction by default; auxiliary-only patients remain
available as training evidence.

### `rna_methylation`

The canonical model is the former V1 variance-normalized residual architecture:

```
RNA -> LayerNorm -> Linear(25017, 256)
CpG -> frozen NTv3 1536-D embedding
RNA x CpG product interaction -> raw_delta
logit(beta_hat) = logit(mu) + sigma * raw_delta
```

The flat residual `RNA2DNAmModel` remains compatibility/ablation code only.

## Genomic scopes

- `chr1`
- `chr123` = chr1 union chr2 union chr3
- `genomewide`

The scope filters the frozen official CpG split; it never changes architecture.
`chr1` and `chr123` are MethylProphet-matched benchmark scopes.  Genome-wide is
the primary general benchmark.

## Training

Use one entrypoint:

```bash
python scripts/train.py --model rna_methylation --scope chr123 ...
python scripts/train.py --model cpg_statistics --scope genomewide ...
```

Learning rate, epoch budget, scheduler and seed can be overridden from the CLI.
The default RNA recipe is the frozen development winner: LR 5e-5, constant
scheduler, 80 epochs, seed 17, Array layout 512 x 512.

For exact reproduction of the validated chr1 E1 run, use:

```bash
python scripts/train.py \
  --model rna_methylation --scope chr1 --engine matched_chr1 \
  --prepared-root /path/to/prepared/benchmark_methylprophet \
  --canonical-root /path/to/canonical \
  --feature-cache /path/to/features \
  --rna-cache /path/to/rna \
  --registry /path/to/array_cpg_map.parquet \
  --recipe configs/models/rna_methylation.yaml \
  --output-root /path/to/experiments
```

For genome-wide training the scalable default is `axis_full_coverage`: every
sample and every CpG is touched in each epoch, but the complete Cartesian matrix
is not enumerated.  Pair-complete training remains explicit for matched smaller
benchmarks; the schedule policy is always recorded in run metadata.

## Hyperparameter search

`python scripts/tune.py` creates an inner-development split wholly inside the
official training universe. RNA selection uses inner double-OOD MAS-PCC. The
official MethylProphet-matched validation cells are not used for model selection.

Search output:

```
searches/<model>/<scope>/<search-id>/
  search_config.yaml
  candidates.csv
  selected.json
  runs/
```

## Evaluation

Any RNA checkpoint can be evaluated on any scope:

| train scope | chr1 | chr123 | genomewide |
|---|---:|---:|---:|
| chr1 | yes | yes | yes |
| chr123 | yes | yes | yes |
| genomewide | yes | yes | yes |

Same-scope chr1 and chr123 are matched benchmark cells; cross-scope cells are
generalization analyses.

```
python scripts/evaluate.py --model rna_methylation \
  --checkpoint ... --eval-scope genomewide ...
```

## Run storage

Training output is never mixed with search output:

```
runs/<model>/<train-scope>/<run-id>/
  config.resolved.yaml
  metadata.json
  checkpoints/{best.pt,last.pt}
  training/{history.*,summary.json}
  evaluation/<eval-scope>/{metrics.json,per_chromosome.csv,manifest.json}
  logs/
```

`metadata.json` records model, training scope, data contract, hyperparameters,
Git commit and the feature/RNA cache provenance available at launch. Evaluation
manifests additionally record checkpoint SHA256 and evaluation scope.

`runs/` and `searches/` are gitignored. Small frozen reference numbers live in
`results/reference/` and are version controlled.
