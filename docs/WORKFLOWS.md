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

The reference architecture is the single-stage shared-backbone model
(`FeatureFusionArchitectureVariantModel` with `encoder.kind=locus_attention`, see
[`RNA_METHYLATION.md`](RNA_METHYLATION.md)):

```
CpG reference context -> frozen NTv3 1536-D embedding -> CpGTrunk -> h_mean
patient RNA -> locus-conditioned RNA-attention encoder -> h_raw (RNA x CpG)
[h_mean, h_raw] -> fusion -> beta_hat
```

Trained by `LocusCLSJointTrainer`, always via `scripts/train.py`'s shared-backbone
engines (`matched_chr1_shared_backbone` for chr1, `shared_backbone` otherwise). The
earlier two-stage frozen-prior + residual architecture (`RNA2DNAmModel`/
`VarianceNormalizedResidualModel`/`RNAMethylationPredictor`) has been retired -- see
`RNA_METHYLATION.md`'s "Retired architecture" section.

## Genomic scopes

- `chr1`
- `chr123` = chr1 union chr2 union chr3
- `genomewide`

The scope filters the frozen official CpG split; it never changes architecture.
`chr1` is the (verified) MethylProphet-matched benchmark scope; genome-wide is
the primary general benchmark. `chr123` is a usable general scope but not
currently a verified MethylProphet comparison -- see
[`BENCHMARK_METHYLPROPHET.md`](BENCHMARK_METHYLPROPHET.md).

## Training

Use one entrypoint:

```bash
python scripts/train.py --model rna_methylation --scope chr123 --engine shared_backbone \
  --cpg-targets-dir /path/to/derived/cpg_statistics/chr123 ...
python scripts/train.py --model cpg_statistics --scope genomewide ...
```

Learning rate, epoch budget, scheduler and seed can be overridden from the CLI.
The reference RNA recipe is `configs/models/rna_methylation_locus_attention.yaml`: LR
5e-5, constant scheduler, 80 epochs, seed 17, Array layout 512 x 512.

For the exact MethylProphet-matched chr1 preparation, use:

```bash
python scripts/train.py \
  --model rna_methylation --scope chr1 --engine matched_chr1_shared_backbone \
  --prepared-root /path/to/prepared/benchmark_methylprophet \
  --canonical-root /path/to/canonical \
  --feature-cache /path/to/features \
  --rna-cache /path/to/rna \
  --registry /path/to/array_cpg_map.parquet \
  --cpg-targets-dir /path/to/derived/cpg_statistics/chr1 \
  --recipe configs/models/rna_methylation_locus_attention.yaml \
  --output-root /path/to/repository/results/experiments
```

For genome-wide training the scalable default is `axis_full_coverage`: every
sample and every CpG is touched in each epoch, but the complete Cartesian matrix
is not enumerated.  Pair-complete training remains explicit for matched smaller
benchmarks; the schedule policy is always recorded in run metadata.

For pair-complete chr123 shared-backbone runs, prepare and use the compact target
cache described in
[`CHR123_TRAINING_OPTIMIZATIONS.md`](CHR123_TRAINING_OPTIMIZATIONS.md). It preserves
the protocol while avoiding sparse HDF5 selections; on the reference server it reduced
epoch time from about 18 minutes to 2.4 minutes.

## Hyperparameter search

`python scripts/tune.py` creates an inner-development split wholly inside the
official training universe. RNA selection uses inner double-OOD MAS-PCC. The
official MethylProphet-matched validation cells are not used for model selection.

Search output:

```
results/experiments/searches/<model>/<scope>/<search-id>/
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

The chr1/chr1 cell is the verified MethylProphet-matched comparison; chr123/chr123
is a same-scope cell but not currently a verified MethylProphet comparison
(see [`BENCHMARK_METHYLPROPHET.md`](BENCHMARK_METHYLPROPHET.md)); other
cross-scope cells are generalization analyses.

```
python scripts/evaluate.py --model rna_methylation \
  --checkpoint ... --eval-scope genomewide ...
```

## Run storage

Training output is never mixed with search output:

```
results/experiments/runs/<model>/<train-scope>/<run-id>/
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

The raw `runs/` and `searches/` contents are gitignored. Small frozen reference
numbers live in `results/reference/` and are version controlled.
