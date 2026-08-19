# MethylPredictor

Research framework for reconstructing DNA methylation from bulk RNA with frozen
NTv3 locus representations.

The repository has **two canonical trainable models** and one shared genomic
scope axis:

- `CpGStatisticsPredictor`: NTv3 CpG embedding -> locus mean `mu` and logit-scale `sigma`;
- `RNAMethylationPredictor`: RNA + CpG embedding + (`mu`, `sigma`) -> sample-specific methylation;
- scopes: `chr1`, `chr123` (`chr1 ∪ chr2 ∪ chr3`) and `genomewide`.

The former “V1” is now the canonical RNA model:

```text
RNA -> LayerNorm -> Linear(25017, 256)
CpG -> frozen NTv3 1536-D embedding
RNA x CpG product interaction -> raw_delta
logit(beta_hat) = logit(mu_i) + sigma_i * raw_delta
```

## Frozen reference results

The current mixed-source chr1 model (Array + EPIC + WGBS; LR `5e-5`, constant
scheduler, 80 epochs, seed 17) reaches MAS-PCC/MSE:

- train-CpG × val-sample: **0.6327 / 0.01278**
- val-CpG × train-sample: **0.5984 / 0.01874**
- val-CpG × val-sample: **0.5613 / 0.01941**

Genome-wide Array evaluation of the frozen current architecture reaches:

- **0.5889 / 0.01482**
- **0.5800 / 0.02099**
- **0.5244 / 0.02223**

Machine-readable references live in `results/reference/`. `chr1` and `chr123`
are the MethylProphet-comparison scopes; `genomewide` is the general benchmark.

## Installation

```bash
conda activate methyl-predictor
python -m pip install -r requirements.txt
python -m pip install -r requirements-genomics.txt
python -m pip install -e .
```

## Unified workflows

Prepare technology-aware static CpG targets, then the RNA model's input cache:

```bash
python scripts/prepare.py --model cpg_statistics \
  --canonical-root "$TCGA_CANONICAL_ROOT" \
  --registry "$TCGA_CANONICAL_ROOT/registries/array_cpg_map.parquet" \
  --scope genomewide \
  --output /path/to/derived/cpg_statistics/genomewide

python scripts/prepare.py --model rna_methylation \
  --checkpoint /path/to/cpg_statistics/best.pt \
  --targets /path/to/derived/cpg_statistics/genomewide \
  --embeddings /path/to/ntv3_cpg_atlas_v1.h5 \
  --output /path/to/derived/rna_cache/genomewide
```

Train either model:

```bash
python scripts/train.py --model cpg_statistics --scope genomewide \
  --recipe configs/models/cpg_statistics.yaml ...

python scripts/train.py --model rna_methylation --scope chr123 \
  --recipe configs/models/rna_methylation.yaml ...
```

Tune LR/scheduler/epoch budget without opening official benchmark validation
cells:

```bash
python scripts/tune.py --model rna_methylation --scope chr123 \
  --lrs 2e-5,5e-5,8e-5 --schedulers constant,cosine_warmup \
  --max-epochs 80 ...
```

Evaluate any RNA checkpoint on any genomic scope:

```bash
python scripts/evaluate.py --model rna_methylation \
  --checkpoint /path/to/best.pt --eval-scope genomewide ...
```

The full 3×3 train/evaluation scope matrix is supported. Same-scope chr1 and
chr123 cells are matched MethylProphet benchmarks; cross-scope cells quantify
generalization.

## Output layout

Generated runs and searches are outside version control:

```text
runs/<model>/<train-scope>/<run-id>/
  config.resolved.yaml
  metadata.json
  checkpoints/
  training/
  evaluation/<eval-scope>/
  logs/

searches/<model>/<scope>/<search-id>/
  search_config.yaml
  candidates.csv
  selected.json
  runs/
```

See `docs/WORKFLOWS.md` for protocol details, `docs/RNA_METHYLATION.md` and
`docs/CPG_STATISTICS.md` for the two model architectures, `docs/BENCHMARKS.md`
for the full reference-result tables, and `docs/BENCHMARK_METHYLPROPHET.md`
for the frozen, exact MethylProphet chr1 reproduction path. Legacy Table-5/
cache names may remain inside prepared-data provenance paths, but they are
not public experiment identities.
