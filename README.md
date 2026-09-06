# MethylPredictor

Research framework for reconstructing DNA methylation from bulk RNA with frozen
NTv3 locus representations.

The repository has **two canonical trainable models** and one shared genomic
scope axis:

- `CpGStatisticsPredictor`: NTv3 CpG embedding -> locus mean `mu` and logit-scale `sigma`;
- `FeatureFusionLocusCLSModel`/`FeatureFusionArchitectureVariantModel`: RNA + CpG
  embedding -> sample-specific methylation `beta_hat` directly (single-stage
  shared backbone, no explicit prior/residual composition);
- scopes: `chr1`, `chr123` (`chr1 ∪ chr2 ∪ chr3`) and `genomewide`.

The primary/reference architecture as of 2026-09-05 is the single-stage
shared-backbone model with a locus-conditioned RNA-attention encoder:

```text
CpG reference context -> frozen NTv3 1536-D embedding e_l -> CpGTrunk -> h_mean_l
patient RNA x_s -> 64 learned RNA tokens -> cross-attention over e_l -> r_s,l
[r_s,l, e_l] -> raw interaction branch h_raw_s,l
[h_mean_l, h_raw_s,l] -> fusion -> beta_hat_s,l
```

See `docs/RNA_METHYLATION.md` for the full architecture history (including the
earlier two-stage frozen-prior + residual generation, retired) and the ongoing
RNA-encoder comparison harness.

## Frozen reference results

The current shared-backbone chr1 reference (Array + EPIC + WGBS; LR `5e-5`,
constant scheduler, seed 17) reaches, on the official MethylProphet chr1 split
-- a **lower bound**, stopped at epoch 47/80:

- train-CpG × val-sample: **0.6431 / 0.01351**
- val-CpG × train-sample: **0.6081 / 0.01613**
- val-CpG × val-sample: **0.5627 / 0.01702**

The chr123/genome-wide numbers in `results/reference/rna_methylation/` predate
this architecture (produced by the retired two-stage engine) and are pending a
shared-backbone rerun -- see `docs/PAPER_EXPERIMENTS.md` for current status.

Machine-readable references live in `results/reference/`. `chr1` is the
MethylProphet-comparison scope (official split independently verified against
the released MethylProphet evaluation artifact — see
`docs/BENCHMARK_METHYLPROPHET.md`); `genomewide` is the general benchmark.
`chr123` is a usable general scope but not currently a verified MethylProphet
comparison.

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

Train either model. RNA-methylation always uses the shared-backbone engine
(`--engine matched_chr1_shared_backbone` for the exact chr1 MethylProphet
preparation, `--engine shared_backbone` for chr123/genome-wide):

```bash
python scripts/train.py --model cpg_statistics --scope genomewide \
  --recipe configs/models/cpg_statistics.yaml ...

python scripts/train.py --model rna_methylation --scope chr123 \
  --engine shared_backbone \
  --recipe configs/models/rna_methylation_locus_attention.yaml \
  --cpg-targets-dir /path/to/derived/cpg_statistics/chr123 ...
```

Tune LR/scheduler/epoch budget without opening official benchmark validation
cells:

```bash
python scripts/tune.py --model rna_methylation --scope chr123 \
  --cpg-targets-dir /path/to/derived/cpg_statistics/chr123 \
  --lrs 2e-5,5e-5,8e-5 --schedulers constant,cosine_warmup \
  --max-epochs 80 ...
```

Evaluate any RNA checkpoint on any genomic scope:

```bash
python scripts/evaluate.py --model rna_methylation \
  --checkpoint /path/to/best.pt --eval-scope genomewide \
  --cpg-targets-dir /path/to/derived/cpg_statistics/genomewide ...
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
for the chr1 MethylProphet-matched data preparation/split verification. Legacy
Table-5/cache names may remain inside prepared-data provenance paths, but they
are not public experiment identities.
