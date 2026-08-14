# MethylPredictor

RNA-conditioned DNA methylation reconstruction using frozen NTv3 CpG
representations.

## Reference model

```text
RNA -> 256-D latent
CpG -> frozen NTv3 1536-D embedding
RNA × CpG product interaction -> standardized residual
logit(beta_hat) = mu_i + sigma_i * residual
```

Current V1 MAS-PCC on the three TCGA chr1 benchmark views:
**0.5811 / 0.5708 / 0.5401**.

Documentation:
- `docs/MODEL.md`
- `docs/BENCHMARK_TABLE5.md`
- `docs/DATA.md`
- `docs/TRAINING.md`
- `docs/RESULTS.md`
- `docs/ABLATIONS.md`
- `docs/TCGA_CHR1_EXPERIMENTS.md`

## Installation

```bash
conda activate methyl-predictor
python -m pip install -r requirements.txt
python -m pip install -r requirements-genomics.txt
python -m pip install -e .
```

## Prepare/audit TCGA chr1 data

```bash
HG38_FASTA=/raid/DATASETS/MethylPredictionData/reference/hg38/hg38.fa GPU=0 PREPARE_ONLY=1 bash scripts/tcga_chr1/run.sh
```

## Train reference / ablations

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/tcga_chr1/run_experiment.py \
  --experiment configs/tcga_chr1/experiments/reference.yaml --epochs 25

CUDA_VISIBLE_DEVICES=0 python scripts/tcga_chr1/run_experiment.py \
  --experiment configs/tcga_chr1/experiments/large_sample_pcc.yaml --epochs 25

CUDA_VISIBLE_DEVICES=0 python scripts/tcga_chr1/run_experiment.py \
  --experiment configs/tcga_chr1/experiments/tail_aware_pcc.yaml --epochs 25

CUDA_VISIBLE_DEVICES=0 python scripts/tcga_chr1/run_experiment.py \
  --experiment configs/tcga_chr1/experiments/array_only_structured.yaml --epochs 25
```

Development uses one fixed seed (`17`). Multi-seed robustness is deliberately
the last experiment after the final configuration is frozen. See
[`docs/TCGA_CHR1_EXPERIMENTS.md`](docs/TCGA_CHR1_EXPERIMENTS.md).

New outputs live under:

```text
/raid/DATASETS/MethylPredictionData/experiments/MethylPredictor/tcga_chr1/<experiment_id>/
```

with `config.resolved.yaml`, `experiment.json`, `manifest.json`,
`checkpoints/`, `metrics/`, `evaluation/`, and `analysis/`. "Table 5" is
retained only inside legacy protocol/cache identifiers (e.g. the prepared
derived-cache path) for provenance; it is not the experiment identity.
