# TCGA chromosome-1 development experiments

## Naming

The benchmark is named **TCGA chr1** throughout active development.
MethylProphet Table 5 is the published comparator, not the identity of the
dataset. Legacy prepared-cache paths may still include `table5` so existing
multi-GB caches can be reused without duplication.

## Reference

The current reference is the variance-normalized residual model:

`logit(beta_hat[s,i]) = mu_i + sigma_i * raw_delta[s,i]`

with beta-MSE 1.0, standardized residual Huber 0.1, standardized shrinkage
1e-4 and locus-Pearson 0.15. Development uses seed 17 only. Multi-seed
robustness is intentionally deferred until the final configuration is frozen.

## Experiments

### `large_sample_pcc`

Changes only the Array Cartesian block geometry from 128 samples × 2048 CpGs
to 512 samples × 512 CpGs. The pair count per optimization unit is almost
unchanged, so activation memory remains comparable, while each per-CpG Pearson
estimate uses up to four times as many patients.

### `tail_aware_pcc`

Keeps the total correlation weight at 0.15 but assigns 0.05 to the lower 60%
of valid per-CpG correlations and 0.10 to the mean. A target-standard-deviation
floor of 0.02 prevents the structured objective from chasing almost-constant
loci whose Pearson coefficient is unstable.

### `array_only_structured`

Pointwise beta-MSE and variance-normalized residual objectives remain active on
Array, EPIC and WGBS. Pearson-family and other structured objectives are active
only on Array. This tests whether WGBS is most useful for genomic locus coverage
while its 32 measurements are too few to provide a stable patient-dynamic
correlation gradient.

## Run order

Run the three experiments independently from the same reference. Do not chain
checkpoints. Combine only changes that individually improve the development
criterion.

```bash
conda activate methyl-predictor

CUDA_VISIBLE_DEVICES=0 python scripts/tcga_chr1/run_experiment.py \
  --experiment configs/tcga_chr1/experiments/large_sample_pcc.yaml --epochs 25

CUDA_VISIBLE_DEVICES=0 python scripts/tcga_chr1/run_experiment.py \
  --experiment configs/tcga_chr1/experiments/tail_aware_pcc.yaml --epochs 25

CUDA_VISIBLE_DEVICES=0 python scripts/tcga_chr1/run_experiment.py \
  --experiment configs/tcga_chr1/experiments/array_only_structured.yaml --epochs 25
```

Outputs are stored by semantic experiment ID under
`.../experiments/MethylPredictor/tcga_chr1/`; the seed is metadata rather than
the directory identity.
