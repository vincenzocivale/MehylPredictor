# TCGA matrix bilinear model

This run keeps the validated NTv3-prior + RNA-residual formulation while replacing pair-at-a-time decoding with two cacheable towers and a matrix product.

## Computation

For patients `s` and CpGs `i`:

```text
patient_factor[s] = P(RNA_encoder(x_s) - RNA_encoder(mean_RNA))
locus_factor[i]   = Q(NTv3_embedding_i)
delta[s, i]       = patient_factor[s] @ locus_factor[i] / sqrt(rank)
beta_hat[s, i]    = sigmoid(logit(prior_i) + gate_i * delta[s, i])
```

Training samples a Cartesian block of patients and CpGs and predicts the full block with one GEMM. At inference, patient factors are computed once, locus factors are computed once per locus block, and all pair predictions are produced by matrix multiplication. Existing bilinear checkpoints remain loadable because parameter names and shapes are unchanged.

## Interaction choice

`model.interaction.kind` defaults to `bilinear` (the factorized/cacheable
path described above). `interaction.kind: concat` is bit-identical to the
Stage F/G winner "F2" (`ConcatInteraction` in `models.py`, verified
byte-identical against commit `f54eb593` on 2026-08-01: `git diff
f54eb593..HEAD -- src/methylation_predictor/rna_branch/models.py` shows no
changes to the class). F2 beat bilinear on validation MSE and
high-variability-tertile skill at all 3 confirm-grid seeds (17/23/41, see
`docs/rna_fusion_stage_f.md`), but its win is a specific synergy — raw,
unprojected 1536-dim NTv3 locus embedding concatenated alongside a
separately-projected elementwise product term — not a clean "concat" or
"product" in isolation (neither alone beats bilinear, see
`docs/rna_fusion_stage_g.md`). On Stage D2's pre-registered hierarchical
(patient × genomic-block) bootstrap, F2's advantage over the bilinear
baseline (C0) does not exclude zero (`docs/stage_d2_f2_report.md`).
Treat `concat` as **promising, not settled**: worth a matched matrix-scale
run against `bilinear`, but not yet a default to switch to without a fresh
paired-bootstrap verdict at this scale.

Unlike `bilinear`, `concat` does not support factorized/cacheable inference
(`ResidualMethylationModel.supports_factorized_inference` is `True` only for
`BilinearInteraction`) — every forward pass materializes the full
`[batch, n_loci, rna_dim + locus_dim + min(rna_dim, locus_dim)]` tensor.
At this project's scale (`locus_dim=1536`) that means the default
`evaluation.sample_chunk_size`/`evaluation.cpg_chunk_size` (tuned for the
bilinear cached path) are far too large for `concat` and will exhaust GPU
memory — shrink both substantially in any `concat`-based config.

## Setup

```bash
python -m pip install -r requirements-rna.txt
wandb login
python -m methylation_predictor.rna_branch.cli validate \
  --config configs/rna_branch/tcga_matrix_bilinear.yaml
```

## Train

```bash
python -m methylation_predictor.rna_branch.cli train \
  --config configs/rna_branch/tcga_matrix_bilinear.yaml
```

On Slurm:

```bash
sbatch jobs/slurm/train_tcga_matrix_bilinear.sbatch
```

The default configuration uses BF16 autocast when supported, TF32 matrix multiplication, fused AdamW when available, balanced CpG-variability sampling, W&B batch/epoch metrics, throughput, peak GPU memory, validation metrics, and a best-checkpoint artifact.

For the final comparison, duplicate the configuration for seeds `23` and `41`, changing `run_name`, `output_dir`, `training.seed`, and `tracking.name` only.

## Memory-bounded full-panel inference

The normal `predict` command still writes an in-memory compressed NPZ. For large panels, stream directly to chunked HDF5:

```bash
python -m methylation_predictor.rna_branch.cli predict-stream \
  --config configs/rna_branch/tcga_matrix_bilinear.yaml \
  --checkpoint artifacts/rna_branch/tcga_matrix_bilinear/seed17/best.pt \
  --sample-split test \
  --cpg-split test \
  --output artifacts/rna_branch/tcga_matrix_bilinear/seed17/double_ood.h5 \
  --include-target
```

The HDF5 file contains `prediction`, optional `target`, `prior`, `sample_idx`, and `cpg_idx`. Increase or decrease `evaluation.sample_chunk_size` and `evaluation.cpg_chunk_size` according to GPU memory.

## Fair comparison with MethylProphet

Report both predictive quality and systems metrics:

- beta-space MSE/MAE, skill versus the same prior, dynamic skill, within-cancer skill;
- patient-wise and locus-wise dynamic correlation;
- wall-clock training time, observed pairs/second, peak allocated GPU memory;
- full-panel inference time and pairs/second;
- model parameter count and checkpoint size.

Do not claim a speedup from a synthetic CPU test. Measure both implementations on the same GPU, panel, precision, and data-loading setup.

## Measure the cache speedup on the target GPU

```bash
python -m methylation_predictor.rna_branch.cli benchmark-inference \
  --config configs/rna_branch/tcga_matrix_bilinear.yaml \
  --checkpoint artifacts/rna_branch/tcga_matrix_bilinear/seed17/best.pt \
  --sample-split test \
  --cpg-split test \
  --max-cpgs 521 \
  --cpg-chunk-size 128
```

This runs the same checkpoint and panel twice. The uncached path repeatedly evaluates the patient tower for every locus chunk; the cached path evaluates each tower once per axis. It reports wall time, throughput, speedup, and the maximum absolute prediction difference.
