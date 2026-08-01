#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="/home/oem/miniconda3/envs/bulkrnabert/bin:$PATH"
export PYTHONUNBUFFERED=1

CACHE=artifacts/rna_encoder_readout/cache/bulkrnabert_l1_l2_l3_4096.h5
RNA=artifacts/rna_branch/pretrained/inputs/tcga_rna_full_gene.h5

echo "=== matched-target baselines (PCA-256, random-projection-256) ==="
python3 scripts/rna_encoder_readout/ridge_probe_compare.py \
  --cache "$CACHE" \
  --rna-h5 "$RNA" \
  --target-gene-count 4096 \
  --selection-seed 17 \
  --candidate "full_mean_layer2=full_mean_layer2" \
  --candidate "full_mean_layer3=full_mean_layer3" \
  --candidate "mean_resume_layer2=artifacts/rna_encoder_readout/search/p0_mean_resume_layer2/embeddings.h5" \
  --candidate "concat_means=artifacts/rna_encoder_readout/search/p0_concat_means_cached_layers/embeddings.h5" \
  --candidate "pca256=pca256" \
  --candidate "randproj256=randproj256" \
  --output artifacts/rna_encoder_readout/ridge_probe_baselines.csv

echo "=== P1: 6 ridge-alternating pooling experiments ==="
python3 scripts/rna_encoder_readout/run_readout_screen.py \
  --config-dir artifacts/rna_encoder_readout/configs/p1 \
  --stop-on-error

echo "=== P1 done ==="
