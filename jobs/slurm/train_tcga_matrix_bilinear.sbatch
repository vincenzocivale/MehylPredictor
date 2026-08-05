#!/usr/bin/env bash
#SBATCH --job-name=tcga-matrix-bilinear
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_DIR="${WANDB_DIR:-$PWD/artifacts/wandb}"
mkdir -p "$WANDB_DIR"

python -m methylation_predictor.rna_branch.cli train \
  --config configs/rna_branch/tcga_matrix_bilinear.yaml
