#!/usr/bin/env bash
#SBATCH --job-name=rna-encoder
#SBATCH --output=logs/rna_encoder_%A_%a.out
#SBATCH --error=logs/rna_encoder_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

MANIFEST=${MANIFEST:?Set MANIFEST to the materialized manifest.txt}
TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit this script as a SLURM array}
CONFIG=$(sed -n "$((TASK_ID + 1))p" "$MANIFEST")
if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
  echo "No config for task ${TASK_ID} in ${MANIFEST}" >&2
  exit 2
fi

mkdir -p logs
python -m methylation_predictor.rna_branch validate --config "$CONFIG"
python -m methylation_predictor.rna_branch train --config "$CONFIG"
