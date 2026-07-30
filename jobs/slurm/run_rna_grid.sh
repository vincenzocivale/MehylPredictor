#!/usr/bin/env bash
#SBATCH --job-name=rna-branch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rna_branch/%A_%a.out
#SBATCH --error=logs/rna_branch/%A_%a.err

set -euo pipefail

# Usage:
#   python -m methylation_predictor.rna_branch.grid \
#     --grid configs/rna_branch/signal_grid.yaml \
#     --output-dir artifacts/rna_branch/generated_configs/signal
#   N=$(wc -l < artifacts/rna_branch/generated_configs/signal/manifest.txt)
#   sbatch --array=1-${N} jobs/slurm/run_rna_grid.sh \
#     artifacts/rna_branch/generated_configs/signal/manifest.txt

MANIFEST=${1:?"pass the generated config manifest"}
CONFIG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
if [[ -z "$CONFIG" ]]; then
  echo "No config for task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

mkdir -p logs/rna_branch

# Adapt these lines to the cluster environment.
# module load cuda/12.4
# source .venv/bin/activate

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

python -m methylation_predictor.rna_branch validate --config "$CONFIG"
python -m methylation_predictor.rna_branch train --config "$CONFIG"
