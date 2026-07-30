#!/usr/bin/env bash
#SBATCH --job-name=ntv3-gene
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ntv3_gene_%A_%a.out
#SBATCH --error=logs/ntv3_gene_%A_%a.err
set -euo pipefail

: "${MANIFEST:?set MANIFEST to the Stage-T gene manifest parquet}"
: "${FASTA:?set FASTA to hg38 multi-FASTA}"
: "${CHECKPOINT:?set CHECKPOINT to the frozen NTv3 checkpoint}"
: "${OUTPUT_DIR:?set OUTPUT_DIR for shard NPZ files}"
: "${NUM_SHARDS:?set NUM_SHARDS equal to the SLURM array size}"

mkdir -p "$OUTPUT_DIR" logs
python -m methylation_predictor.rna_branch.extract_ntv3_gene_embeddings \
  --manifest "$MANIFEST" \
  --fasta "$FASTA" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT_DIR/gene_embeddings_shard_${SLURM_ARRAY_TASK_ID}.npz" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$NUM_SHARDS" \
  --length 32768 \
  --pool-radius 128 \
  --batch-size "${BATCH_SIZE:-1}" \
  --device cuda \
  --bf16
