#!/bin/bash
# Waits for run_gated_residual_ablation.sh's full 80-epoch run to finish,
# then evaluates the resulting checkpoint on all three TRUE official
# MethylProphet chr1 views in one call (evaluate_rna_checkpoint).
# Self-activates conda (see run_gated_residual_ablation.sh's comment on
# why -- a previous companion script forgot this and silently failed with
# "python: command not found" right after training completed).
set -euo pipefail
source /home/vcivale/miniconda3/etc/profile.d/conda.sh
conda activate methyl-predictor

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="ablation-depth1-gated-residual-final-chr1-seed17"
RUN_DIR="/home/vcivale/MethylPredictorData/experiments/runs/runs/locus_cls_joint/chr1/${RUN_ID}"
ORCH_LOG="$ROOT/logs/functional_concat_final/gated_residual_orchestrator.log"
LOGDIR="$ROOT/logs/functional_concat_final"
mkdir -p "$LOGDIR"

echo "[eval] waiting for training to finish (watching $ORCH_LOG)..."
while ! grep -q "ablation_depth1_gated_residual finished successfully" "$ORCH_LOG" 2>/dev/null; do
  if grep -q "FAILED" "$ORCH_LOG" 2>/dev/null; then
    echo "[eval] training FAILED per orchestrator log; not evaluating." | tee -a "$LOGDIR/eval_j6_orchestrator.log"
    exit 1
  fi
  sleep 30
done
echo "[eval] training finished. Evaluating checkpoint..." | tee -a "$LOGDIR/eval_j6_orchestrator.log"

cd "$ROOT"
python scripts/evaluate.py --model rna_methylation  \
  --checkpoint "$RUN_DIR/checkpoints/last.pt" --eval-scope chr1 \
  --recipe configs/models/functional_fusion/j6_gated_residual.yaml \
  --canonical-root /home/vcivale/MethylPredictorData/datasets/methylprophet_repro_v1 \
  --prior-cache /home/vcivale/MethylPredictorData/derived/methylprophet_table5_tcga_chr1/features \
  --rna-cache /home/vcivale/MethylPredictorData/derived/methylprophet_table5_tcga_chr1/rna \
  --registry /home/vcivale/MethylPredictorData/datasets/methylprophet_repro_v1/cpg/registries/array_cpg_map.parquet \
  --prepared-root /home/vcivale/MethylPredictorData/derived/methylprophet_table5_tcga_chr1 \
  --cpg-targets-dir /home/vcivale/MethylPredictorData/derived/cpg_statistics/chr1 \
  --functional-atlas /home/vcivale/MethylPredictorData/derived/ntv3_functional_peak_atlas_chr1_all_sources \
  --annotation-cache /home/vcivale/MethylPredictorData/derived/ntv3_probe_targets/chr1_annotation_features_all_sources \
  --output "$RUN_DIR/evaluation/chr1/official_split.json" \
  > "$LOGDIR/eval_${RUN_ID}.log" 2>&1

echo "[eval] evaluation finished, see $RUN_DIR/evaluation/chr1/official_split.json" | tee -a "$LOGDIR/eval_j6_orchestrator.log"
