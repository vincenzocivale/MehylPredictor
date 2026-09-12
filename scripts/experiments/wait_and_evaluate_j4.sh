#!/bin/bash
# Waits for run_efficient_single_attn_ablation.sh's full 80-epoch run to
# finish, then evaluates the resulting checkpoint on all three TRUE
# official MethylProphet chr1 views in one call
# (evaluate_official_split: train_cpg_x_val_sample, val_cpg_x_train_sample,
# val_cpg_x_val_sample -- see rna_training/locus_cls_trainer.py).
# Meant to be started via nohup, independent of the training orchestrator
# process, so it runs even if the interactive session that launched
# training has ended.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="efficient-single-attn-residual-ffn-final-chr1-seed17"
RUN_DIR="/home/vcivale/MethylPredictorData/experiments/runs/runs/locus_cls_joint/chr1/${RUN_ID}"
ORCH_LOG="$ROOT/logs/functional_concat_final/efficient_single_attn_orchestrator.log"
LOGDIR="$ROOT/logs/functional_concat_final"
mkdir -p "$LOGDIR"

echo "[eval] waiting for training to finish (watching $ORCH_LOG)..."
while ! grep -q "efficient_single_attn_residual_ffn finished successfully" "$ORCH_LOG" 2>/dev/null; do
  if grep -q "FAILED" "$ORCH_LOG" 2>/dev/null; then
    echo "[eval] training FAILED per orchestrator log; not evaluating." | tee -a "$LOGDIR/eval_j4_orchestrator.log"
    exit 1
  fi
  sleep 30
done
echo "[eval] training finished. Evaluating checkpoint..." | tee -a "$LOGDIR/eval_j4_orchestrator.log"

cd "$ROOT"
python scripts/evaluate.py --model rna_methylation --engine matched_chr1_shared_backbone \
  --checkpoint "$RUN_DIR/checkpoints/last.pt" --eval-scope chr1 \
  --recipe configs/models/functional_fusion/j4_efficient_single_attn.yaml \
  --canonical-root /home/vcivale/MethylPredictorData/datasets/methylprophet_repro_v1 \
  --feature-cache /home/vcivale/MethylPredictorData/derived/methylprophet_table5_tcga_chr1/features \
  --rna-cache /home/vcivale/MethylPredictorData/derived/methylprophet_table5_tcga_chr1/rna \
  --registry /home/vcivale/MethylPredictorData/datasets/methylprophet_repro_v1/cpg/registries/array_cpg_map.parquet \
  --prepared-root /home/vcivale/MethylPredictorData/derived/methylprophet_table5_tcga_chr1 \
  --cpg-targets-dir /home/vcivale/MethylPredictorData/derived/cpg_statistics/chr1 \
  --functional-atlas /home/vcivale/MethylPredictorData/derived/ntv3_functional_peak_atlas_chr1_all_sources \
  --annotation-cache /home/vcivale/MethylPredictorData/derived/ntv3_probe_targets/chr1_annotation_features_all_sources \
  --functional-only \
  --output "$RUN_DIR/evaluation/chr1/official_split.json" \
  > "$LOGDIR/eval_${RUN_ID}.log" 2>&1

echo "[eval] evaluation finished, see $RUN_DIR/evaluation/chr1/official_split.json" | tee -a "$LOGDIR/eval_j4_orchestrator.log"
