#!/bin/bash
# Efficient single-attention, 16-FFN candidate (2026-09-12): pushes the
# efficient-depth question further than J7 (8 FFN blocks) -- does the
# benefit keep scaling with FFN depth, or plateau? See
# src/methylation_predictor/modeling/ablation.py's module docstring
# (EfficientSingleAttentionPredictor / J8) and
# configs/models/functional_fusion/j8_efficient_single_attn_16ffn.yaml's
# comments for the full rationale, including why the batch was cut
# pre-emptively (array sample_size 160, vs J7's 320).
#
# This GPU is shared, so this script POLLS for enough free VRAM before
# attempting anything (see MIN_FREE_GB; override via env var like
# `MIN_FREE_GB=15 bash ...` if you want to accept more OOM risk to start
# sooner -- see J7's own history of doing exactly that in the chat log).
# Self-activates conda. Meant to be started via nohup:
#
#   nohup bash scripts/experiments/run_efficient_16ffn_ablation.sh \
#     > logs/functional_concat_final/efficient_16ffn_orchestrator_stdout.log 2>&1 &
#   disown
#
# NOT launched automatically when this file was added -- deliberately
# queued behind J7's result (see chat: the user agreed to wait and see if
# J7 recovers J5's depth-8 gain before committing GPU time to depth-16).
set -euo pipefail
source /home/vcivale/miniconda3/etc/profile.d/conda.sh
conda activate methyl-predictor

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"
LOGDIR="$ROOT/logs/functional_concat_final"
mkdir -p "$LOGDIR"

# J7 (n_ffn_blocks=8, array batch 320) peaked at 14.13GB. J8 doubles the
# FFN stack again at a smaller batch (160) -- budget generously since the
# 4->8 step wasn't a clean linear reference (batch changed at the same
# time); this is a conservative starting bar, not a measured number.
MIN_FREE_GB="${MIN_FREE_GB:-20}"
echo "[efficient16ffn] waiting for >= ${MIN_FREE_GB}GB free VRAM (shared GPU)..." \
  | tee -a "$LOGDIR/efficient_16ffn_orchestrator.log"
while true; do
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  free_gb=$(( free_mb / 1024 ))
  if [ "$free_gb" -ge "$MIN_FREE_GB" ]; then
    echo "[efficient16ffn] ${free_gb}GB free, proceeding." | tee -a "$LOGDIR/efficient_16ffn_orchestrator.log"
    break
  fi
  sleep 60
done

DATA_ARGS=(
  --canonical-root /home/vcivale/MethylPredictorData/datasets/methylprophet_repro_v1
  --feature-cache /home/vcivale/MethylPredictorData/derived/methylprophet_table5_tcga_chr1/features
  --rna-cache /home/vcivale/MethylPredictorData/derived/methylprophet_table5_tcga_chr1/rna
  --registry /home/vcivale/MethylPredictorData/datasets/methylprophet_repro_v1/cpg/registries/array_cpg_map.parquet
  --prepared-root /home/vcivale/MethylPredictorData/derived/methylprophet_table5_tcga_chr1
  --cpg-targets-dir /home/vcivale/MethylPredictorData/derived/cpg_statistics/chr1
  --functional-atlas /home/vcivale/MethylPredictorData/derived/ntv3_functional_peak_atlas_chr1_all_sources
  --annotation-cache /home/vcivale/MethylPredictorData/derived/ntv3_probe_targets/chr1_annotation_features_all_sources
  --functional-only
)
OUTROOT="/home/vcivale/MethylPredictorData/experiments/runs"
RUN_ID="efficient-single-attn-16ffn-final-chr1-seed17"
RECIPE="configs/models/functional_fusion/j8_efficient_single_attn_16ffn.yaml"

cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[efficient16ffn] 1-epoch mode=final smoke test..." | tee -a "$LOGDIR/efficient_16ffn_orchestrator.log"
rm -rf "/tmp/${RUN_ID}_smoke"
if ! "$PY" scripts/train.py --model rna_methylation --scope chr1 --engine matched_chr1_shared_backbone \
    --mode final --recipe "$RECIPE" \
    --output-root "/tmp/${RUN_ID}_smoke" --run-id smoke-1ep --epochs 1 --seed 17 \
    "${DATA_ARGS[@]}" > "$LOGDIR/smoke_${RUN_ID}.log" 2>&1; then
  echo "[efficient16ffn] smoke test FAILED; see $LOGDIR/smoke_${RUN_ID}.log; STOPPING." \
    | tee -a "$LOGDIR/efficient_16ffn_orchestrator.log"
  exit 1
fi
rm -rf "/tmp/${RUN_ID}_smoke"
echo "[efficient16ffn] smoke test passed. Launching the full 80-epoch run..." \
  | tee -a "$LOGDIR/efficient_16ffn_orchestrator.log"

rm -rf "$OUTROOT/runs/locus_cls_joint/chr1/${RUN_ID}"  # fresh run, not a resume
if ! "$PY" scripts/train.py --model rna_methylation --scope chr1 --engine matched_chr1_shared_backbone \
    --mode final --recipe "$RECIPE" \
    --output-root "$OUTROOT" --run-id "$RUN_ID" --seed 17 \
    "${DATA_ARGS[@]}" > "$LOGDIR/${RUN_ID}.log" 2>&1; then
  echo "[efficient16ffn] efficient_single_attn_16ffn_residual FAILED (see $LOGDIR/${RUN_ID}.log)." \
    | tee -a "$LOGDIR/efficient_16ffn_orchestrator.log"
  exit 1
fi
echo "[efficient16ffn] efficient_single_attn_16ffn_residual finished successfully." \
  | tee -a "$LOGDIR/efficient_16ffn_orchestrator.log"
