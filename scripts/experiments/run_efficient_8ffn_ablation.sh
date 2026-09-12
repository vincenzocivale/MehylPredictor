#!/bin/bash
# Efficient single-attention, 8-FFN candidate (2026-09-12): follow-up to
# ablation_depth8_residual (J5) outperforming J1 (depth 4) -- tests
# whether 1x cross-attention + 8 cheap FFN blocks recovers most of J5's
# gain without paying for cross-attention 8x. See
# src/methylation_predictor/modeling/ablation.py's module docstring
# (EfficientSingleAttentionPredictor / J7).
#
# This GPU is shared with other users and (at launch time) another one of
# our own runs (J6), so this script POLLS for enough free VRAM before
# attempting anything, instead of assuming it's available -- see the
# MIN_FREE_GB threshold below. Self-activates conda (see
# run_gated_residual_ablation.sh's comment on why this matters). Meant to
# be started via nohup so it survives SSH disconnection:
#
#   nohup bash scripts/experiments/run_efficient_8ffn_ablation.sh \
#     > logs/functional_concat_final/efficient_8ffn_orchestrator_stdout.log 2>&1 &
#   disown
set -euo pipefail
source /home/vcivale/miniconda3/etc/profile.d/conda.sh
conda activate methyl-predictor

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"
LOGDIR="$ROOT/logs/functional_concat_final"
mkdir -p "$LOGDIR"

# J4 (n_ffn_blocks=4) peaked at 16.5GB; J7 adds 4 more cheap FFN blocks, so
# budget generously above that plus headroom for other processes on this
# shared GPU to fluctuate without starving anyone.
MIN_FREE_GB="${MIN_FREE_GB:-22}"
echo "[efficient8ffn] waiting for >= ${MIN_FREE_GB}GB free VRAM (shared GPU)..." \
  | tee -a "$LOGDIR/efficient_8ffn_orchestrator.log"
while true; do
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  free_gb=$(( free_mb / 1024 ))
  if [ "$free_gb" -ge "$MIN_FREE_GB" ]; then
    echo "[efficient8ffn] ${free_gb}GB free, proceeding." | tee -a "$LOGDIR/efficient_8ffn_orchestrator.log"
    break
  fi
  sleep 60
done

DATA_ARGS=(
  --canonical-root /home/vcivale/MethylPredictorData/datasets/methylprophet_repro_v1
  --prior-cache /home/vcivale/MethylPredictorData/derived/methylprophet_table5_tcga_chr1/features
  --rna-cache /home/vcivale/MethylPredictorData/derived/methylprophet_table5_tcga_chr1/rna
  --registry /home/vcivale/MethylPredictorData/datasets/methylprophet_repro_v1/cpg/registries/array_cpg_map.parquet
  --prepared-root /home/vcivale/MethylPredictorData/derived/methylprophet_table5_tcga_chr1
  --cpg-targets-dir /home/vcivale/MethylPredictorData/derived/cpg_statistics/chr1
  --functional-atlas /home/vcivale/MethylPredictorData/derived/ntv3_functional_peak_atlas_chr1_all_sources
  --annotation-cache /home/vcivale/MethylPredictorData/derived/ntv3_probe_targets/chr1_annotation_features_all_sources
  --functional-only
)
OUTROOT="/home/vcivale/MethylPredictorData/experiments/runs"
RUN_ID="efficient-single-attn-8ffn-final-chr1-seed17"
RECIPE="configs/models/functional_fusion/j7_efficient_single_attn_8ffn.yaml"

cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[efficient8ffn] 1-epoch mode=final smoke test..." | tee -a "$LOGDIR/efficient_8ffn_orchestrator.log"
rm -rf "/tmp/${RUN_ID}_smoke"
if ! "$PY" scripts/train.py --model rna_methylation --scope chr1 --engine matched_chr1_shared_backbone \
    --mode final --recipe "$RECIPE" \
    --output-root "/tmp/${RUN_ID}_smoke" --run-id smoke-1ep --epochs 1 --seed 17 \
    "${DATA_ARGS[@]}" > "$LOGDIR/smoke_${RUN_ID}.log" 2>&1; then
  echo "[efficient8ffn] smoke test FAILED; see $LOGDIR/smoke_${RUN_ID}.log; STOPPING." \
    | tee -a "$LOGDIR/efficient_8ffn_orchestrator.log"
  exit 1
fi
rm -rf "/tmp/${RUN_ID}_smoke"
echo "[efficient8ffn] smoke test passed. Launching the full 80-epoch run..." \
  | tee -a "$LOGDIR/efficient_8ffn_orchestrator.log"

rm -rf "$OUTROOT/runs/locus_cls_joint/chr1/${RUN_ID}"  # fresh run, not a resume
if ! "$PY" scripts/train.py --model rna_methylation --scope chr1 --engine matched_chr1_shared_backbone \
    --mode final --recipe "$RECIPE" \
    --output-root "$OUTROOT" --run-id "$RUN_ID" --seed 17 \
    "${DATA_ARGS[@]}" > "$LOGDIR/${RUN_ID}.log" 2>&1; then
  echo "[efficient8ffn] efficient_single_attn_8ffn_residual FAILED (see $LOGDIR/${RUN_ID}.log)." \
    | tee -a "$LOGDIR/efficient_8ffn_orchestrator.log"
  exit 1
fi
echo "[efficient8ffn] efficient_single_attn_8ffn_residual finished successfully." \
  | tee -a "$LOGDIR/efficient_8ffn_orchestrator.log"
