#!/bin/bash
# Efficient single-attention candidate (2026-09-11): follow-up to the
# depth-vs-residual ablation (run_depth_residual_ablation.sh). If Cell A
# there (ablation_depth1_residual: same depth as J0, with J1's residual)
# shows the residual alone -- not the repeated cross-attention -- drives
# J1's gain, this tests whether that gain survives paying the expensive
# cross-attention op only ONCE instead of 4x: single locus<-RNA
# cross-attention + residual, then 4 FFN-only residual blocks (matching
# J1's total FFN depth). See
# src/methylation_predictor/modeling/ablation.py's module docstring
# (EfficientSingleAttentionPredictor / J4).
#
# Runs ONE cell on the SAME TRUE chr1 mode=final protocol as J0/J1/the
# depth-vs-residual ablation (seed=17, 80 epochs, identical
# loss/optimizer/batching/data pools):
#
#   efficient_single_attn_residual_ffn (j4_efficient_single_attn.yaml)
#
# Runs a 1-epoch mode=final GPU smoke test first; the full run only starts
# if that succeeds. Meant to be started via nohup so it survives SSH
# disconnection:
#
#   nohup bash scripts/experiments/run_efficient_single_attn_ablation.sh \
#     > logs/functional_concat_final/efficient_single_attn_orchestrator_stdout.log 2>&1 &
#   disown
#
# Adjust DATA_ARGS below if this machine mounts the canonical bundle/derived
# caches at different paths than /home/vcivale/MethylPredictorData/...
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"
LOGDIR="$ROOT/logs/functional_concat_final"
mkdir -p "$LOGDIR"

DATA_ARGS=(
  --canonical-root /home/vcivale/MethylPredictorData/datasets/methylprophet_repro_v1
  --prior-cache /home/vcivale/MethylPredictorData/derived/methylprophet_table5_tcga_chr1/features
  --rna-cache /home/vcivale/MethylPredictorData/derived/methylprophet_table5_tcga_chr1/rna
  --registry /home/vcivale/MethylPredictorData/datasets/methylprophet_repro_v1/cpg/registries/array_cpg_map.parquet
  --prepared-root /home/vcivale/MethylPredictorData/derived/methylprophet_table5_tcga_chr1
  --cpg-targets-dir /home/vcivale/MethylPredictorData/derived/cpg_statistics/chr1
  --functional-atlas /home/vcivale/MethylPredictorData/derived/ntv3_functional_peak_atlas_chr1_all_sources
  --annotation-cache /home/vcivale/MethylPredictorData/derived/ntv3_probe_targets/chr1_annotation_features_all_sources
  
)
OUTROOT="/home/vcivale/MethylPredictorData/experiments/runs"

cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_cell () {
  local recipe="$1" run_id="$2" label="$3"
  echo "[efficient] $label: 1-epoch mode=final smoke test..." | tee -a "$LOGDIR/efficient_single_attn_orchestrator.log"
  rm -rf "/tmp/${run_id}_smoke"
  if ! "$PY" scripts/train.py --model rna_methylation --scope chr1  \
      --mode final --recipe "$recipe" \
      --output-root "/tmp/${run_id}_smoke" --run-id smoke-1ep --epochs 1 --seed 17 \
      "${DATA_ARGS[@]}" > "$LOGDIR/smoke_${run_id}.log" 2>&1; then
    echo "[efficient] $label smoke test FAILED; see $LOGDIR/smoke_${run_id}.log; STOPPING." \
      | tee -a "$LOGDIR/efficient_single_attn_orchestrator.log"
    exit 1
  fi
  rm -rf "/tmp/${run_id}_smoke"
  echo "[efficient] $label smoke test passed. Launching the full 80-epoch run..." \
    | tee -a "$LOGDIR/efficient_single_attn_orchestrator.log"

  rm -rf "$OUTROOT/runs/locus_cls_joint/chr1/${run_id}"  # fresh run, not a resume
  if ! "$PY" scripts/train.py --model rna_methylation --scope chr1  \
      --mode final --recipe "$recipe" \
      --output-root "$OUTROOT" --run-id "$run_id" --seed 17 \
      "${DATA_ARGS[@]}" > "$LOGDIR/${run_id}.log" 2>&1; then
    echo "[efficient] $label FAILED (see $LOGDIR/${run_id}.log); STOPPING." \
      | tee -a "$LOGDIR/efficient_single_attn_orchestrator.log"
    exit 1
  fi
  echo "[efficient] $label finished successfully." | tee -a "$LOGDIR/efficient_single_attn_orchestrator.log"
}

run_cell configs/models/functional_fusion/j4_efficient_single_attn.yaml \
  efficient-single-attn-residual-ffn-final-chr1-seed17 "efficient_single_attn_residual_ffn"

echo "[efficient] cell finished successfully." | tee -a "$LOGDIR/efficient_single_attn_orchestrator.log"
