#!/bin/bash
# Waits for the queued run-1 (simple single-retrieval candidate, PID $1) to
# exit -- so the two full mode=final candidates never train on the GPU at
# once -- then, only if run 1's own metrics.json reports success, runs a
# 1-epoch mode=final GPU smoke test of the iterative-retrieval candidate
# (j1_iterative.yaml) and, only on that success, launches its full 80-epoch
# run. Started via nohup so it (and the run it launches) survive SSH
# disconnection, mirroring wait_and_run_mas_concat_sweep.sh's pattern.
set -euo pipefail
RUN1_PID="$1"
RUN1_DIR="$2"   # run-1's own run directory, to check metadata.json/last checkpoint for success
ROOT="/home/vcivale/MehylPredictor"
PY="/home/vcivale/miniconda/envs/methyl-predictor/bin/python"
LOGDIR="$ROOT/logs/functional_concat_final"
mkdir -p "$LOGDIR"

DATA_ARGS=(
  --canonical-root /dune/DATASETS/MethylPredictionData/datasets/methylprophet_repro_v1
  --prior-cache /dune/DATASETS/MethylPredictionData/derived/methylprophet_table5_tcga_chr1/features
  --rna-cache /dune/DATASETS/MethylPredictionData/derived/methylprophet_table5_tcga_chr1/rna
  --registry /dune/DATASETS/MethylPredictionData/datasets/methylprophet_repro_v1/cpg/registries/array_cpg_map.parquet
  --prepared-root /dune/DATASETS/MethylPredictionData/derived/methylprophet_table5_tcga_chr1
  --cpg-targets-dir /dune/DATASETS/MethylPredictionData/derived/cpg_statistics/chr1
  --functional-atlas /dune/DATASETS/MethylPredictionData/derived/ntv3_functional_peak_atlas_chr1_all_sources
  --annotation-cache /dune/DATASETS/MethylPredictionData/derived/ntv3_probe_targets/chr1_annotation_features_all_sources
  --functional-only
)

echo "[wait-and-run] waiting for run-1 PID $RUN1_PID to exit..." | tee -a "$LOGDIR/orchestrator.log"
while kill -0 "$RUN1_PID" 2>/dev/null; do
  sleep 30
done
echo "[wait-and-run] run-1 PID $RUN1_PID has exited." | tee -a "$LOGDIR/orchestrator.log"

cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HISTORY="$RUN1_DIR/training/history.json"
LAST_CKPT="$RUN1_DIR/checkpoints/last.pt"
RUN1_OK=0
if [ -f "$HISTORY" ] && [ -f "$LAST_CKPT" ]; then
  EPOCHS_RUN="$("$PY" -c "import json,sys; h=json.load(open(sys.argv[1])); print(len(h))" "$HISTORY" 2>/dev/null || echo 0)"
  if [ "${EPOCHS_RUN:-0}" -ge 80 ]; then
    RUN1_OK=1
  fi
fi
if [ "$RUN1_OK" -ne 1 ]; then
  echo "[wait-and-run] run-1 did not reach 80 recorded epochs (history has ${EPOCHS_RUN:-0}) at $HISTORY -- run 1 FAILED; NOT launching run 2." \
    | tee -a "$LOGDIR/orchestrator.log"
  exit 1
fi
echo "[wait-and-run] run-1 succeeded (${EPOCHS_RUN} epochs recorded, checkpoint present at $LAST_CKPT). Running j1 1-epoch mode=final GPU smoke test..." \
  | tee -a "$LOGDIR/orchestrator.log"

SMOKE_OK=0
rm -rf /tmp/j1_iterative_smoke
"$PY" scripts/train.py --model rna_methylation --scope chr1 --engine matched_chr1_shared_backbone \
  --mode final --recipe configs/models/functional_fusion/j1_iterative.yaml \
  --output-root /tmp/j1_iterative_smoke --run-id j1-smoke-1ep --epochs 1 --seed 17 \
  "${DATA_ARGS[@]}" > "$LOGDIR/smoke_j1.log" 2>&1 && SMOKE_OK=1 || SMOKE_OK=0

if [ "$SMOKE_OK" -ne 1 ]; then
  echo "[wait-and-run] j1 smoke test FAILED; see $LOGDIR/smoke_j1.log; NOT launching run 2." \
    | tee -a "$LOGDIR/orchestrator.log"
  exit 1
fi
rm -rf /tmp/j1_iterative_smoke
echo "[wait-and-run] j1 smoke test passed. Launching the full 80-epoch iterative-retrieval run..." \
  | tee -a "$LOGDIR/orchestrator.log"

RUN2_ID="functional-iterative-rna2-final-chr1-seed17"
OUTROOT="/dune/DATASETS/MethylPredictionData/experiments/runs"
"$PY" scripts/train.py --model rna_methylation --scope chr1 --engine matched_chr1_shared_backbone \
  --mode final --recipe configs/models/functional_fusion/j1_iterative.yaml \
  --output-root "$OUTROOT" --run-id "$RUN2_ID" --seed 17 \
  "${DATA_ARGS[@]}" > "$LOGDIR/${RUN2_ID}.log" 2>&1
RUN2_EXIT=$?
echo "[wait-and-run] run 2 ($RUN2_ID) exited with code $RUN2_EXIT." | tee -a "$LOGDIR/orchestrator.log"
exit "$RUN2_EXIT"
