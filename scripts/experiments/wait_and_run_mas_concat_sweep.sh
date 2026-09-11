#!/bin/bash
# Waits for the currently-running G0-G4 regulatory-fusion ladder orchestrator
# (PID $1) to exit -- so the two never run on the GPU at once -- then runs a
# 1-epoch GPU smoke test of h0, and only on success launches the full H0/H1/H2
# sweep. Meant to be started once via nohup so it (and the sweep it launches)
# survive SSH disconnection.
set -euo pipefail
LADDER_PID="$1"
ROOT="/home/vcivale/MehylPredictor"
PY="/home/vcivale/miniconda/envs/methyl-predictor/bin/python"
LOGDIR="$ROOT/logs/mas_concat_sweep"
mkdir -p "$LOGDIR"

echo "[wait-and-run] waiting for ladder PID $LADDER_PID to exit..." | tee -a "$LOGDIR/wait.log"
while kill -0 "$LADDER_PID" 2>/dev/null; do
  sleep 30
done
echo "[wait-and-run] ladder PID $LADDER_PID has exited; running h0 1-epoch GPU smoke test..." | tee -a "$LOGDIR/wait.log"

cd "$ROOT"
SMOKE_OK=0
"$PY" scripts/train.py --model rna_methylation --scope chr1 --engine matched_chr1_shared_backbone \
  --mode development --recipe configs/models/functional_fusion/h0.yaml \
  --output-root /tmp/mas_concat_smoke --run-id h0-smoke-1ep --epochs 1 --seed 17 \
  --development-split-seed 17 \
  --canonical-root /dune/DATASETS/MethylPredictionData/datasets/methylprophet_repro_v1 \
  --feature-cache /dune/DATASETS/MethylPredictionData/derived/methylprophet_table5_tcga_chr1/features \
  --rna-cache /dune/DATASETS/MethylPredictionData/derived/methylprophet_table5_tcga_chr1/rna \
  --registry /dune/DATASETS/MethylPredictionData/datasets/methylprophet_repro_v1/cpg/registries/array_cpg_map.parquet \
  --prepared-root /dune/DATASETS/MethylPredictionData/derived/methylprophet_table5_tcga_chr1 \
  --cpg-targets-dir /dune/DATASETS/MethylPredictionData/derived/cpg_statistics/chr1 \
  --functional-atlas /dune/DATASETS/MethylPredictionData/derived/ntv3_functional_peak_atlas_chr1_all_sources \
  --annotation-cache /dune/DATASETS/MethylPredictionData/derived/ntv3_probe_targets/chr1_annotation_features_all_sources \
  --functional-only > "$LOGDIR/smoke_h0.log" 2>&1 && SMOKE_OK=1 || SMOKE_OK=0

if [ "$SMOKE_OK" -ne 1 ]; then
  echo "[wait-and-run] h0 smoke test FAILED; see $LOGDIR/smoke_h0.log; NOT launching the sweep." | tee -a "$LOGDIR/wait.log"
  exit 1
fi
rm -rf /tmp/mas_concat_smoke
echo "[wait-and-run] smoke test OK; launching H0->H1->H2 sweep." | tee -a "$LOGDIR/wait.log"

exec "$PY" scripts/experiments/run_mas_concat_sweep.py \
  --variants h0,h1,h2 --epochs 40 --seed 17 \
  --output-root /dune/DATASETS/MethylPredictionData/experiments \
  --canonical-root /dune/DATASETS/MethylPredictionData/datasets/methylprophet_repro_v1 \
  --feature-cache /dune/DATASETS/MethylPredictionData/derived/methylprophet_table5_tcga_chr1/features \
  --rna-cache /dune/DATASETS/MethylPredictionData/derived/methylprophet_table5_tcga_chr1/rna \
  --registry /dune/DATASETS/MethylPredictionData/datasets/methylprophet_repro_v1/cpg/registries/array_cpg_map.parquet \
  --prepared-root /dune/DATASETS/MethylPredictionData/derived/methylprophet_table5_tcga_chr1 \
  --cpg-targets-dir /dune/DATASETS/MethylPredictionData/derived/cpg_statistics/chr1 \
  > "$ROOT/logs/mas_concat_sweep.log" 2>&1
