#!/bin/bash
# Depth-vs-residual ablation (2026-09-11): disentangles J0 (j0_final.yaml,
# depth=1/no-residual) vs J1 (j1_iterative.yaml, depth=4/residual) along
# their two confounded architectural axes -- see
# src/methylation_predictor/modeling/ablation.py's module docstring.
#
# Runs TWO cells sequentially on the SAME TRUE chr1 mode=final protocol as
# J0/J1 (seed=17, 80 epochs, identical loss/optimizer/batching/data pools):
#
#   Cell A: ablation_depth1_residual    (j2_ablation_depth1_residual.yaml)
#     Same depth as J0, with J1's per-block residual added.
#   Cell B: ablation_depth4_noresidual  (j3_ablation_depth4_noresidual.yaml)
#     Same depth as J1, with every block's attention output REPLACING the
#     running state instead of being added to it (no residual).
#
# Each cell gets a 1-epoch mode=final GPU smoke test first; the next step
# (cell B, then the run finishing) only starts if the previous one succeeded.
# Meant to be started via nohup so it survives SSH disconnection:
#
#   nohup bash scripts/experiments/run_depth_residual_ablation.sh \
#     > logs/functional_concat_final/ablation_orchestrator_stdout.log 2>&1 &
#   disown
#
# Adjust DATA_ARGS below if this machine mounts the canonical bundle/derived
# caches at different paths than /dune/DATASETS/MethylPredictionData/...
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"
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
  
)
OUTROOT="/dune/DATASETS/MethylPredictionData/experiments/runs"

cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_cell () {
  local recipe="$1" run_id="$2" label="$3"
  echo "[ablation] $label: 1-epoch mode=final smoke test..." | tee -a "$LOGDIR/ablation_orchestrator.log"
  rm -rf "/tmp/${run_id}_smoke"
  if ! "$PY" scripts/train.py --model rna_methylation --scope chr1  \
      --mode final --recipe "$recipe" \
      --output-root "/tmp/${run_id}_smoke" --run-id smoke-1ep --epochs 1 --seed 17 \
      "${DATA_ARGS[@]}" > "$LOGDIR/smoke_${run_id}.log" 2>&1; then
    echo "[ablation] $label smoke test FAILED; see $LOGDIR/smoke_${run_id}.log; STOPPING." \
      | tee -a "$LOGDIR/ablation_orchestrator.log"
    exit 1
  fi
  rm -rf "/tmp/${run_id}_smoke"
  echo "[ablation] $label smoke test passed. Launching the full 80-epoch run..." \
    | tee -a "$LOGDIR/ablation_orchestrator.log"

  rm -rf "$OUTROOT/runs/locus_cls_joint/chr1/${run_id}"  # fresh run, not a resume
  if ! "$PY" scripts/train.py --model rna_methylation --scope chr1  \
      --mode final --recipe "$recipe" \
      --output-root "$OUTROOT" --run-id "$run_id" --seed 17 \
      "${DATA_ARGS[@]}" > "$LOGDIR/${run_id}.log" 2>&1; then
    echo "[ablation] $label FAILED (see $LOGDIR/${run_id}.log); STOPPING, not launching the next cell." \
      | tee -a "$LOGDIR/ablation_orchestrator.log"
    exit 1
  fi
  echo "[ablation] $label finished successfully." | tee -a "$LOGDIR/ablation_orchestrator.log"
}

run_cell configs/models/functional_fusion/j2_ablation_depth1_residual.yaml \
  ablation-depth1-residual-final-chr1-seed17 "Cell A (depth=1, residual)"

run_cell configs/models/functional_fusion/j3_ablation_depth4_noresidual.yaml \
  ablation-depth4-noresidual-final-chr1-seed17 "Cell B (depth=4, no residual)"

echo "[ablation] both cells finished successfully." | tee -a "$LOGDIR/ablation_orchestrator.log"
