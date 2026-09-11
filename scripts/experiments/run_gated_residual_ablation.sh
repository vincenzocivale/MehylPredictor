#!/bin/bash
# Flamingo-style gated residual (2026-09-11): follow-up to
# ablation_depth1_residual (J2) -- same depth (n_blocks=1), but the
# per-block residual add is a learned scalar gate (tanh(alpha), alpha
# init 0) instead of unconditional. See
# src/methylation_predictor/modeling/ablation.py's module docstring
# (GatedResidualPredictor / J6).
#
# Runs ONE cell on the SAME TRUE chr1 mode=final protocol as J0-J5
# (seed=17, 80 epochs, identical loss/optimizer/batching/data pools):
#
#   ablation_depth1_gated_residual (j6_gated_residual.yaml)
#
# Runs a 1-epoch mode=final GPU smoke test first; the full run only starts
# if that succeeds. Self-activates the methyl-predictor conda env (a
# previous companion script forgot this and failed with "python: command
# not found" after training finished -- don't repeat that). Meant to be
# started via nohup so it survives SSH disconnection:
#
#   nohup bash scripts/experiments/run_gated_residual_ablation.sh \
#     > logs/functional_concat_final/gated_residual_orchestrator_stdout.log 2>&1 &
#   disown
set -euo pipefail
source /home/vcivale/miniconda3/etc/profile.d/conda.sh
conda activate methyl-predictor

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"
LOGDIR="$ROOT/logs/functional_concat_final"
mkdir -p "$LOGDIR"

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
RUN_ID="ablation-depth1-gated-residual-final-chr1-seed17"
RECIPE="configs/models/functional_fusion/j6_gated_residual.yaml"

cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[gated] ablation_depth1_gated_residual: 1-epoch mode=final smoke test..." | tee -a "$LOGDIR/gated_residual_orchestrator.log"
rm -rf "/tmp/${RUN_ID}_smoke"
if ! "$PY" scripts/train.py --model rna_methylation --scope chr1 --engine matched_chr1_shared_backbone \
    --mode final --recipe "$RECIPE" \
    --output-root "/tmp/${RUN_ID}_smoke" --run-id smoke-1ep --epochs 1 --seed 17 \
    "${DATA_ARGS[@]}" > "$LOGDIR/smoke_${RUN_ID}.log" 2>&1; then
  echo "[gated] smoke test FAILED; see $LOGDIR/smoke_${RUN_ID}.log; STOPPING." \
    | tee -a "$LOGDIR/gated_residual_orchestrator.log"
  exit 1
fi
rm -rf "/tmp/${RUN_ID}_smoke"
echo "[gated] smoke test passed. Launching the full 80-epoch run..." \
  | tee -a "$LOGDIR/gated_residual_orchestrator.log"

rm -rf "$OUTROOT/runs/locus_cls_joint/chr1/${RUN_ID}"  # fresh run, not a resume
if ! "$PY" scripts/train.py --model rna_methylation --scope chr1 --engine matched_chr1_shared_backbone \
    --mode final --recipe "$RECIPE" \
    --output-root "$OUTROOT" --run-id "$RUN_ID" --seed 17 \
    "${DATA_ARGS[@]}" > "$LOGDIR/${RUN_ID}.log" 2>&1; then
  echo "[gated] ablation_depth1_gated_residual FAILED (see $LOGDIR/${RUN_ID}.log)." \
    | tee -a "$LOGDIR/gated_residual_orchestrator.log"
  exit 1
fi
echo "[gated] ablation_depth1_gated_residual finished successfully." | tee -a "$LOGDIR/gated_residual_orchestrator.log"
