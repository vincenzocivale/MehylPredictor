#!/bin/bash
# FFN-fusion ladder, concat cell only (2026-09-12) -- run on THIS host
# (godzilla) in parallel with ekko's sequential ladder
# (run_ffn_fusion_ladder.sh: two_stream_residual -> film -> concat), since
# concat is last in ekko's queue. Running it here concurrently gets all
# three fusion-mechanism results roughly in parallel instead of fully
# serial on one host -- see chat: "trovare il prima possibile il
# meccanismo di combinazione finale migliore".
#
# Same cost class as J4 (full J0-J6 batching, no VRAM pre-check needed,
# unlike J7/J8's 8/16-FFN cells) -- see
# configs/models/functional_fusion/j9a_ffn_fusion_concat.yaml.
# godzilla-specific paths (ekko's own run_ffn_fusion_ladder.sh uses
# /home/vcivale/dune_data/... which doesn't exist here).
set -euo pipefail
source /home/vcivale/miniconda3/etc/profile.d/conda.sh
conda activate methyl-predictor

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
RUN_ID="ffn-fusion-concat-final-chr1-seed17"
RECIPE="configs/models/functional_fusion/j9a_ffn_fusion_concat.yaml"

cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[ffnfusion-concat] 1-epoch mode=final smoke test..." | tee -a "$LOGDIR/ffn_fusion_concat_godzilla_orchestrator.log"
rm -rf "/tmp/${RUN_ID}_smoke"
if ! "$PY" scripts/train.py --model rna_methylation --scope chr1  \
    --mode final --recipe "$RECIPE" \
    --output-root "/tmp/${RUN_ID}_smoke" --run-id smoke-1ep --epochs 1 --seed 17 \
    "${DATA_ARGS[@]}" > "$LOGDIR/smoke_${RUN_ID}.log" 2>&1; then
  echo "[ffnfusion-concat] smoke test FAILED; see $LOGDIR/smoke_${RUN_ID}.log; STOPPING." \
    | tee -a "$LOGDIR/ffn_fusion_concat_godzilla_orchestrator.log"
  exit 1
fi
rm -rf "/tmp/${RUN_ID}_smoke"
echo "[ffnfusion-concat] smoke test passed. Launching the full 80-epoch run..." \
  | tee -a "$LOGDIR/ffn_fusion_concat_godzilla_orchestrator.log"

rm -rf "$OUTROOT/runs/locus_cls_joint/chr1/${RUN_ID}"  # fresh run, not a resume
if ! "$PY" scripts/train.py --model rna_methylation --scope chr1  \
    --mode final --recipe "$RECIPE" \
    --output-root "$OUTROOT" --run-id "$RUN_ID" --seed 17 \
    "${DATA_ARGS[@]}" > "$LOGDIR/${RUN_ID}.log" 2>&1; then
  echo "[ffnfusion-concat] ffn_fusion_concat FAILED (see $LOGDIR/${RUN_ID}.log)." \
    | tee -a "$LOGDIR/ffn_fusion_concat_godzilla_orchestrator.log"
  exit 1
fi
echo "[ffnfusion-concat] ffn_fusion_concat finished successfully." \
  | tee -a "$LOGDIR/ffn_fusion_concat_godzilla_orchestrator.log"

echo "[ffnfusion-concat] evaluating checkpoint on all three official views..." \
  | tee -a "$LOGDIR/ffn_fusion_concat_godzilla_orchestrator.log"
RUN_DIR="$OUTROOT/runs/locus_cls_joint/chr1/${RUN_ID}"
"$PY" scripts/evaluate.py --model rna_methylation  \
  --checkpoint "$RUN_DIR/checkpoints/last.pt" --eval-scope chr1 --recipe "$RECIPE" \
  --output "$RUN_DIR/evaluation/chr1/official_split.json" \
  "${DATA_ARGS[@]}" > "$LOGDIR/eval_${RUN_ID}.log" 2>&1 \
  && echo "[ffnfusion-concat] evaluation finished, see $RUN_DIR/evaluation/chr1/official_split.json" \
       | tee -a "$LOGDIR/ffn_fusion_concat_godzilla_orchestrator.log" \
  || echo "[ffnfusion-concat] evaluation FAILED, see $LOGDIR/eval_${RUN_ID}.log" \
       | tee -a "$LOGDIR/ffn_fusion_concat_godzilla_orchestrator.log"
