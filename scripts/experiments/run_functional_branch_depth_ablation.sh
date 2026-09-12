#!/bin/bash
# Functional-branch depth candidate (2026-09-12, J9b): follow-up to J7/J8
# showing FFN depth helps the RNA-conditioned retrieval branch -- tests
# whether the SAME kind of depth also helps the functional-annotation
# branch (h_c, from track_embedding+dense_encoder), held fixed at J7's
# retrieval depth (8 FFN blocks) so a gain is attributable to the
# functional branch alone. deep_query stays False here -- the cross-
# attention query is unchanged from J7. See
# src/methylation_predictor/modeling/ablation.py's module docstring
# (EfficientSingleAttentionPredictor / J9b) and
# configs/models/functional_fusion/j9b_functional_branch_depth.yaml.
#
# This GPU is shared with other users, so this script POLLS for enough free
# VRAM before attempting anything, instead of assuming it's available -- see
# the MIN_FREE_GB threshold below. Self-activates conda. Meant to be started
# via nohup so it survives SSH disconnection:
#
#   nohup bash scripts/experiments/run_functional_branch_depth_ablation.sh \
#     > logs/functional_concat_final/j9b_orchestrator_stdout.log 2>&1 &
#   disown
set -euo pipefail
source /home/vcivale/miniconda/etc/profile.d/conda.sh
conda activate methyl-predictor

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"
LOGDIR="$ROOT/logs/functional_concat_final"
mkdir -p "$LOGDIR"

# J7 (n_ffn_blocks=8, n_functional_ffn_blocks=0) needed ~18GB at
# array sample_size=320; the 8 extra functional_ffn blocks operate on h_c
# ([n_loci, WIDTH], not replicated per RNA sample), so expected overhead is
# small, but this has not been VRAM-profiled yet -- budget generously above
# J7's own footprint plus headroom for other processes on this shared GPU.
MIN_FREE_GB="${MIN_FREE_GB:-24}"
echo "[j9b] waiting for >= ${MIN_FREE_GB}GB free VRAM (shared GPU)..." \
  | tee -a "$LOGDIR/j9b_orchestrator.log"
while true; do
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  free_gb=$(( free_mb / 1024 ))
  if [ "$free_gb" -ge "$MIN_FREE_GB" ]; then
    echo "[j9b] ${free_gb}GB free, proceeding." | tee -a "$LOGDIR/j9b_orchestrator.log"
    break
  fi
  sleep 60
done

DATAROOT="${DATAROOT:-/dune/DATASETS/MethylPredictionData}"
DATA_ARGS=(
  --canonical-root "$DATAROOT/datasets/methylprophet_repro_v1"
  --prior-cache "$DATAROOT/derived/methylprophet_table5_tcga_chr1/features"
  --rna-cache "$DATAROOT/derived/methylprophet_table5_tcga_chr1/rna"
  --registry "$DATAROOT/datasets/methylprophet_repro_v1/cpg/registries/array_cpg_map.parquet"
  --prepared-root "$DATAROOT/derived/methylprophet_table5_tcga_chr1"
  --cpg-targets-dir "$DATAROOT/derived/cpg_statistics/chr1"
  --functional-atlas "$DATAROOT/derived/ntv3_functional_peak_atlas_chr1_all_sources"
  --annotation-cache "$DATAROOT/derived/ntv3_probe_targets/chr1_annotation_features_all_sources"
  --functional-only
)
OUTROOT="$DATAROOT/experiments/runs"
RUN_ID="functional-branch-depth-8ffn-final-chr1-seed17"
RECIPE="configs/models/functional_fusion/j9b_functional_branch_depth.yaml"

cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[j9b] 1-epoch mode=final smoke test..." | tee -a "$LOGDIR/j9b_orchestrator.log"
rm -rf "/tmp/${RUN_ID}_smoke"
if ! "$PY" scripts/train.py --model rna_methylation --scope chr1 --engine matched_chr1_shared_backbone \
    --mode final --recipe "$RECIPE" \
    --output-root "/tmp/${RUN_ID}_smoke" --run-id smoke-1ep --epochs 1 --seed 17 \
    "${DATA_ARGS[@]}" > "$LOGDIR/smoke_${RUN_ID}.log" 2>&1; then
  echo "[j9b] smoke test FAILED; see $LOGDIR/smoke_${RUN_ID}.log; STOPPING." \
    | tee -a "$LOGDIR/j9b_orchestrator.log"
  exit 1
fi
rm -rf "/tmp/${RUN_ID}_smoke"
echo "[j9b] smoke test passed. Launching the full 80-epoch run..." \
  | tee -a "$LOGDIR/j9b_orchestrator.log"

rm -rf "$OUTROOT/runs/locus_cls_joint/chr1/${RUN_ID}"  # fresh run, not a resume
if ! "$PY" scripts/train.py --model rna_methylation --scope chr1 --engine matched_chr1_shared_backbone \
    --mode final --recipe "$RECIPE" \
    --output-root "$OUTROOT" --run-id "$RUN_ID" --seed 17 \
    "${DATA_ARGS[@]}" > "$LOGDIR/${RUN_ID}.log" 2>&1; then
  echo "[j9b] efficient_single_attn_8ffn_residual_functional8 FAILED (see $LOGDIR/${RUN_ID}.log)." \
    | tee -a "$LOGDIR/j9b_orchestrator.log"
  exit 1
fi
echo "[j9b] efficient_single_attn_8ffn_residual_functional8 finished successfully." \
  | tee -a "$LOGDIR/j9b_orchestrator.log"
