#!/bin/bash
# Symmetric-depth control (2026-09-12, J10): 4 FFN blocks on EACH branch
# (retrieval matching J4's depth, functional matching it too), a different
# question from J9b's asymmetric 8/8 (retrieval held at J7's depth) -- see
# src/methylation_predictor/modeling/ablation.py's module docstring
# (EfficientSingleAttentionPredictor / J10) and
# configs/models/functional_fusion/j10_symmetric_4ffn_both_branches.yaml.
#
# NOTE: J9b (functional-branch-depth-8ffn-final-chr1-seed17) is running
# concurrently on this same shared GPU/NFS. Both jobs compete for the same
# GPU compute AND the same NFS bandwidth (the actual bottleneck for this
# whole model family, per j4/j7's batching comments) -- launching this WILL
# slow J9b down too, not just share idle capacity. This is a deliberate
# trade-off (parallel throughput over either job's individual wall-clock),
# not a mistake.
#
# This GPU is shared with other users, so this script POLLS for enough free
# VRAM before attempting anything. Self-activates conda. Meant to be started
# via nohup so it survives SSH disconnection:
#
#   nohup bash scripts/experiments/run_symmetric_4ffn_ablation.sh \
#     > logs/functional_concat_final/j10_orchestrator_stdout.log 2>&1 &
#   disown
set -euo pipefail
source /home/vcivale/miniconda/etc/profile.d/conda.sh
conda activate methyl-predictor

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"
LOGDIR="$ROOT/logs/functional_concat_final"
mkdir -p "$LOGDIR"

# J4 (n_ffn_blocks=4, no functional depth) profiled at 16.5GB peak; retrieval
# depth here is the same 4, plus 4 cheap functional_ffn blocks on h_c (not
# replicated per RNA sample) -- expected well under J9b's own ~14GB observed
# peak too. Budget generously above that plus headroom, since J9b is already
# running on this GPU.
MIN_FREE_GB="${MIN_FREE_GB:-18}"
echo "[j10] waiting for >= ${MIN_FREE_GB}GB free VRAM (shared GPU, J9b also running)..." \
  | tee -a "$LOGDIR/j10_orchestrator.log"
while true; do
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  free_gb=$(( free_mb / 1024 ))
  if [ "$free_gb" -ge "$MIN_FREE_GB" ]; then
    echo "[j10] ${free_gb}GB free, proceeding." | tee -a "$LOGDIR/j10_orchestrator.log"
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
RUN_ID="symmetric-4ffn-both-branches-final-chr1-seed17"
RECIPE="configs/models/functional_fusion/j10_symmetric_4ffn_both_branches.yaml"

cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[j10] 1-epoch mode=final smoke test..." | tee -a "$LOGDIR/j10_orchestrator.log"
rm -rf "/tmp/${RUN_ID}_smoke"
if ! "$PY" scripts/train.py --model rna_methylation --scope chr1 --engine matched_chr1_shared_backbone \
    --mode final --recipe "$RECIPE" \
    --output-root "/tmp/${RUN_ID}_smoke" --run-id smoke-1ep --epochs 1 --seed 17 \
    "${DATA_ARGS[@]}" > "$LOGDIR/smoke_${RUN_ID}.log" 2>&1; then
  echo "[j10] smoke test FAILED; see $LOGDIR/smoke_${RUN_ID}.log; STOPPING." \
    | tee -a "$LOGDIR/j10_orchestrator.log"
  exit 1
fi
rm -rf "/tmp/${RUN_ID}_smoke"
echo "[j10] smoke test passed. Launching the full 80-epoch run..." \
  | tee -a "$LOGDIR/j10_orchestrator.log"

rm -rf "$OUTROOT/runs/locus_cls_joint/chr1/${RUN_ID}"  # fresh run, not a resume
if ! "$PY" scripts/train.py --model rna_methylation --scope chr1 --engine matched_chr1_shared_backbone \
    --mode final --recipe "$RECIPE" \
    --output-root "$OUTROOT" --run-id "$RUN_ID" --seed 17 \
    "${DATA_ARGS[@]}" > "$LOGDIR/${RUN_ID}.log" 2>&1; then
  echo "[j10] efficient_single_attn_4ffn_residual_functional4 FAILED (see $LOGDIR/${RUN_ID}.log)." \
    | tee -a "$LOGDIR/j10_orchestrator.log"
  exit 1
fi
echo "[j10] efficient_single_attn_4ffn_residual_functional4 finished successfully." \
  | tee -a "$LOGDIR/j10_orchestrator.log"
