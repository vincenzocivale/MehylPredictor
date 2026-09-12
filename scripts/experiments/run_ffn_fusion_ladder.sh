#!/bin/bash
# FFN-fusion ladder (2026-09-12, J9a/b/c): 2 extra FFN residual blocks on
# EACH branch (functional locus branch + gene-expression/RNA-attention
# branch, on top of J4's single locus<-RNA cross-attention) -- identical
# extra capacity across all three cells, only the final recombination
# mechanism varies:
#
#   ffn_fusion_two_stream_residual (j9c) -- cross-injection residual mixing:
#       h_mix = h_c + W_gf(state); s_mix = state + W_fg(h_c); z = h_mix+s_mix
#   ffn_fusion_film                (j9b) -- gene-expression branch generates
#       (gamma, beta) that modulate the functional branch (FiLM)
#   ffn_fusion_concat              (j9a) -- plain concatenation, the
#       baseline mechanism already used by J0/J4/J5/J7/J8
#
# See src/methylation_predictor/modeling/ablation.py's
# FunctionalGeneFFNFusionPredictor docstring for the full rationale.
#
# Run order (user-requested, 2026-09-12): two_stream_residual, then film,
# then concat. Each cell: 1-epoch mode=final smoke test (must pass before
# the full run starts), then the full 80-epoch run, then
# evaluate_rna_checkpoint on all three TRUE chr1 official views. A cell's
# failure stops the ladder (does not attempt the next cell).
#
# Same cost class as J4 (1x cross-attention + 4 total FFN-residual blocks,
# here split 2+2 across branches instead of 4 sequential) -- reuses J0-J6's
# batching (array sample_size 640) unchanged, no VRAM pre-check needed
# unlike J7/J8's 8/16-FFN cells.
#
# Meant to be started via nohup so it survives SSH disconnection:
#
#   nohup bash scripts/experiments/run_ffn_fusion_ladder.sh \
#     > logs/functional_concat_final/ffn_fusion_orchestrator_stdout.log 2>&1 &
#   disown
#
# Adjust DATA_ARGS below if this machine mounts the canonical bundle/derived
# caches at different paths than /home/vcivale/dune_data/...
set -euo pipefail
source /home/vcivale/miniconda3/etc/profile.d/conda.sh
conda activate methyl-predictor

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"
LOGDIR="$ROOT/logs/functional_concat_final"
mkdir -p "$LOGDIR"
ORCH_LOG="$LOGDIR/ffn_fusion_orchestrator.log"

DATA_ARGS=(
  --canonical-root /home/vcivale/dune_data/datasets/methylprophet_repro_v1
  --prior-cache /home/vcivale/dune_data/derived/methylprophet_table5_tcga_chr1/features
  --rna-cache /home/vcivale/dune_data/derived/methylprophet_table5_tcga_chr1/rna
  --registry /home/vcivale/dune_data/datasets/methylprophet_repro_v1/cpg/registries/array_cpg_map.parquet
  --prepared-root /home/vcivale/dune_data/derived/methylprophet_table5_tcga_chr1
  --cpg-targets-dir /home/vcivale/dune_data/derived/cpg_statistics/chr1
  --functional-atlas /home/vcivale/dune_data/derived/ntv3_functional_peak_atlas_chr1_all_sources
  --annotation-cache /home/vcivale/dune_data/derived/ntv3_probe_targets/chr1_annotation_features_all_sources
  
)
OUTROOT="/home/vcivale/dune_data/experiments/runs"

cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# `wandb login` was completed on this machine (2026-09-12) -- recipes now
# set tracking.mode=online directly.

run_cell () {
  local recipe="$1" run_id="$2" label="$3"
  local run_dir="$OUTROOT/runs/locus_cls_joint/chr1/${run_id}"

  echo "[ffn_fusion] $label: 1-epoch mode=final smoke test..." | tee -a "$ORCH_LOG"
  rm -rf "/tmp/${run_id}_smoke"
  if ! "$PY" scripts/train.py --model rna_methylation --scope chr1  \
      --mode final --recipe "$recipe" \
      --output-root "/tmp/${run_id}_smoke" --run-id smoke-1ep --epochs 1 --seed 17 \
      "${DATA_ARGS[@]}" > "$LOGDIR/smoke_${run_id}.log" 2>&1; then
    echo "[ffn_fusion] $label smoke test FAILED; see $LOGDIR/smoke_${run_id}.log; STOPPING LADDER." \
      | tee -a "$ORCH_LOG"
    exit 1
  fi
  rm -rf "/tmp/${run_id}_smoke"
  echo "[ffn_fusion] $label smoke test passed. Launching the full 80-epoch run..." \
    | tee -a "$ORCH_LOG"

  rm -rf "$run_dir"  # fresh run, not a resume
  if ! "$PY" scripts/train.py --model rna_methylation --scope chr1  \
      --mode final --recipe "$recipe" \
      --output-root "$OUTROOT" --run-id "$run_id" --seed 17 \
      "${DATA_ARGS[@]}" > "$LOGDIR/${run_id}.log" 2>&1; then
    echo "[ffn_fusion] $label training FAILED (see $LOGDIR/${run_id}.log); STOPPING LADDER." \
      | tee -a "$ORCH_LOG"
    exit 1
  fi
  echo "[ffn_fusion] $label training finished. Evaluating checkpoint..." | tee -a "$ORCH_LOG"

  if ! "$PY" scripts/evaluate.py --model rna_methylation  \
      --checkpoint "$run_dir/checkpoints/last.pt" --eval-scope chr1 \
      --recipe "$recipe" \
      "${DATA_ARGS[@]}" \
      --output "$run_dir/evaluation/chr1/official_split.json" \
      > "$LOGDIR/eval_${run_id}.log" 2>&1; then
    echo "[ffn_fusion] $label EVALUATION FAILED (see $LOGDIR/eval_${run_id}.log); STOPPING LADDER." \
      | tee -a "$ORCH_LOG"
    exit 1
  fi
  echo "[ffn_fusion] $label finished successfully -- see $run_dir/evaluation/chr1/official_split.json" \
    | tee -a "$ORCH_LOG"
}

run_cell configs/models/functional_fusion/j9c_ffn_fusion_two_stream_residual.yaml \
  ffn-fusion-two-stream-residual-final-chr1-seed17 "ffn_fusion_two_stream_residual (J9c)"

run_cell configs/models/functional_fusion/j9b_ffn_fusion_film.yaml \
  ffn-fusion-film-final-chr1-seed17 "ffn_fusion_film (J9b)"

run_cell configs/models/functional_fusion/j9a_ffn_fusion_concat.yaml \
  ffn-fusion-concat-final-chr1-seed17 "ffn_fusion_concat (J9a)"

echo "[ffn_fusion] ladder finished successfully -- all three cells trained and evaluated." \
  | tee -a "$ORCH_LOG"
