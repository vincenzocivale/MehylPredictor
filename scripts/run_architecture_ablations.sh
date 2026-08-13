#!/usr/bin/env bash
# Controlled architecture ablations on the exact released MethylProphet
# Array-chr1 split. This is the cheap architecture-selection stage: every
# variant sees the same Array train universe with deterministic full coverage.
# Only the winning architecture should later be promoted to mixed-source E2.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

SEED="${SEED:-17}"
GPUS_CSV="${GPUS:-0}"
VARIANTS_CSV="${VARIANTS:-baseline,no_gate,no_anchor,no_product,rna256,direct_prediction}"
CHECKPOINT_METRIC="${CHECKPOINT_METRIC:-mas_pcc}"

DATA_ROOT="${DATA_ROOT:-/raid/DATASETS/MethylPredictionData}"
CANONICAL_ROOT="${TCGA_CANONICAL_ROOT:-$DATA_ROOT/datasets/methylprophet_repro_v1}"
NTV3_ATLAS="${NTV3_ATLAS:-$CANONICAL_ROOT/cpg/ntv3/ntv3_cpg_atlas_v1.h5}"
EMBEDDING_CACHE="${EMBEDDING_CACHE:-$DATA_ROOT/derived/tcga_canonical/array_chr1_ntv3_atlas_v1.h5}"
LOCUS_FEATURES="${LOCUS_FEATURES:-$DATA_ROOT/locus_features.parquet}"
BASE_CONFIG="${BASE_CONFIG:-configs/train.yaml}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$DATA_ROOT/experiments}"
RUN_ROOT="${RUN_ROOT:-$EXPERIMENT_ROOT/current_model/architecture_ablations_array_chr1/seed${SEED}}"

mkdir -p "$RUN_ROOT/base_configs" "$RUN_ROOT/logs" "$RUN_ROOT/runs"

for path in \
  "$CANONICAL_ROOT" \
  "$CANONICAL_ROOT/rna/tcga_rna_official_full.h5" \
  "$CANONICAL_ROOT/methylation/tcga_array_official_full.h5" \
  "$CANONICAL_ROOT/protocols/tcga_array_chr1/protocol.json" \
  "$NTV3_ATLAS" \
  "$LOCUS_FEATURES" \
  "$BASE_CONFIG"
do
  [[ -e "$path" ]] || {
    echo "FATAL: missing prerequisite: $path" >&2
    if [[ "$path" == "$LOCUS_FEATURES" ]]; then
      echo "The Array scalar prior/variability table is required for architecture ablations." >&2
      echo "Override with LOCUS_FEATURES=/path/to/locus_features.parquet if it was moved." >&2
    fi
    exit 2
  }
done

IFS=',' read -r -a GPUS_ARR <<< "$GPUS_CSV"
IFS=',' read -r -a VARIANTS_ARR <<< "$VARIANTS_CSV"
[[ "${#GPUS_ARR[@]}" -gt 0 ]] || { echo "FATAL: GPUS is empty" >&2; exit 2; }

python -c "import torch, h5py, pyarrow" 2>/dev/null || {
  echo "FATAL: the python resolved on PATH ($(command -v python)) cannot import torch/h5py/pyarrow." >&2
  echo "Activate the correct environment first, e.g.: conda activate methyl-predictor" >&2
  exit 2
}

echo "=== architecture ablations / exact Array-chr1 ==="
echo "seed=$SEED"
echo "gpus=${GPUS_ARR[*]}"
echo "variants=${VARIANTS_ARR[*]}"
echo "checkpoint_metric=$CHECKPOINT_METRIC"
echo "canonical_root=$CANONICAL_ROOT"
echo "ntv3_atlas=$NTV3_ATLAS"
echo "embedding_cache=$EMBEDDING_CACHE"
echo "locus_features=$LOCUS_FEATURES"
echo "run_root=$RUN_ROOT"

# Materialize only the 40,627 exact Array-chr1 rows from the 5.7M-row atlas.
# This avoids building a multi-million-entry Python ID map in every ablation.
if [[ ! -f "$EMBEDDING_CACHE" ]]; then
  echo "=== building shared Array-chr1 NTv3 embedding cache ==="
  python scripts/build_array_chr1_embedding_cache.py \
    --canonical-root "$CANONICAL_ROOT" \
    --atlas "$NTV3_ATLAS" \
    --output "$EMBEDDING_CACHE"
fi

for variant in "${VARIANTS_ARR[@]}"; do
  python scripts/render_architecture_ablation_config.py \
    --base-config "$BASE_CONFIG" \
    --variant "$variant" \
    --output "$RUN_ROOT/base_configs/${variant}.yaml" \
    --run-name "array-chr1-${variant}-seed${SEED}" \
    --run-output-dir "$RUN_ROOT/runs/$variant" \
    --checkpoint-metric "$CHECKPOINT_METRIC"
done

run_variant () {
  local variant="$1" gpu="$2"
  local output="$RUN_ROOT/runs/$variant"
  local log="$RUN_ROOT/logs/${variant}.log"
  local base_cfg="$RUN_ROOT/base_configs/${variant}.yaml"
  local dev_cfg="$output/dev_config.yaml"
  local final_cfg="$output/final_config.yaml"

  mkdir -p "$output/evaluation"
  echo "[$(date -Is)] START variant=$variant physical_gpu=$gpu" | tee -a "$log"

  # NOTE: run_variant is invoked as `if ! run_variant ...; then` below, which
  # disables `set -e` for this entire function body (a documented bash
  # behavior: errexit does not apply to commands whose exit status is being
  # tested). Every step below must therefore check its own exit status
  # explicitly with `|| { ...; return 1; }` instead of relying on -e.
  if [[ ! -f "$dev_cfg" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/prepare_array_chr1_ablation.py prepare \
      --canonical-root "$CANONICAL_ROOT" \
      --locus-embeddings "$EMBEDDING_CACHE" \
      --locus-features "$LOCUS_FEATURES" \
      --base-config "$base_cfg" \
      --output-root "$output" \
      --seed "$SEED" >> "$log" 2>&1 || {
        echo "[$(date -Is)] FAILED variant=$variant step=prepare (see $log)" | tee -a "$log" >&2
        return 1
      }
  fi

  CUDA_VISIBLE_DEVICES="$gpu" python -m methylation_predictor validate \
    --config "$dev_cfg" >> "$log" 2>&1 || {
      echo "[$(date -Is)] FAILED variant=$variant step=validate-dev (see $log)" | tee -a "$log" >&2
      return 1
    }

  if [[ ! -f "$output/development/metrics.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" python -m methylation_predictor train \
      --config "$dev_cfg" >> "$log" 2>&1 || {
        echo "[$(date -Is)] FAILED variant=$variant step=train-dev (see $log)" | tee -a "$log" >&2
        return 1
      }
  fi

  local best_epoch
  best_epoch="$(
    python - "$output/development/metrics.json" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1]))["best_epoch"]))
PY
  )" || {
    echo "[$(date -Is)] FAILED variant=$variant step=select-best-epoch (see $log)" | tee -a "$log" >&2
    return 1
  }
  echo "[$(date -Is)] variant=$variant selected_best_epoch=$best_epoch" | tee -a "$log"

  if [[ ! -f "$final_cfg" ]]; then
    python scripts/prepare_array_chr1_ablation.py render-final \
      --dev-config "$dev_cfg" \
      --output-root "$output" \
      --best-epoch "$best_epoch" \
      --seed "$SEED" >> "$log" 2>&1 || {
        echo "[$(date -Is)] FAILED variant=$variant step=render-final (see $log)" | tee -a "$log" >&2
        return 1
      }
  fi

  CUDA_VISIBLE_DEVICES="$gpu" python -m methylation_predictor validate \
    --config "$final_cfg" >> "$log" 2>&1 || {
      echo "[$(date -Is)] FAILED variant=$variant step=validate-final (see $log)" | tee -a "$log" >&2
      return 1
    }

  if [[ ! -f "$output/final_refit/metrics.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" python -m methylation_predictor train \
      --config "$final_cfg" >> "$log" 2>&1 || {
        echo "[$(date -Is)] FAILED variant=$variant step=train-final (see $log)" | tee -a "$log" >&2
        return 1
      }
  fi

  CUDA_VISIBLE_DEVICES="$gpu" python scripts/evaluate_current_model_vs_methylprophet.py \
    --config "$final_cfg" \
    --checkpoint "$output/final_refit/best.pt" \
    --output "$output/evaluation/headline.json" >> "$log" 2>&1 || {
      echo "[$(date -Is)] FAILED variant=$variant step=evaluate (see $log)" | tee -a "$log" >&2
      return 1
    }

  echo "[$(date -Is)] DONE variant=$variant physical_gpu=$gpu" | tee -a "$log"
}

# Static round-robin queues. Each GPU runs one variant at a time; with two GPUs
# two ablations are trained concurrently and each GPU takes the next item in
# its own queue when finished.
PIDS=()
WORLD="${#GPUS_ARR[@]}"
for ((rank=0; rank<WORLD; rank++)); do
  gpu="${GPUS_ARR[$rank]}"
  (
    worker_rc=0
    for ((i=rank; i<${#VARIANTS_ARR[@]}; i+=WORLD)); do
      if ! run_variant "${VARIANTS_ARR[$i]}" "$gpu"; then
        worker_rc=1
      fi
    done
    exit "$worker_rc"
  ) &
  PIDS+=("$!")
done

FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAIL=1
done

python scripts/summarize_architecture_ablations.py --run-root "$RUN_ROOT" || true

if [[ "$FAIL" != 0 ]]; then
  echo "One or more ablations failed. Inspect $RUN_ROOT/logs/*.log" >&2
  exit 4
fi

echo "=== COMPLETE ==="
echo "results: $RUN_ROOT/runs"
echo "summary: $RUN_ROOT/summary.tsv"
