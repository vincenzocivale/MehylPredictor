#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

DATA_ROOT="${DATA_ROOT:-/raid/DATASETS/MethylPredictionData}"
CANONICAL_ROOT="${TCGA_CANONICAL_ROOT:-$DATA_ROOT/datasets/methylprophet_repro_v1}"
OUTPUT="${GENOMIC_PRIOR_V2_ROOT:-$DATA_ROOT/derived/genomic_prior_v2/array_genomewide}"
GPUS_CSV="${GPUS:-0}"
IFS=',' read -r -a GPU_IDS <<< "$GPUS_CSV"

if [[ ${#GPU_IDS[@]} -lt 1 ]]; then
  echo "FATAL: GPUS is empty" >&2
  exit 2
fi
for path in \
  "$CANONICAL_ROOT/rna/tcga_rna_official_full.h5" \
  "$CANONICAL_ROOT/methylation/tcga_array_official_full.h5" \
  "$CANONICAL_ROOT/protocols/array_genomewide/protocol.json" \
  "$CANONICAL_ROOT/cpg/ntv3/ntv3_cpg_atlas_v1.h5"; do
  [[ -e "$path" ]] || { echo "FATAL: missing prerequisite: $path" >&2; exit 2; }
done

mkdir -p "$OUTPUT/logs"
echo "canonical_root=$CANONICAL_ROOT"
echo "output=$OUTPUT"
echo "gpus=$GPUS_CSV"

echo "=== Stage 1/3: exact train-sample targets + Array NTv3 cache ==="
python scripts/rebuild_genomic_prior_v2.py \
  --canonical-root "$CANONICAL_ROOT" \
  --output "$OUTPUT" \
  prepare \
  2>&1 | tee "$OUTPUT/logs/prepare.log"

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "PREPARE_ONLY=1: stopping after target/embedding preparation"
  exit 0
fi

echo "=== Stage 2/3: 5 OOF probes + one full-fit probe ==="
TASKS=(fold0 fold1 fold2 fold3 fold4 full)
worker() {
  local worker_index="$1"
  local gpu="${GPU_IDS[$worker_index]}"
  local task_index task fold log
  for ((task_index=worker_index; task_index<${#TASKS[@]}; task_index+=${#GPU_IDS[@]})); do
    task="${TASKS[$task_index]}"
    log="$OUTPUT/logs/${task}.log"
    echo "[worker $worker_index gpu=$gpu] starting $task" | tee -a "$log"
    if [[ "$task" == fold* ]]; then
      fold="${task#fold}"
      CUDA_VISIBLE_DEVICES="$gpu" python scripts/rebuild_genomic_prior_v2.py \
        --canonical-root "$CANONICAL_ROOT" \
        --output "$OUTPUT" \
        fit-fold --fold "$fold" --device cuda \
        >> "$log" 2>&1
    else
      CUDA_VISIBLE_DEVICES="$gpu" python scripts/rebuild_genomic_prior_v2.py \
        --canonical-root "$CANONICAL_ROOT" \
        --output "$OUTPUT" \
        fit-full --device cuda \
        >> "$log" 2>&1
    fi
    echo "[worker $worker_index gpu=$gpu] finished $task" | tee -a "$log"
  done
}

pids=()
for ((i=0; i<${#GPU_IDS[@]}; i++)); do
  worker "$i" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "FATAL: at least one probe worker failed. Inspect $OUTPUT/logs/*.log" >&2
  exit 1
fi

echo "=== Stage 3/3: assemble frozen locus_features.parquet ==="
python scripts/rebuild_genomic_prior_v2.py \
  --canonical-root "$CANONICAL_ROOT" \
  --output "$OUTPUT" \
  assemble \
  2>&1 | tee "$OUTPUT/logs/assemble.log"

echo
echo "DONE"
echo "locus_features=$OUTPUT/locus_features.parquet"
echo "provenance=$OUTPUT/provenance.json"
echo
echo "Use for the architecture ablations with:"
echo "  LOCUS_FEATURES=$OUTPUT/locus_features.parquet GPUS=$GPUS_CSV bash scripts/run_architecture_ablations.sh"
