#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="${RUN_DIR:-artifacts/train/seed17}"
DEV_DIR="$RUN_DIR/development"
FINAL_DIR="$RUN_DIR/final_refit"
MANIFEST_DIR="$RUN_DIR/manifests"

DEV_CONFIG="${DEV_CONFIG:-configs/train.yaml}"
FINAL_CONFIG="${FINAL_CONFIG:-$RUN_DIR/final_config.yaml}"

mkdir -p "$DEV_DIR" "$FINAL_DIR" "$MANIFEST_DIR"

phase_done() { [[ -f "$1/.done" ]]; }
mark_phase_done() { touch "$1/.done"; }

echo "=== source ==="
git rev-parse HEAD
echo

if phase_done "$MANIFEST_DIR"; then
  echo "[skip] preflight already complete"
else
  echo "=== 0/4 preflight ==="
  PREFLIGHT_EXTRA_ARGS=()
  if [[ ! -d "/data/dataset/methylation/MethylProphetData/parquet/241231-tcga_array" ]]; then
    echo "[preflight] official MethylProphet split source not found on this machine -- skipping cross-check"
    PREFLIGHT_EXTRA_ARGS+=("--skip-official-split-check")
  fi
  python scripts/preflight_genomewide_fullcoverage.py \
    --output "$MANIFEST_DIR/preflight.json" \
    "${PREFLIGHT_EXTRA_ARGS[@]}"
  mark_phase_done "$MANIFEST_DIR"
fi

DEV_SPLIT_DIR="${DEV_SPLIT_DIR:-/raid/DATASETS/MethylPredictionData/rna_branch_inputs_dev_seed17}"
if [[ -f "$DEV_SPLIT_DIR/manifest.json" ]]; then
  echo "[skip] nested development split already exists"
else
  echo "=== 1/4 build nested development split ==="
  python scripts/build_dev_split.py
fi

if phase_done "$DEV_DIR"; then
  echo "[skip] development training already complete"
else
  echo "=== 2/4 development training ==="
  python -m methylation_predictor train --config "$DEV_CONFIG"
  mark_phase_done "$DEV_DIR"
fi

BEST_EPOCH="$(
  python - "$DEV_DIR/metrics.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    payload = json.load(handle)
print(payload["best_epoch"])
PY
)"
echo "selected best_epoch=$BEST_EPOCH"

echo "=== 3/4 render final-refit config ==="
python scripts/render_final_refit_config.py \
  --dev-config "$DEV_CONFIG" \
  --best-epoch "$BEST_EPOCH" \
  --output "$FINAL_CONFIG"

if phase_done "$FINAL_DIR"; then
  echo "[skip] final refit already complete"
else
  echo "=== 4/4 final refit ==="
  python -m methylation_predictor train --config "$FINAL_CONFIG"
  mark_phase_done "$FINAL_DIR"
fi

echo
echo "=== training complete ==="
echo "final checkpoint: $FINAL_DIR/best.pt"
echo "No evaluation, baseline comparison, bootstrap, ablation or report stage is run."
