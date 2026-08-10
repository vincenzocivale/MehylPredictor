#!/usr/bin/env bash
# One-command unattended benchmark of the CURRENT RNA2DNAmModel architecture
# on the exact MethylProphet Array-chr1 protocol.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SEED="${SEED:-17}"
RUN_ROOT="${RUN_ROOT:-artifacts/methylprophet_comparison/current_model_tcga_array_chr1_seed${SEED}}"
CANONICAL_ROOT="${TCGA_CANONICAL_ROOT:-/raid/DATASETS/MethylPredictionData/methylprophet_official/official_training_data}"
LOCUS_EMBEDDINGS="${LOCUS_EMBEDDINGS:-/raid/DATASETS/MethylPredictionData/locus_embeddings.h5}"
LOCUS_FEATURES="${LOCUS_FEATURES:-/raid/DATASETS/MethylPredictionData/locus_features.parquet}"
BASE_CONFIG="${BASE_CONFIG:-configs/train.yaml}"
MP_EVAL_DIR="${MP_EVAL_DIR:-}"
MP_EVAL_REPO="${MP_EVAL_REPO:-MethylProphet/eval-tcga_array_chr1-32xl40s-c2b2}"
AUTO_DOWNLOAD_MP_EVAL="${AUTO_DOWNLOAD_MP_EVAL:-0}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/manifests" "$RUN_ROOT/evaluation"
exec > >(tee -a "$RUN_ROOT/logs/overnight.log") 2>&1

echo "=== current-model vs MethylProphet overnight benchmark ==="
date -Is
echo "repo=$(pwd)"
echo "git_head=$(git rev-parse HEAD)"
echo "seed=$SEED"
echo "run_root=$RUN_ROOT"
echo "canonical_root=$CANONICAL_ROOT"
echo

# Keep the current checkout importable without changing model code.
python - <<'PY'
import methylation_predictor
print("methylation_predictor import: PASS")
PY

if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
  echo "=== 0/6 regression tests ==="
  pytest -q
fi

echo "=== 1/6 exact protocol + NTv3 feature coverage preflight ==="
set +e
python scripts/prepare_current_model_mp_benchmark.py prepare \
  --canonical-root "$CANONICAL_ROOT" \
  --locus-embeddings "$LOCUS_EMBEDDINGS" \
  --locus-features "$LOCUS_FEATURES" \
  --base-config "$BASE_CONFIG" \
  --output-root "$RUN_ROOT" \
  --seed "$SEED"
PREP_RC=$?
set -e
if [[ "$PREP_RC" -eq 42 ]]; then
  echo
  echo "FATAL: the current architecture is missing NTv3-derived CpG features for the exact chr1 protocol."
  echo "No training was started. See: $RUN_ROOT/manifests/feature_audit.json"
  echo "Missing IDs are saved as missing_embedding_cpg_idx.npy / missing_feature_cpg_idx.npy."
  echo "This fail-closed behavior prevents silently changing RNA2DNAmModel or its prior."
  exit 42
elif [[ "$PREP_RC" -ne 0 ]]; then
  exit "$PREP_RC"
fi

DEV_CONFIG="$RUN_ROOT/dev_config.yaml"
FINAL_CONFIG="$RUN_ROOT/final_config.yaml"
DEV_DIR="$RUN_ROOT/development"
FINAL_DIR="$RUN_ROOT/final_refit"

echo "=== validate canonical adapter ==="
python -m methylation_predictor validate --config "$DEV_CONFIG" | tee "$RUN_ROOT/manifests/dev_bundle_summary.json"

if [[ ! -f "$DEV_DIR/.done" ]]; then
  echo "=== 2/6 nested-development training ==="
  python -m methylation_predictor train --config "$DEV_CONFIG"
  touch "$DEV_DIR/.done"
else
  echo "[skip] development training already complete"
fi

BEST_EPOCH="$(python - "$DEV_DIR/metrics.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["best_epoch"])
PY
)"
echo "selected best_epoch=$BEST_EPOCH"

echo "=== 3/6 render exact final refit ==="
python scripts/prepare_current_model_mp_benchmark.py render-final \
  --dev-config "$DEV_CONFIG" \
  --output-root "$RUN_ROOT" \
  --best-epoch "$BEST_EPOCH" \
  --seed "$SEED"
python -m methylation_predictor validate --config "$FINAL_CONFIG" | tee "$RUN_ROOT/manifests/final_bundle_summary.json"

if [[ ! -f "$FINAL_DIR/.done" ]]; then
  echo "=== 4/6 final refit on all official train IDs ==="
  python -m methylation_predictor train --config "$FINAL_CONFIG"
  touch "$FINAL_DIR/.done"
else
  echo "[skip] final refit already complete"
fi

# Optional automatic download of the exact Array-only released evaluation.
# Disabled by default because the repository may be gated. Failure here does
# NOT invalidate our training/evaluation; it only disables the paired MP scan.
if [[ -z "$MP_EVAL_DIR" && "$AUTO_DOWNLOAD_MP_EVAL" == "1" ]]; then
  MP_CACHE="$RUN_ROOT/methylprophet_released_eval"
  echo "=== 5/6 acquire released MethylProphet evaluation (optional) ==="
  set +e
  if command -v hf >/dev/null 2>&1; then
    hf download "$MP_EVAL_REPO" --repo-type dataset --local-dir "$MP_CACHE"
    DL_RC=$?
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$MP_EVAL_REPO" --repo-type dataset --local-dir "$MP_CACHE"
    DL_RC=$?
  else
    DL_RC=127
  fi
  set -e
  if [[ "$DL_RC" -eq 0 ]]; then
    MP_EVAL_DIR="$MP_CACHE"
  else
    echo "[warning] could not auto-download $MP_EVAL_REPO (rc=$DL_RC)."
    echo "[warning] continuing with exact OURS evaluation only. Set MP_EVAL_DIR and rerun; training will be skipped."
  fi
fi

echo "=== 6/6 full official three-view evaluation ==="
EVAL_ARGS=(
  --config "$FINAL_CONFIG"
  --checkpoint "$FINAL_DIR/best.pt"
  --output "$RUN_ROOT/evaluation/headline.json"
)
if [[ -n "$MP_EVAL_DIR" && -d "$MP_EVAL_DIR" ]]; then
  EVAL_ARGS+=(--mp-eval "$MP_EVAL_DIR" --mp-label "$MP_EVAL_REPO")
fi
python scripts/evaluate_current_model_vs_methylprophet.py "${EVAL_ARGS[@]}"

echo
echo "=== COMPLETE ==="
date -Is
echo "checkpoint: $FINAL_DIR/best.pt"
echo "headline:   $RUN_ROOT/evaluation/headline.json"
echo "log:        $RUN_ROOT/logs/overnight.log"
