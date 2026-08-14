#!/usr/bin/env bash
# TCGA chromosome-1 benchmark (published as MethylProphet Table 5).
# IMPORTANT: the published TCGA Table-5 experiment is Array+EPIC+WGBS on chr1.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

DATA_ROOT="${DATA_ROOT:-/raid/DATASETS/MethylPredictionData}"
CANONICAL_ROOT="${TCGA_CANONICAL_ROOT:-$DATA_ROOT/datasets/methylprophet_repro_v1}"
ATLAS="${NTV3_ATLAS:-$CANONICAL_ROOT/cpg/ntv3/ntv3_cpg_atlas_v1.h5}"
HG38_FASTA="${HG38_FASTA:-}"
DERIVED_ROOT="${DERIVED_ROOT:-$DATA_ROOT/derived/methylprophet_table5_tcga_chr1}"
ALL_EXPERIMENTS_ROOT="${ALL_EXPERIMENTS_ROOT:-$DATA_ROOT/experiments}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$ALL_EXPERIMENTS_ROOT/MethylPredictor/tcga_chr1}"
SEED="${SEED:-17}"
# Full DataConfig used to (re)build the derived caches -- see scripts/tcga_chr1/prepare.py.
CONFIG="${CONFIG:-configs/tcga_chr1/reference.yaml}"
# Experiment spec used for training -- see scripts/tcga_chr1/run_experiment.py.
EXPERIMENT="${EXPERIMENT:-configs/tcga_chr1/experiments/reference.yaml}"
EXPERIMENT_ID="$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['experiment_id'])" "$EXPERIMENT")"
RUN_ROOT="${RUN_ROOT:-$EXPERIMENT_ROOT/$EXPERIMENT_ID}"
GPU="${GPU:-0}"
MP_EVAL_DIR="${MP_EVAL_DIR:-}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

# Used only when FINAL_EPOCHS is not explicitly supplied.  The architecture is
# already frozen; this only transfers the optimizer-update budget to the much
# larger pair-complete Table-5 epoch.
CONFIRM_RUN="${CONFIRM_RUN:-$ALL_EXPERIMENTS_ROOT/current_model/architecture_ablations_array_chr1/seed17/runs/confirm_rna256_no_gate_no_anchor}"
FINAL_EPOCHS="${FINAL_EPOCHS:-}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/evaluation"
exec > >(tee -a "$RUN_ROOT/logs/launcher.log") 2>&1

echo "=== MethylPredictor / TCGA chromosome-1 exact benchmark (MethylProphet Table 5) ==="
date -Is
echo "git_head=$(git rev-parse HEAD)"
echo "canonical_root=$CANONICAL_ROOT"
echo "atlas=$ATLAS"
echo "derived_root=$DERIVED_ROOT"
echo "experiment_id=$EXPERIMENT_ID"
echo "run_root=$RUN_ROOT"
echo "gpu=$GPU seed=$SEED"
echo "reference=TCGA Array+EPIC+WGBS, chromosome 1, MethylProphet Table 5"

if [[ -z "$HG38_FASTA" ]]; then
  for candidate in \
    "$DATA_ROOT/hg38.fa" \
    "$DATA_ROOT/reference/hg38.fa" \
    "$DATA_ROOT/reference/hg38/hg38.fa" \
    "$CANONICAL_ROOT/reference/hg38.fa" \
    "$CANONICAL_ROOT/reference/hg38/hg38.fa" \
    "$DATA_ROOT/hg38.fa.gz"
  do
    if [[ -f "$candidate" ]]; then
      HG38_FASTA="$candidate"
      break
    fi
  done
fi
if [[ -z "$HG38_FASTA" || ! -f "$HG38_FASTA" ]]; then
  echo "FATAL: HG38_FASTA is required only to reproduce MethylProphet's 1000-bp no-N CpG filter." >&2
  echo "No NTv3 inference will be run. Locate the same hg38 FASTA and relaunch, e.g.:" >&2
  echo "  find /raid/DATASETS \"$HOME\" -type f \( -name 'hg38.fa' -o -name 'hg38.fa.gz' \) 2>/dev/null" >&2
  echo "  HG38_FASTA=/path/to/hg38.fa GPU=$GPU bash scripts/tcga_chr1/run.sh" >&2
  exit 2
fi
echo "hg38_fasta=$HG38_FASTA"

for path in "$CANONICAL_ROOT" "$ATLAS" "$CONFIG" "$HG38_FASTA"; do
  [[ -e "$path" ]] || { echo "FATAL missing prerequisite: $path" >&2; exit 2; }
done

python - <<'PY'
mods = ["torch", "h5py", "pyarrow", "pandas", "yaml", "wandb"]
for name in mods:
    try: __import__(name)
    except Exception as exc: raise SystemExit(f"FATAL: cannot import {name}: {exc}")
import torch
if not torch.cuda.is_available(): raise SystemExit("FATAL: CUDA unavailable")
print("Python/CUDA/W&B dependency preflight: PASS")
PY

CUDA_VISIBLE_DEVICES="$GPU" python - <<'PY'
import torch
print("selected GPU:", torch.cuda.get_device_name(0))
x = torch.randn((512, 512), device="cuda"); (x @ x).sum().item()
print("CUDA smoke: PASS")
PY

if [[ ! -f "$DERIVED_ROOT/.done" || -n "$MP_EVAL_DIR" ]]; then
  if [[ -f "$DERIVED_ROOT/.done" ]]; then
    echo "=== verify cached Table-5 protocol directly against released MP eval ==="
  else
    echo "=== reconstruct and audit exact Table-5 protocol + caches + prior ==="
  fi
  PREP_ARGS=(
    --canonical-root "$CANONICAL_ROOT"
    --atlas "$ATLAS"
    --hg38-fasta "$HG38_FASTA"
    --config "$CONFIG"
    --output "$DERIVED_ROOT"
    --device cuda
  )
  if [[ -n "$MP_EVAL_DIR" ]]; then
    [[ -e "$MP_EVAL_DIR" ]] || { echo "FATAL MP_EVAL_DIR does not exist: $MP_EVAL_DIR" >&2; exit 2; }
    PREP_ARGS+=(--mp-eval "$MP_EVAL_DIR")
  fi
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/tcga_chr1/prepare.py "${PREP_ARGS[@]}"
else
  echo "[skip] exact Table-5 derived caches already prepared"
fi

# Fail closed even for cached preparation.
python - "$DERIVED_ROOT/table5_protocol/protocol.json" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
assert p["protocol"] == "methylprophet_table5_tcga_chr1", p
assert p["status"] == "exact_table5_ready", p.get("status")
a=p["finite_pair_audit"]
assert a["status"] == "exact_match", a
# This repo's canonical bundle carries no Array<->WGBS patient crosswalk, so
# the reconstructed seed=42 split is 8,260/918 (not the paper's 8,258/920);
# these are the resulting reproducible observed-pair counts -- see
# src/methylation_predictor/benchmark/table5/protocol.py and docs/BENCHMARK_TABLE5.md.
assert a["training_total_observed"] == 454_931_749, a
assert a["training_observed"] == {"array":275_093_377,"epic":115_856_100,"wgbs":63_982_272}, a
assert a["evaluation_observed"] == {
    "train_cpg_x_val_sample":30_563_936,
    "val_cpg_x_train_sample":55_154_676,
    "val_cpg_x_val_sample":6_129_992,
}, a
print("Table-5 exact protocol preflight: PASS")
PY

if [[ "$PREPARE_ONLY" == 1 ]]; then
  echo "PREPARE_ONLY=1: exact protocol/caches are ready; stopping before training"
  exit 0
fi

if [[ -z "$FINAL_EPOCHS" ]]; then
  echo "=== resolve frozen final epoch budget from architecture confirm ==="
  [[ -d "$CONFIRM_RUN" ]] || {
    echo "FATAL: confirm run not found: $CONFIRM_RUN" >&2
    echo "Set FINAL_EPOCHS explicitly or CONFIRM_RUN to the completed confirmation run." >&2
    exit 2
  }
  FINAL_EPOCHS="$(python scripts/resolve_final_epoch_budget.py \
    --confirm-run "$CONFIRM_RUN" \
    --table5-protocol "$DERIVED_ROOT/table5_protocol" \
    --json-output "$RUN_ROOT/epoch_budget.json" \
    --epochs-only)"
else
  python - "$FINAL_EPOCHS" "$RUN_ROOT/epoch_budget.json" <<'PY'
import json, sys
n=int(sys.argv[1])
if n < 1: raise SystemExit("FINAL_EPOCHS must be >= 1")
open(sys.argv[2],"w").write(json.dumps({
  "resolved_final_epochs":n,
  "policy":"explicit FINAL_EPOCHS override",
  "schedule":"complete Table-5 Cartesian pair coverage",
  "finite_training_pairs_per_epoch":454_931_749,
},indent=2)+"\n")
PY
fi

echo "fixed_final_epochs=$FINAL_EPOCHS"
cat "$RUN_ROOT/epoch_budget.json"

if [[ ! -f "$RUN_ROOT/.done" ]]; then
  echo "=== one-stage exact-Table5 pair-complete training ($EXPERIMENT_ID) ==="
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/tcga_chr1/run_experiment.py \
    --experiment "$EXPERIMENT" \
    --canonical-root "$CANONICAL_ROOT" \
    --prepared-root "$DERIVED_ROOT" \
    --output-root "$EXPERIMENT_ROOT" \
    --epochs "$FINAL_EPOCHS" \
    --seed "$SEED"
else
  echo "[skip] final training already complete"
fi

python scripts/tcga_chr1/report.py \
  --input "$RUN_ROOT/evaluation/headline.json" \
  --output-dir "$RUN_ROOT/evaluation"

echo
echo "=== TCGA CHR1 TRAINING/EVALUATION COMPLETE ==="
echo "checkpoint=$RUN_ROOT/checkpoints/final.pt"
echo "table5_report=$RUN_ROOT/evaluation/headline.json"
echo "table5_csv=$RUN_ROOT/evaluation/table5_comparison.csv"
echo "table5_markdown=$RUN_ROOT/evaluation/table5_comparison.md"
echo "protocol_audit=$DERIVED_ROOT/table5_protocol/protocol.json"
echo "The headline report already contains OURS, the published MethylProphet Table-5 values, and deltas."
