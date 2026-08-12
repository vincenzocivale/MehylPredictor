#!/usr/bin/env bash
# Isolated E0 architecture ablation:
#   - remove mean-RNA anchoring
#   - remove the locus-specific variability gate
#
# Everything else is inherited from the current E0 baseline config and exact
# MethylProphet Array-chr1 adapter: data, split, seed, loss, optimizer,
# full-coverage schedule, batches, early stopping and final-refit protocol.
#
# This script never modifies configs/train.yaml. It materializes the exact
# derived ablation config inside RUN_ROOT for provenance.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SEED="${SEED:-17}"
SOURCE_CONFIG="${SOURCE_CONFIG:-configs/train.yaml}"
RUN_ROOT="${RUN_ROOT:-artifacts/architecture_ablation/e0_no_anchor_no_gate_seed${SEED}}"
BASELINE_RUN="${BASELINE_RUN:-artifacts/methylprophet_comparison/current_model_tcga_array_chr1_seed${SEED}}"
ABLATION_CONFIG="$RUN_ROOT/ablation_base_config.yaml"

mkdir -p "$RUN_ROOT"

python - "$SOURCE_CONFIG" "$ABLATION_CONFIG" <<'PY'
from pathlib import Path
import sys
import yaml

source = Path(sys.argv[1])
target = Path(sys.argv[2])

raw = yaml.safe_load(source.read_text())
raw["model"]["anchor_to_mean_rna"] = False
raw["model"]["gate"]["kind"] = "none"

target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(yaml.safe_dump(raw, sort_keys=False))

# Fail closed: this experiment is intended to differ from the baseline in
# exactly these two architecture switches.
check = yaml.safe_load(target.read_text())
assert check["model"]["anchor_to_mean_rna"] is False
assert check["model"]["gate"]["kind"] == "none"

print(f"wrote isolated ablation config: {target}")
print("architecture: anchor_to_mean_rna=false, gate.kind=none")
PY

echo "=== E0 no-anchor/no-gate ablation ==="
echo "seed=$SEED"
echo "run_root=$RUN_ROOT"
echo "baseline_run=$BASELINE_RUN"
echo "base_config=$ABLATION_CONFIG"

# The released MP prediction dataset is currently optional/gated. Do not turn
# an architecture experiment into a failed download attempt unless explicitly
# requested by the caller.
export SEED
export RUN_ROOT
export BASE_CONFIG="$ABLATION_CONFIG"
export AUTO_DOWNLOAD_MP_EVAL="${AUTO_DOWNLOAD_MP_EVAL:-0}"

bash scripts/run_overnight_current_model_vs_mp.sh

if [[ -f "$BASELINE_RUN/evaluation/headline.json" && -f "$RUN_ROOT/evaluation/headline.json" ]]; then
  python scripts/compare_e0_architecture_runs.py \
    --baseline "$BASELINE_RUN/evaluation/headline.json" \
    --candidate "$RUN_ROOT/evaluation/headline.json" \
    --label "no_anchor_no_gate" \
    --output "$RUN_ROOT/evaluation/compare_to_current_e0.json"
else
  echo "[warning] comparison skipped: baseline or candidate headline.json is missing"
fi
