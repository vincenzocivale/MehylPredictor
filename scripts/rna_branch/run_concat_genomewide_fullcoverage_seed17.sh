#!/usr/bin/env bash
# Autonomous orchestrator for the rigorous genome-wide concat full-coverage
# training run, seed 17. Every stage is guarded by a `.done` marker so the
# whole script is safely re-runnable (e.g. after an unexpected process death
# -- the trainer's own resume support, via checkpoint_latest.pt, picks up
# mid-training within a stage; this script's markers pick up BETWEEN stages).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="artifacts/rna_branch/concat_genomewide_fullcoverage_seed17_v1"
DEV_DIR="$RUN_DIR/development"
FINAL_DIR="$RUN_DIR/final_refit"
EVAL_DIR="$RUN_DIR/evaluation"
MANIFEST_DIR="$RUN_DIR/manifests"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$DEV_DIR" "$FINAL_DIR" "$EVAL_DIR" "$MANIFEST_DIR" "$LOG_DIR"

DEV_CONFIG="configs/rna_branch/tcga_genome_wide_concat_dev_seed17.yaml"
FINAL_CONFIG="configs/rna_branch/tcga_genome_wide_concat_final_seed17.yaml"
BILINEAR_CONFIG="artifacts/rna_branch/tcga_genome_wide_bilinear/seed17/config.yaml"
BILINEAR_CHECKPOINT="artifacts/rna_branch/tcga_genome_wide_bilinear/seed17/best.pt"
MP_RELEASED="artifacts/cache/methylprophet/upstream_outputs/eval/eval-tcga_mix_chr1-bs_512-c2b2/eval_results-test.parquet"
MP_CHECKPOINT="artifacts/cache/methylprophet/upstream_outputs/ckpts/tcga_mix_chr1-bs_512-c2b2/version_0/finished.ckpt"
EMPIRICAL_PRIOR="/data/dataset/methylation/genomic_encoder_genome_wide_scratch/genome_wide_features.parquet"

set +u  # conda's own activation hooks reference unbound vars under `set -u`
source /home/oem/miniconda3/etc/profile.d/conda.sh
conda activate methil-predictor
set -u

echo "=== host/environment info ==="
hostname
date -u
python -c "
import torch
print('python', __import__('sys').version)
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda device', torch.cuda.get_device_name(0))
    print('cuda version', torch.version.cuda)
"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
git rev-parse HEAD

stage_done() { [ -f "$1/.done" ]; }
mark_done() { touch "$1/.done"; }

# ---------------------------------------------------------------- Stage 0
if stage_done "$MANIFEST_DIR"; then
  echo "[skip] preflight already complete"
else
  echo "=== Stage 0: preflight ==="
  python scripts/rna_branch/preflight_genomewide_fullcoverage.py --output "$MANIFEST_DIR/preflight.json"
  mark_done "$MANIFEST_DIR"
fi

# ---------------------------------------------------------------- Stage 1
DEV_SPLIT_DIR="/data/dataset/methylation/genomic_encoder_genome_wide_scratch/rna_branch_inputs_dev_seed17"
if [ -f "$DEV_SPLIT_DIR/manifest.json" ]; then
  echo "[skip] nested dev split already built"
else
  echo "=== Stage 1: build nested dev split ==="
  python scripts/rna_branch/build_dev_split.py
fi

# ---------------------------------------------------------------- Stage 2
if stage_done "$DEV_DIR"; then
  echo "[skip] Stage 3A dev training already complete"
else
  echo "=== Stage 2: Stage 3A development training ==="
  python -m methylation_predictor.rna_branch.cli train --config "$DEV_CONFIG"
  mark_done "$DEV_DIR"
fi

# ---------------------------------------------------------------- Stage 3
BEST_EPOCH="$(python -c "import json; print(json.load(open('$DEV_DIR/metrics.json'))['best_epoch'])")"
echo "=== Stage 3: render final-refit config (best_epoch=$BEST_EPOCH) ==="
python scripts/rna_branch/render_final_refit_config.py \
  --dev-config "$DEV_CONFIG" --best-epoch "$BEST_EPOCH" --output "$FINAL_CONFIG"

# ---------------------------------------------------------------- Stage 4
if stage_done "$FINAL_DIR"; then
  echo "[skip] Stage 3B final refit already complete"
else
  echo "=== Stage 4: Stage 3B final refit (best_epoch=$BEST_EPOCH full-coverage epochs) ==="
  python -m methylation_predictor.rna_branch.cli train --config "$FINAL_CONFIG"
  mark_done "$FINAL_DIR"
fi

FINAL_CHECKPOINT="$FINAL_DIR/best.pt"

# ---------------------------------------------------------------- Stage 5
# run_full_evaluation.py has its OWN per-panel resume logic (skips any panel
# key already present in --output), so it's always safe/cheap to re-invoke --
# the check here must confirm all 4 required panel keys are present, not
# merely that the file exists (a file containing only double_ood, e.g. from
# an interrupted prior run, must NOT be treated as "stage complete").
full_eval_complete() {
  python -c "
import json, sys
try:
    d = json.load(open('$EVAL_DIR/full_evaluation.json'))
except FileNotFoundError:
    sys.exit(1)
required = {'double_ood', 'in_distribution', 'sample_ood', 'locus_ood'}
sys.exit(0 if required.issubset(d.keys()) else 1)
"
}
if full_eval_complete; then
  echo "[skip] Fase 7 full evaluation already complete"
else
  echo "=== Stage 5: Fase 7 full evaluation (official val_sample x val_cpg + 3 huge panels) ==="
  python scripts/rna_branch/run_full_evaluation.py \
    --config "$FINAL_CONFIG" --checkpoint "$FINAL_CHECKPOINT" \
    --output "$EVAL_DIR/full_evaluation.json" \
    --double-ood-predictions-h5 "/data/dataset/methylation/genomic_encoder_genome_wide_scratch/concat_genomewide_fullcoverage_seed17_v1_double_ood.h5"
fi

# ---------------------------------------------------------------- Stage 6
# stage_d5 also has its own per-comparison resume logic (partial saves after
# each of concat_vs_prior/concat_vs_bilinear/concat_vs_mp) -- same reasoning
# as Stage 5: the file existing doesn't mean all 3 comparisons are done.
bootstrap_complete() {
  python -c "
import json, sys
try:
    d = json.load(open('$EVAL_DIR/hierarchical_bootstrap.json'))
except FileNotFoundError:
    sys.exit(1)
required = {'concat_vs_prior', 'concat_vs_bilinear', 'concat_vs_mp'}
sys.exit(0 if required.issubset(d.get('paired_bootstrap', {}).keys()) else 1)
"
}
if bootstrap_complete; then
  echo "[skip] Fase 8/9 MP comparison + bootstrap already complete"
else
  echo "=== Stage 6: Fase 8/9 official-val bootstrap (concat vs prior/bilinear/MethylProphet) ==="
  python -m methylation_predictor.diagnostics.methylprophet.stage_d5_genome_wide_official_val \
    --concat-config "$FINAL_CONFIG" --concat-checkpoint "$FINAL_CHECKPOINT" \
    --bilinear-config "$BILINEAR_CONFIG" --bilinear-checkpoint "$BILINEAR_CHECKPOINT" \
    --mp-released "$MP_RELEASED" --empirical-prior "$EMPIRICAL_PRIOR" \
    --output-dir "$EVAL_DIR"
  python -c "
import json
result = json.load(open('$EVAL_DIR/hierarchical_bootstrap.json'))
json.dump(result['mp_comparison_chr1_only'], open('$EVAL_DIR/methylprophet_official_comparison.json', 'w'), indent=2, sort_keys=True, default=str)
"
fi

# ---------------------------------------------------------------- Stage 7
if [ -f "$EVAL_DIR/efficiency_benchmark.json" ]; then
  echo "[skip] Fase 10 efficiency benchmark already complete"
else
  echo "=== Stage 7: Fase 10 efficiency benchmark ==="
  python scripts/rna_branch/benchmark_vs_methylprophet.py \
    --concat-config "$FINAL_CONFIG" --concat-checkpoint "$FINAL_CHECKPOINT" \
    --bilinear-config "$BILINEAR_CONFIG" --bilinear-checkpoint "$BILINEAR_CHECKPOINT" \
    --mp-checkpoint "$MP_CHECKPOINT" \
    --output "$EVAL_DIR/efficiency_benchmark.json"
fi

# ---------------------------------------------------------------- Stage 8
echo "=== Stage 8: report assembly ==="
python scripts/rna_branch/assemble_run_summary.py --run-dir "$RUN_DIR" --best-epoch "$BEST_EPOCH"

echo "=== run complete ==="
