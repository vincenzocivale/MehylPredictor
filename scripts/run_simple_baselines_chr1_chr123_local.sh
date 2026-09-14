#!/bin/bash
# Sequential runner for the two "simple baselines" (cpg_prior,
# functional_global_rna_shift) on chr1 and chr123, on this machine
# (spyro) only.
#
# METHYL_DATA_ROOT is dune_data/ (repo-relative, gitignored), an sshfs
# mount of the shared dune raid mounted read-write on this machine --
# results are visible from every machine immediately, same as
# kingkong/ekko/dune.micc, no local mirroring needed.
#
# chr1 goes through scripts/paper_experiment.py (both recipes are already
# registered paper_facing in configs/experiment_surface.yaml). chr123 is
# not in configs/paper_studies.yaml's frozen matrix (same as every other
# chr123 run so far -- see docs/EXPERIMENT_LOG.md), so it goes straight
# through scripts/train.py + scripts/evaluate.py, matching the precedent
# set by the ad hoc chr123 runs already logged there.
#
# Single seed (17) only, per user direction 2026-09-13, same
# secondary/supplementary-study convention already used for
# functional_representation/mean_proxy on ekko/kingkong.
set -uo pipefail
cd "$(dirname "$0")/.."

source /mnt/hdd/vcivale/miniconda3/etc/profile.d/conda.sh
conda activate methyl-predictor
export METHYL_DATA_ROOT="$(pwd)/dune_data"
ROOT="$METHYL_DATA_ROOT"

fail=0

echo "=== cpg_prior / chr1 (paper_experiment.py) ==="
python3 scripts/paper_experiment.py \
  --profile configs/data/paper_chr1.yaml \
  --recipe configs/models/baselines/baseline_cpg_prior.yaml \
  --run-id cpg_prior__cpg_prior \
  --study cpg_prior --arm cpg_prior \
  --stage all --allow-dirty
[ $? -ne 0 ] && { echo "!!! cpg_prior/chr1 FAILED"; fail=1; }

echo "=== cpg_prior / chr123 (scripts/evaluate.py, ad hoc) ==="
# Adapter for LocusPriorCache's cpg_idx.npy/prior.npy naming contract.
# Uses derived/cpg_statistics/chr123 (the pure-Array official 93,104-CpG
# reference, all mu in (0,1)) -- NOT chr123_rna_universe, whose 5.4M rows
# are mostly auxiliary-only placeholders with target_mu=0 (see its
# manifest.json), which would violate LocusPriorCache's (0,1) contract
# and give meaningless values for anything outside the official CpGs.
CHR123_PRIOR_ADAPTER="$ROOT/derived/prior_cache_adapter/chr123"
mkdir -p "$CHR123_PRIOR_ADAPTER"
ln -sf "$ROOT/derived/cpg_statistics/chr123/cpg_idx.npy" "$CHR123_PRIOR_ADAPTER/cpg_idx.npy"
ln -sf "$ROOT/derived/cpg_statistics/chr123/target_mu.npy" "$CHR123_PRIOR_ADAPTER/prior.npy"
CHR123_CPG_PRIOR_RUN="$ROOT/experiments/runs/cpg_prior/chr123/cpg_prior__cpg_prior"
mkdir -p "$CHR123_CPG_PRIOR_RUN/evaluation/chr123"
python3 scripts/evaluate.py \
  --model cpg_prior \
  --eval-scope chr123 \
  --canonical-root "$ROOT/datasets/methylprophet_repro_v1" \
  --feature-cache "$CHR123_PRIOR_ADAPTER" \
  --registry "$ROOT/datasets/methylprophet_repro_v1/cpg/registries/array_cpg_map.parquet" \
  --output "$CHR123_CPG_PRIOR_RUN/evaluation/chr123"
[ $? -ne 0 ] && { echo "!!! cpg_prior/chr123 FAILED"; fail=1; }

echo "=== global_rna_shift / chr1 (paper_experiment.py) ==="
python3 scripts/paper_experiment.py \
  --profile configs/data/paper_chr1.yaml \
  --recipe configs/models/baselines/functional_global_rna_shift.yaml \
  --run-id functional_baselines__global_rna_shift__seed17 \
  --seed 17 \
  --study functional_baselines --arm global_rna_shift \
  --stage all --allow-dirty --resume
[ $? -ne 0 ] && { echo "!!! global_rna_shift/chr1 FAILED"; fail=1; }

echo "=== global_rna_shift / chr123 (scripts/train.py + evaluate.py, ad hoc) ==="
CHR123_GRS_RUN_ID="functional_baselines__global_rna_shift__seed17"
CHR123_GRS_RUN_DIR="$ROOT/experiments/runs/rna_methylation/chr123/$CHR123_GRS_RUN_ID"
python3 scripts/train.py \
  --model rna_methylation \
  --scope chr123 \
  --recipe configs/models/chr123/baseline_global_rna_shift_chr123.yaml \
  --mode final \
  --canonical-root "$ROOT/datasets/methylprophet_repro_v1" \
  --registry "$ROOT/datasets/methylprophet_repro_v1/cpg/registries/array_cpg_map.parquet" \
  --rna-cache "$ROOT/derived/methylprophet_table5_tcga_chr1/rna" \
  --cpg-targets-dir "$ROOT/derived/cpg_statistics/chr123_rna_universe" \
  --locus-store "$ROOT/derived/locus_features_v1" \
  --output-root "$ROOT/experiments" \
  --run-id "$CHR123_GRS_RUN_ID" \
  --seed 17 \
  --resume
if [ $? -ne 0 ]; then
  echo "!!! global_rna_shift/chr123 training FAILED"
  fail=1
else
  mkdir -p "$CHR123_GRS_RUN_DIR/evaluation/chr123"
  python3 scripts/evaluate.py \
    --model rna_methylation \
    --checkpoint "$CHR123_GRS_RUN_DIR/checkpoints/best.pt" \
    --eval-scope chr123 \
    --recipe configs/models/chr123/baseline_global_rna_shift_chr123.yaml \
    --canonical-root "$ROOT/datasets/methylprophet_repro_v1" \
    --registry "$ROOT/datasets/methylprophet_repro_v1/cpg/registries/array_cpg_map.parquet" \
    --rna-cache "$ROOT/derived/methylprophet_table5_tcga_chr1/rna" \
    --prior-cache "$CHR123_PRIOR_ADAPTER" \
    --cpg-targets-dir "$ROOT/derived/cpg_statistics/chr123_rna_universe" \
    --locus-store "$ROOT/derived/locus_features_v1" \
    --output "$CHR123_GRS_RUN_DIR/evaluation/chr123"
  [ $? -ne 0 ] && { echo "!!! global_rna_shift/chr123 evaluation FAILED"; fail=1; }
fi

if [ $fail -eq 0 ]; then
  echo ALL_SIMPLE_BASELINE_JOBS_DONE
else
  echo SOME_SIMPLE_BASELINE_JOBS_FAILED
fi
exit $fail
