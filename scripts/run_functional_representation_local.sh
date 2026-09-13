#!/bin/bash
# Sequential local-storage runner for the functional_representation (E03)
# study on this machine only. Bypasses paper_worker.py's shared campaign
# (which hardcodes configs/data/paper_chr1.yaml) and instead uses
# configs/data/paper_chr1_local_ekko.yaml, whose heavy read inputs are
# mirrored on /mnt/hdd to avoid read failures from concurrent writes to the
# shared dune root by other machines. output_root stays on the shared mount
# so results are still visible everywhere and go through the normal
# results/paper/** git publication path (run scripts/collect_paper_runs.py
# and commit/push manually after each completed arm/seed, same as any
# other machine).
set -uo pipefail
cd "$(dirname "$0")/.."
source /home/vcivale/miniconda3/etc/profile.d/conda.sh
conda activate methyl-predictor
export METHYL_DATA_ROOT=/home/vcivale/dune_data

PROFILE=configs/data/paper_chr1_local_ekko.yaml

jobs=(
  "configs/models/functional_representation/basic_context.yaml basic_context"
  "configs/models/main.yaml full_functional"
  "configs/models/functional_representation/minimal.yaml minimal"
)
# Secondary/supplementary study: single seed for now (per user direction
# 2026-09-13), not the full 3-seed paper matrix.
seeds=(17)

for job in "${jobs[@]}"; do
  read -r recipe arm <<< "$job"
  for seed in "${seeds[@]}"; do
    run_id="functional_representation__${arm}__seed${seed}"
    echo "=== $run_id ==="
    python3 scripts/paper_experiment.py \
      --profile "$PROFILE" \
      --recipe "$recipe" \
      --run-id "$run_id" \
      --seed "$seed" \
      --study functional_representation \
      --arm "$arm" \
      --resume
    status=$?
    if [ $status -ne 0 ]; then
      echo "!!! $run_id FAILED (exit $status) -- continuing with next job"
    fi
  done
done
echo "ALL_FUNCTIONAL_REPRESENTATION_JOBS_DONE"
