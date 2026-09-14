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
# methyl-predictor is a shared conda env on ekko (other users too, e.g.
# lbenedet); a pip upgrade to torch 2.14.0/cu13 there on 2026-09-13 21:56
# broke CUDA against this machine's driver (550.163.01, max CUDA 12.4) for
# every user, mid-way through basic_context/seed17's run (unaffected only
# because its process already had the old torch loaded in memory).
# methyl-predictor-ekko is an isolated clone with torch pinned back to
# 2.4.1+cu121, used only by this repo's local runs so we no longer collide
# with -- or get broken by -- other users' changes to the shared env.
conda activate methyl-predictor-ekko
export METHYL_DATA_ROOT=/home/vcivale/dune_data

PROFILE=configs/data/paper_chr1_local_ekko.yaml

jobs=(
  "configs/models/functional_representation/basic_context.yaml basic_context"
  "configs/models/functional_representation/minimal.yaml minimal"
)
# full_functional (configs/models/main.yaml) is deliberately NOT trained
# here: it is the exact same recipe+seed+scope as main-seed17-r3, already
# running on kingkong as part of the `main` study. Once that run completes,
# reuse its checkpoint/metrics for functional_representation/full_functional
# /seed17 (evaluate+record only, no retraining) instead of duplicating ~16h
# of GPU work. See docs/EXPERIMENT_LOG.md.
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
