#!/usr/bin/env bash
# One-shot watcher: wait until one of the two currently-running GPU0
# training processes exits, then launch mean_proxy/no_branch/seed17
# (crashed earlier with CUDA OOM while 3 trainings shared GPU0).
set -uo pipefail

PID_MAIN=3299507       # main-seed123-r3 train.py
PID_NO_SUP=3353032     # mean_proxy__no_supervision__seed17 train.py

echo "$(date -u +%FT%TZ) watcher started, waiting on PID_MAIN=$PID_MAIN or PID_NO_SUP=$PID_NO_SUP"

while kill -0 "$PID_MAIN" 2>/dev/null && kill -0 "$PID_NO_SUP" 2>/dev/null; do
  sleep 30
done

echo "$(date -u +%FT%TZ) one of the watched trainings exited, launching no_branch"

cd /home/vcivale/MehylPredictor || exit 1
source /home/vcivale/miniconda/etc/profile.d/conda.sh
conda activate methyl-predictor

exec python scripts/paper_experiment.py \
  --profile configs/data/paper_chr1.yaml \
  --data-root /dune/DATASETS/MethylPredictionData \
  --recipe configs/models/mean_contribution/no_mean_branch.yaml \
  --run-id mean_proxy__no_branch__seed17 \
  --seed 17 --study mean_proxy --arm no_branch \
  --stage all --allow-dirty \
  >> logs/mean_proxy__no_branch__seed17.log 2>&1
