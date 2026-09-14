#!/usr/bin/env bash
# One-shot watcher: wait until no chr1-scope rna_methylation training is
# left running on this GPU (main-seed123-r3, mean_proxy__no_supervision__seed17,
# and whatever mean_proxy__no_branch__seed17 launches into once one of those
# two frees up -- see _watch_and_launch_no_branch.sh), then launch
# mean_proxy/no_supervision on chr123 (seed 17). Matches by command-line
# pattern rather than a fixed PID list so it queues behind whichever chr1
# job(s) are still running when this starts, including no_branch once it
# actually starts.
set -uo pipefail

PATTERN="scripts/train.py --model rna_methylation --scope chr1"

echo "$(date -u +%FT%TZ) watcher started, waiting for all '$PATTERN' processes to exit"

while pgrep -f "$PATTERN" >/dev/null 2>&1; do
  sleep 30
done

echo "$(date -u +%FT%TZ) no chr1-scope training left running, launching mean_proxy/no_supervision on chr123"

cd /home/vcivale/MehylPredictor || exit 1
source /home/vcivale/miniconda/etc/profile.d/conda.sh
conda activate methyl-predictor

exec python scripts/train.py \
  --model rna_methylation \
  --scope chr123 \
  --recipe configs/models/mean_contribution/no_mean_supervision_chr123.yaml \
  --mode final \
  --seed 17 \
  --run-id mean_proxy__no_supervision__seed17-chr123 \
  --canonical-root /dune/DATASETS/MethylPredictionData/datasets/methylprophet_repro_v1 \
  --registry /dune/DATASETS/MethylPredictionData/datasets/methylprophet_repro_v1/cpg/registries/array_cpg_map.parquet \
  --rna-cache /dune/DATASETS/MethylPredictionData/derived/methylprophet_table5_tcga_chr1/rna \
  --cpg-targets-dir /dune/DATASETS/MethylPredictionData/derived/cpg_statistics/chr123_rna_universe \
  --locus-store /dune/DATASETS/MethylPredictionData/derived/locus_features_v1 \
  --output-root /dune/DATASETS/MethylPredictionData/experiments \
  >> logs/mean_proxy__no_supervision__seed17-chr123.log 2>&1
