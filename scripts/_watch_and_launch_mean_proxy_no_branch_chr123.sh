#!/usr/bin/env bash
# One-shot watcher: wait until no chr1-scope rna_methylation training AND no
# chr123 mean_proxy/no_supervision training are left running on this GPU,
# then launch mean_proxy/no_branch on chr123 (seed 17). Matches by
# command-line pattern rather than fixed PIDs, so it naturally queues behind
# whatever is still running when this starts, including
# mean_proxy__no_supervision__seed17-chr123 (queued separately via
# _watch_and_launch_mean_proxy_no_supervision_chr123.sh) so the two chr123
# arms run sequentially instead of sharing the single GPU.
#
# mean_proxy/no_branch was NOT relaunched on chr1 (crashed twice with CUDA
# OutOfMemoryError, 2026-09-13 and again 2026-09-14 -- see
# docs/EXPERIMENT_LOG.md); per user direction 2026-09-14, chr123 is wanted
# instead of a third chr1 attempt.
set -uo pipefail

PATTERN_CHR1="scripts/train.py --model rna_methylation --scope chr1"
PATTERN_CHR123_NOSUP="scope chr123.*no_mean_supervision_chr123"

echo "$(date -u +%FT%TZ) watcher started, waiting for '$PATTERN_CHR1' and '$PATTERN_CHR123_NOSUP' to exit"

while pgrep -f "$PATTERN_CHR1" >/dev/null 2>&1 || pgrep -f "$PATTERN_CHR123_NOSUP" >/dev/null 2>&1; do
  sleep 30
done

echo "$(date -u +%FT%TZ) GPU free of chr1 and chr123/no_supervision training, launching mean_proxy/no_branch on chr123"

cd /home/vcivale/MehylPredictor || exit 1
source /home/vcivale/miniconda/etc/profile.d/conda.sh
conda activate methyl-predictor

exec python scripts/train.py \
  --model rna_methylation \
  --scope chr123 \
  --recipe configs/models/mean_contribution/no_mean_branch_chr123.yaml \
  --mode final \
  --seed 17 \
  --run-id mean_proxy__no_branch__seed17-chr123 \
  --canonical-root /dune/DATASETS/MethylPredictionData/datasets/methylprophet_repro_v1 \
  --registry /dune/DATASETS/MethylPredictionData/datasets/methylprophet_repro_v1/cpg/registries/array_cpg_map.parquet \
  --rna-cache /dune/DATASETS/MethylPredictionData/derived/methylprophet_table5_tcga_chr1/rna \
  --cpg-targets-dir /dune/DATASETS/MethylPredictionData/derived/cpg_statistics/chr123_rna_universe \
  --locus-store /dune/DATASETS/MethylPredictionData/derived/locus_features_v1 \
  --output-root /dune/DATASETS/MethylPredictionData/experiments \
  >> logs/mean_proxy__no_branch__seed17-chr123.log 2>&1
