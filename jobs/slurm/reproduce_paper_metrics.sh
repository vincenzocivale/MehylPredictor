#!/bin/bash
# Recompute ICLR 2026 metrics from released prediction Parquet files.
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 <prediction_dir> <group_mapping_json> <output_dir> <encode|tcga>" >&2
    exit 2
fi

prediction_dir=$1
group_mapping_json=$2
output_dir=$3
dataset_name=$4

case "$dataset_name" in
    encode|tcga) ;;
    *) echo "dataset must be encode or tcga" >&2; exit 2 ;;
esac

python -m methylation_predictor.diagnostics.methylprophet.reproduce_paper_metrics \
    --input_result_df="$prediction_dir" \
    --input_group_idx_name_mapping_json="$group_mapping_json" \
    --output_dir="$output_dir" \
    --overwrite
