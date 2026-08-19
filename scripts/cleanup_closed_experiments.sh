#!/usr/bin/env bash
set -euo pipefail

# Run only after the refactored tests and one real-data smoke run pass.
# Git history remains the archive for all removed development experiments.
FILES=(
  configs/protocols/ablations/tcga_mix_chr123_array_heavy.yaml
  configs/protocols/ablations/tcga_mix_chr123_equal_source.yaml
  configs/protocols/ablations/tcga_mix_chr123_proportional_to_measurements.yaml
  configs/protocols/ablations/tcga_mix_chr1_array_heavy.yaml
  configs/protocols/ablations/tcga_mix_chr1_equal_source.yaml
  configs/protocols/ablations/tcga_mix_chr1_proportional_to_measurements.yaml
  configs/tcga_chr1/experiments/array_only_structured.yaml
  configs/tcga_chr1/experiments/large_sample_pcc.yaml
  configs/tcga_chr1/experiments/tail_aware_pcc.yaml
  configs/tcga_chr1/v2_locus_pcc.yaml
  scripts/prepare_array_chr1_ablation.py
  docs/TCGA_CHR1_EXPERIMENTS.md
)
for path in "${FILES[@]}"; do
  if git ls-files --error-unmatch "$path" >/dev/null 2>&1; then
    git rm "$path"
  fi
done

echo "Closed experiment files staged for deletion. Review with: git diff --cached --stat"
