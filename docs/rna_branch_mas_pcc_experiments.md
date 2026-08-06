# MAS-PCC-oriented RNA-branch experiments

## Scientific target

The primary metric is **median-across-sample PCC (MAS-PCC)**: Pearson correlation
is computed across patients independently for each CpG, then aggregated with the
median across CpGs. In this repository it is reported both as `mas_pcc` and as the
backward-compatible `locus_dynamic_pearson_median`.

A locus-static prior cannot directly increase this correlation because subtracting
or adding the same value for every patient leaves a locus-wise PCC unchanged. The
new objectives therefore act on the patient-dependent branch while retaining a
small value-space loss for calibration.

## Implementation

`residual_loss` now supports three opt-in terms:

- `locus_pearson_weight`: mean `1 - PCC` across valid CpGs in a Cartesian batch;
- `locus_lower_tail_weight`: the same objective on the lowest-correlation fraction,
  intended to move the median rather than only improve already easy loci;
- `pairwise_difference_weight`: Huber loss on differences between patients at the
  same CpG, which cancels the static prior exactly.

The existing trainer already samples rectangular `sample_batch_size ×
cpg_batch_size` blocks. Stage MAS-PCC uses 128 patients per batch and 256 CpGs;
CpGs with fewer than 32 observed patients or effectively zero target/prediction
variance do not enter the correlation loss.

All new weights default to zero. Existing configurations preserve their historical
objective and checkpoint behavior.

## Screening runs

```bash
python -m pip install -e .
pytest -q tests/test_rna_branch_mas_pcc_losses.py

python -m methylation_predictor.rna_branch.grid \
  --grid configs/rna_branch/stage_mas_pcc_grid.yaml \
  --output-dir artifacts/rna_branch/generated_configs/stage_mas_pcc

# Local sequential smoke/screening, or submit the manifest as a Slurm array.
while read -r config; do
  python -m methylation_predictor.rna_branch validate --config "$config"
  python -m methylation_predictor.rna_branch train --config "$config"
done < artifacts/rna_branch/generated_configs/stage_mas_pcc/manifest.txt
```

The grid compares:

- P0: established F2 architecture with the historical MSE objective;
- P1: MSE + locus Pearson;
- P2: beta-Huber + locus Pearson;
- P3: P2 + pairwise patient differences;
- P4: P3 + lower-60% locus Pearson;
- P5: five-epoch historical MSE warm-up, then a correlation-dominant objective.

## Decision rule

Select checkpoints only by validation `mas_pcc`. Do not inspect test metrics during
screening. Promote a run to seeds 23 and 41 only when it improves validation
MAS-PCC over P0 and does not show pathological beta calibration.

For the promoted comparison report:

1. MAS-PCC and its per-CpG distribution;
2. fraction of finite CpGs with PCC > 0;
3. MAS-PCC by variability tertile;
4. overall and within-cancer dynamic correlation;
5. MSE, dynamic amplitude ratio, and calibration alpha as secondary diagnostics;
6. paired bootstrap confidence intervals over CpGs against P0 and MethylProphet.

A gain in global MAS-PCC without a gain after cancer-type residualization should be
interpreted as stronger between-cancer discrimination, not necessarily improved
patient-specific modelling.
