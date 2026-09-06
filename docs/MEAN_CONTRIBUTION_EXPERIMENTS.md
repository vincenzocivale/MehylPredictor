# Mean-contribution experiment — paper protocol

This study isolates what the locus-mean proxy task contributes to the current
RNA→DNA-methylation reference model. It deliberately does **not** change the
RNA encoder, token count, CpG embedding, optimizer, beta loss, data sources,
batching, or official split.

## Claims and matching experiments

1. **Biological proxy supervision improves final prediction on unseen CpGs.**
   Compare `full_reference` vs `no_mean_supervision`.
2. **The effect is not merely extra branch capacity.**
   Compare `no_mean_supervision` vs `no_mean_branch`.
3. **The proxy makes `h_mean` methylation-aware.**
   Fit a linear probe using train CpGs only; evaluate it on official val CpGs.
4. **The mean primarily corrects absolute locus calibration.**
   Report MSE and locus-level bias as primary metrics; MAS-PCC is secondary.
5. **The biological motivation exists in the data.**
   Decompose methylation variance into between-locus mean variance and
   within-locus patient variance on the official training matrix.

## Arms

| arm | mean branch | mean auxiliary loss | RNA encoder | product |
|---|---:|---:|---|---:|
| `full_reference` | yes | 0.15 | locus attention K=64 | no |
| `no_mean_supervision` | yes | 0 | locus attention K=64 | no |
| `no_mean_branch` | no | — | locus attention K=64 | no |

Seeds: **17, 29, 43**.

The full-reference run ids intentionally reuse the already-launched
`ref-locus-attn-k64-noproduct-seed{seed}` runs.

## Launch

```bash
conda activate methyl-predictor
export METHYL_DATA_ROOT=/dune/DATASETS/MethylPredictionData

# Distribute these two arms across free GPUs/machines.
python scripts/experiments/run_mean_contribution.py \
  --arms no_mean_supervision --gpu 0

python scripts/experiments/run_mean_contribution.py \
  --arms no_mean_branch --gpu 1

# Reuse reference training and fill only missing eval/diagnostics.
python scripts/experiments/run_mean_contribution.py \
  --arms full_reference --gpu 2
```

The runner is idempotent: it resumes training when possible and skips
training/evaluation/diagnostics that already exist.

## Dataset variance decomposition

No checkpoint or GPU is required:

```bash
python scripts/experiments/analyze_mean_contribution.py dataset \
  --canonical-root "$METHYL_DATA_ROOT/datasets/methylprophet_repro_v1" \
  --prepared-root "$METHYL_DATA_ROOT/derived/methylprophet_table5_tcga_chr1" \
  --output "$METHYL_DATA_ROOT/experiments/analysis/mean_contribution_2026_09/dataset_variance.json"
```

The primary decomposition is `train_cpg × train_sample`, so the biological
motivation does not inspect the held-out evaluation matrix. A
`val_cpg × train_sample` confirmation is saved separately.

## Collect paper results

After the runs finish:

```bash
python scripts/experiments/collect_mean_contribution.py \
  --dataset-diagnostics \
  "$METHYL_DATA_ROOT/experiments/analysis/mean_contribution_2026_09/dataset_variance.json"
```

Generated, version-controlled outputs:

```text
results/reference/mean_contribution_2026_09/
  summary.yaml
  summary.md
  paper_table.csv
  variance_decile_effects.csv
  dataset_variance.json
  runs/
    full_reference__seed17.json
    ...
```

Each per-run JSON stores recipe SHA-256, resolved-config SHA-256, checkpoint
SHA-256, checkpoint epoch, seed, run id, all official views, mean-head
accuracy, `h_mean` linear-probe performance, locus-bias metrics, and
variance-decile metrics.

## Statistical reporting

Use paired seeds.

- `full_reference` vs `no_mean_supervision` = **proxy-task supervision**.
- `no_mean_supervision` vs `no_mean_branch` = **branch capacity**.
- `full_reference` vs `no_mean_branch` = **total mean contribution**.

For unseen-CpG views, emphasize percentage MSE reduction and percentage
reduction in median absolute locus bias. Keep MAS-PCC visible, but do not use
it as the primary evidence for a CpG-specific mean effect because Pearson
correlation across samples is invariant to a CpG-wise constant shift.
