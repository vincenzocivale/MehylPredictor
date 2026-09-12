# CpG statistics and mean-proxy targets

`CpGStatisticsPredictor` is a retained auxiliary/static-locus workflow. It can
predict locus-level methylation mean and scale, while the RNA model itself is
trained separately.

For the paper-facing RNA model, the relevant artifact is the split-safe
training-locus mean target:

```text
derived/cpg_statistics/<scope>/
  cpg_idx.npy
  target_mu.npy
  official_train_mask.npy
  official_val_mask.npy
  ...
```

`RNAMethylationTrainer` uses `target_mu.npy` only for the training-only
mean-proxy objective. The scale target is not consumed by the current RNA
model.

The target builder can aggregate multiple methylation technologies while
recording source-specific observation counts. Official held-out Array patients
and held-out CpGs must not leak into training-locus supervision.

The historical genomic feature cache is not an RNA-model input anymore.
Prior-relative evaluation uses the separate metric-only `cpg_idx.npy +
prior.npy` cache described in `docs/WORKFLOWS.md`.
