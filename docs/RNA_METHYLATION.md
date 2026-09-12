# RNA methylation model

The executable repository now has one paper-facing RNA runtime.

- Architecture and equations: [`MODEL.md`](MODEL.md)
- Train/evaluate commands: [`WORKFLOWS.md`](WORKFLOWS.md)
- Exact chr1 MethylProphet protocol: [`BENCHMARK_METHYLPROPHET.md`](BENCHMARK_METHYLPROPHET.md)

Current implementation:

```text
src/methylation_predictor/modeling/
src/methylation_predictor/rna_training/rna_methylation_trainer.py
configs/models/main.yaml
```

Historical shared-backbone, FeatureFusion, query-representation and earlier
two-stage residual implementations are intentionally not maintained as live
APIs. Their code remains available in Git history and frozen result ledgers.
