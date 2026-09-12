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
two-stage residual implementations are not paper-facing APIs. Minimal compatibility
scaffolding may remain in the current branch until the protected J-series experiments finish;
the full research history remains available through Git history and frozen result ledgers.
