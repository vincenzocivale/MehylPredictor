# Methylation predictor audit

This repository has three deliberately separate domains:

- `methylation_predictor.diagnostics.methylprophet` diagnoses the upstream MethylProphet release through its public outputs and a read-only upstream dependency.
- `methylation_predictor.genomic_encoder` evaluates an independent genomic encoder and consumes diagnostics only through the documented export contract.
- `methylation_predictor.rna_encoder_quality` audits frozen transcriptomic representations without loading methylation targets, CpG features or a downstream methylation regressor.

The upstream source is pinned at `third_party/MethylProphet`. Generated inputs and outputs belong below the ignored `artifacts/` tree.

Install the local package before invoking the documented module CLIs:

```bash
python -m pip install -e .
```

Run the RNA encoder audit:

```bash
python -m methylation_predictor.rna_encoder_quality validate \
  --config configs/rna_encoder_quality/bulkrnabert_tcga.yaml
python -m methylation_predictor.rna_encoder_quality run \
  --config configs/rna_encoder_quality/bulkrnabert_tcga.yaml
```

See [`docs/rna_encoder_quality.md`](docs/rna_encoder_quality.md) for cleanup, token-level analysis, technical stability views and interpretation.

Run the RNA encoder audit:

```bash
python -m methylation_predictor.rna_encoder_quality validate \
  --config configs/rna_encoder_quality/bulkrnabert_tcga.yaml
python -m methylation_predictor.rna_encoder_quality run \
  --config configs/rna_encoder_quality/bulkrnabert_tcga.yaml
```

See [`docs/rna_encoder_quality.md`](docs/rna_encoder_quality.md) for cleanup, token-level analysis, technical stability views and interpretation.
