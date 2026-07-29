# Methylation predictor audit

This repository has two deliberately separate domains:

- `methylation_predictor.diagnostics.methylprophet` diagnoses the upstream MethylProphet release through its public outputs and a read-only upstream dependency.
- `methylation_predictor.genomic_encoder` evaluates an independent genomic encoder and consumes diagnostics only through the documented export contract.

The upstream source is pinned at `third_party/MethylProphet`. Generated inputs and outputs belong below the ignored `artifacts/` tree.

Install the local package before invoking the documented module CLIs:

```bash
python -m pip install -e .
```
