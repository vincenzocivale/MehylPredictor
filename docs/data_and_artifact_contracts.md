# Contratti dati e artefatti

```text
MethylProphet upstream → diagnostics export contract → genomic encoder evaluation
```

L'export diagnostico contiene `cpg_train_prior.parquet`, `mp_dynamic_component.parquet`, `mp_mean_rna_prior.parquet`, `sample_metadata.parquet`, `cpg_split_manifest.parquet`, `diagnostic_metrics.json` e `manifest.json`.

Ogni run usa `artifacts/<domain>/<experiment>/<run_id>/` con `config.yaml`, `manifest.json`, `metrics.json`, eventuale `selection.json` e `predictions.parquet` soltanto se necessario. Il manifest registra commit root/upstream, revisione NTv3, comando, hash input, seed, split, righe, ambiente, timestamp e stato.
