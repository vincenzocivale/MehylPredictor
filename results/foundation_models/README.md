# Foundation-model benchmark results

This directory contains the versioned, lightweight result artifacts for the
completed foundation-model runs. Reports and metadata are committed; the
large patient-by-CpG prediction parquet files remain in the local
`local_methyl_data/runs/foundation_models/` data store (ignored by Git).

| Scope | Completed models |
|---|---|
| `chr1` | MethylGPT, CpGPT, DeepCpG HCC DNA-only, DeepCpG HepG2 DNA-only |
| `chr123` | MethylGPT, CpGPT |
| `encode` | MethylGPT, CpGPT |

All runs use the `val_cpg_x_train_sample` view. DeepCpG is patient-agnostic;
its CpG-level prediction is replicated across training patients. DeepCpG
`chr123` was still running when this snapshot was created and is intentionally
not included.

The corresponding raw predictions are reproducible from the runners and are
available locally at:

```text
local_methyl_data/runs/foundation_models/<scope>/
```
