# Data / Results V2 contract

The machine-readable source of truth is `configs/data/storage_v2.yaml`.

This phase only freezes the target contract. It does not move or delete data.

## Responsibilities

The Git repository stores code, recipes, compact JSON/JSONL/CSV paper results,
and manifests.

`METHYL_DATA_ROOT` stores canonical data, reference assets, derived caches,
checkpoints, histories, logs, evaluations, and archived historical run
metadata.

Machine-specific absolute paths are forbidden in curated paper results.

## Portable artifacts

Durable artifact references use `methyl-data://...`.

Example:

```text
methyl-data://experiments/runs/rna_methylation/chr1/main-seed17/checkpoints/best.pt
```

A machine resolves the suffix against its own `METHYL_DATA_ROOT`.

## Final data surface

The final chr1 runtime requires the canonical TCGA bundle, hg38 reference,
matched chr1 preparation, CpG statistics, functional regulatory atlas, and
annotation cache. RNA-comparator studies additionally require BulkFormer and
BulkRNABert caches.

## Final run namespace

New RNA runs target:

```text
experiments/runs/rna_methylation/<scope>/<run-id>/
```

Historical `locus_cls_joint` runs are not moved implicitly.

A run is considered complete only when `paper/record.json` exists.

## Curated results

The repository target is:

```text
results/paper/
├── registry.json
├── runs.jsonl
├── runs.csv
├── main/
├── functional_baselines/
├── mean_proxy/
├── rna_encoder/
└── cpg_prior/
```

Per-run JSON files contain metrics, provenance, and `methyl-data://` artifact
URIs. Heavy artifacts never enter Git.

## Cleanup

Historical derived roots can become deletion candidates only after a dependency
inventory proves that no frozen paper input or retained run record needs them.

Historical run directories are handled separately: metadata is slim-archived
before heavy artifacts are deleted.

## Multi-machine use

Each machine sets:

```bash
export METHYL_DATA_ROOT=<its mount of MethylPredictionData>
```

An optional local SSD work area is reserved via `METHYL_WORK_ROOT`.
