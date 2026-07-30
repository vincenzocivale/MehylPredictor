# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

An audit/evaluation project with two deliberately separate domains (see `AGENTS.md`, `README.md`):

- `methylation_predictor.diagnostics.methylprophet` — diagnoses the upstream **MethylProphet** paper/release through its public outputs and a read-only upstream git submodule (`third_party/MethylProphet`).
- `methylation_predictor.genomic_encoder` — evaluates an independent genomic encoder (NTv3). It must consume diagnostics **only** through the documented export contract, never by reading upstream shards/MDS/checkpoints directly.

These two domains must stay decoupled. Don't add code that lets `genomic_encoder` import from `diagnostics.methylprophet` (or vice versa) outside the export contract in `artifacts/diagnostics/methylprophet/export/`.

## Setup and commands

```bash
python -m pip install -e .          # install local package (src layout, package = methylation_predictor)
pytest -q                            # default run: excludes gpu/data/regression-marked tests (see pyproject addopts)
pytest tests/unit/genomic_encoder/test_ntv3_readout.py -q                       # single file
pytest tests/unit/genomic_encoder/test_ntv3_readout.py::NTv3PriorTest::test_output_mapping_and_local_pooling -q  # single test
pytest -m regression -q              # opt-in: expensive reproducibility regression tests
pytest -m "gpu or data" -q           # opt-in: tests needing a GPU or external data artifacts
```

`pyproject.toml` sets `pythonpath = ["src"]` for pytest, so `src/` does not need to be on `PYTHONPATH` manually when running pytest from the repo root. No lint/type-check tooling (ruff/black/mypy) is configured in this repo.

Both domains expose an argparse CLI dispatched by subcommand:

```bash
python -m methylation_predictor.diagnostics.methylprophet.cli <released-audit|locus-decomposition|gene-intervention|empirical-hybrid|mean-rna-baseline|export> ...
python -m methylation_predictor.genomic_encoder.cli <build-features|static-baselines|reference-neighborhood|extract-ntv3|prior-probe|variability-probe|downstream-evaluation> ...
```

Each subcommand's own `--flags` are the real interface (see each module's `argparse` block). The YAML files under `configs/diagnostics/` and `configs/genomic_encoder/` are **not** loaded by any code — they are human-readable records of an experiment's identity/hyperparameters, not `--config` inputs. Don't assume a CLI reads its matching YAML; check the module's `parser()`/`main()` for actual flags.

## The upstream boundary

`third_party/MethylProphet` is a pinned, read-only git submodule (pinned commit recorded in `docs/reproducibility.md`). Its worktree must always stay clean.

- The **only** file allowed to touch it is `src/methylation_predictor/diagnostics/methylprophet/upstream.py`. It provides `import_upstream()` (adds the submodule to `sys.path` without copying source) and `assert_clean()` (returns the pinned commit hash, raises if the submodule worktree is dirty).
- For PCC, always delegate to the upstream `src/eval.py:compute_pcc_by_group` via this boundary — never reimplement the metric. Any out-of-core/chunking logic may only partition input data, not redefine the metric.
- Released prediction rows (`group_idx`, `cpg_idx`, `sample_idx`, `pred_methyl`, `gt_methyl`) are the authoritative evidence of which split was actually evaluated — prefer them over inferring splits some other way.
- Keep three sources separate and never silently substitute one for another: paper-reported values, author-provided log files, and recomputed metrics from released predictions. State explicitly which of these a result reproduces (paper table, author logs, both, or neither) — see `docs/methylprophet_diagnosis.md` for the existing example (recomputed TCGA metrics match author `log_dict`, not the ICLR table; that gap is intentional and documented, not a bug to fix).

## Export contract (diagnostics → genomic_encoder)

`diagnostics/methylprophet/export_contract.py` is the sole, validated hand-off point. It copies and schema-checks five parquet files (`cpg_train_prior.parquet`, `mp_dynamic_component.parquet`, `mp_mean_rna_prior.parquet`, `sample_metadata.parquet`, `cpg_split_manifest.parquet`) plus `diagnostic_metrics.json` into an output dir, and writes a `manifest.json` recording `{upstream_commit, rows, contract_version}`. Required columns per file are the `REQUIRED` dict at the top of that module — treat it as the schema source of truth, not `docs/data_and_artifact_contracts.md` (which is a prose summary and can drift).

## Artifacts layout

`artifacts/` is gitignored (generated/downloaded, never commit under it). Convention: `artifacts/<domain>/<experiment>/<run_id>/` containing `config.yaml`, `manifest.json`, `metrics.json`, and optionally `selection.json` / `predictions.parquet`. `artifacts/cache/` holds large downloaded matrices/checkpoints/shards/FASTA/embeddings — never read upstream cache files from `genomic_encoder` code. Manifests should record: root/upstream commit, NTv3 revision, command, input hashes, seed, split, row counts, environment, timestamp, status.

## `common/` package

`src/methylation_predictor/common/` holds tiny, single-purpose, dependency-light helpers shared by both domains: `schemas.py` (column presence checks), `splits.py` (split-disjointness invariant), `metrics.py` (mse/mae), `manifests.py` (JSON manifest writer), `hashing.py` (sha256), `io.py` (parquet reader). Each file is ~10-15 lines by design — extend in the same terse style rather than growing a generic utils module.

## Genomic encoder decision (already settled — don't relitigate without new evidence)

Per `docs/decisions/genomic_encoder_selection.md`: backbone NTv3-650M-post, **frozen**, hg38, 32,768 bp context, forward orientation, readout = mean of final per-base embedding at the central C/G. Local pooling was rejected (didn't improve readout), reference-neighborhood features are optional/non-core, and fine-tuning on chr1 was rejected as unjustified. Known limitation: chr1-only exposure, and NTv3 was exposed to Borzoi post-training data.

## `MethylProphetTest-rna-experiments/`

This top-level directory is an **untracked, unmerged overlay** (see its own `PATCH_NOTES.md` / `INSTALL_RNA_BRANCH.md`) — a staged RNA-branch experiment package meant to be `rsync`'d into the repo root (`src/methylation_predictor/rna_branch/`, `configs/rna_branch/`, etc.) but not yet integrated. Don't assume `methylation_predictor.rna_branch` exists in `src/` — it doesn't yet. Treat this directory as pending work-in-progress, not part of the current architecture, unless the user is actively merging it.

## Documentation language

Most `docs/*.md` files (data contracts, diagnosis, evaluation, reproducibility) are written in Italian; `docs/decisions/genomic_encoder_selection.md` is in English. Match the existing language when editing a given doc rather than converting it.
