# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Research framework for reconstructing DNA methylation from bulk RNA using frozen NTv3 CpG
representations. Two canonical trainable models share one genomic-scope axis:

- `CpGStatisticsPredictor` (`src/methylation_predictor/cpg_statistics/`): frozen NTv3 CpG embedding -> locus mean `mu` and logit-scale `sigma`.
- `RNAMethylationPredictor` (`src/methylation_predictor/models.py`, trained via `rna_training/`): RNA + CpG embedding + `(mu, sigma)` -> sample-specific methylation.
- Scopes: `chr1`, `chr123` (`chr1 ∪ chr2 ∪ chr3`), `genomewide`. `chr1`/`chr123` are MethylProphet-matched comparison scopes; `genomewide` is the primary general benchmark.

Read `README.md`, `docs/WORKFLOWS.md`, `docs/RNA_METHYLATION.md`, `docs/CPG_STATISTICS.md`, and
`docs/BENCHMARKS.md` before making architectural changes — they hold the current design rationale
and frozen reference numbers, not just usage instructions.

## Commands

Environment (a `methyl-predictor` conda env already exists on this machine with torch installed;
the sandboxed shell used for quick edits does not have torch):

```bash
conda activate methyl-predictor
python -m pip install -r requirements.txt
python -m pip install -r requirements-genomics.txt
python -m pip install -e .
```

Tests (pytest config lives in `pyproject.toml`; `pythonpath = ["src"]` means no editable install
is strictly required to run them):

```bash
pytest -q                                    # full suite
pytest tests/test_model.py -q                # one file
pytest tests/test_model.py::test_name -q     # one test
pytest -m "not slow" -q                      # skip tests that read multi-GB real TCGA slices
```

Most tests are pure-logic/synthetic-data and always run. Tests marked `@pytest.mark.slow` or that
use the `bundle`/`bundle_root` fixtures (`tests/conftest.py`) read the real canonical TCGA bundle
and auto-skip if `TCGA_CANONICAL_ROOT` (or `configs/data/tcga_canonical.yaml`'s `root:`) doesn't
resolve to an existing directory — expect those to skip outside the target machine.

Static checks:

```bash
python -m compileall src scripts
```

The four public entrypoints (all support `--model {cpg_statistics,rna_methylation}` except
`evaluate.py`/`prepare.py` where noted):

```bash
python scripts/prepare.py --model cpg_statistics --canonical-root ... --registry ... --scope genomewide --output ...
python scripts/prepare.py --model rna_methylation --checkpoint ... --targets ... --embeddings ... --output ...
python scripts/train.py --model rna_methylation --scope chr123 --recipe configs/models/rna_methylation.yaml ...
python scripts/tune.py --model rna_methylation --scope chr123 --lrs 2e-5,5e-5,8e-5 --schedulers constant,cosine_warmup ...
python scripts/evaluate.py --model rna_methylation --checkpoint /path/to/best.pt --eval-scope genomewide ...
```

Exact MethylProphet chr1 reproduction path (frozen, pair-complete):

```bash
python scripts/train.py --model rna_methylation --scope chr1 --engine matched_chr1 \
  --prepared-root ... --canonical-root ... --feature-cache ... --rna-cache ... \
  --registry ... --recipe configs/models/rna_methylation.yaml --output-root ...
```

which consumes caches built by `scripts/benchmark_methylprophet/prepare.py` (see
`docs/BENCHMARK_METHYLPROPHET.md`).

## Architecture

### Data layer is separate from everything else

`src/methylation_predictor/tcga_canonical/` is the *only* code that opens the raw TCGA HDF5/parquet
bundle (`/raid/DATASETS/MethylPredictionData/...` by default, resolved by
`tcga_canonical/config.py::resolve_bundle_root` — explicit arg > `TCGA_CANONICAL_ROOT` env var >
`configs/data/tcga_canonical.yaml`). It is strictly read-only and lazy/chunked (WGBS alone is
~23M CpGs; nothing here materializes a full matrix or a plain `id -> position` dict — see
`tcga_canonical/ids.py`). `cpg_statistics/`, `rna_training/`, and `benchmark/methylprophet/` all
read through this layer rather than touching HDF5 directly. See `docs/DATA.md` for the exact
artifact shapes/keys and the "never regenerate raw data" rules.

### Shared config dataclasses vs. per-path config loaders

`src/methylation_predictor/config.py` holds only the dataclasses genuinely shared across both
models: `EncoderConfig`, `InteractionConfig`, `ModelConfig`, `LossConfig`, `TrainingConfig`,
`TrackingConfig`. Each training path has its own thin loader on top of these:

- `rna_training/config.py::RNARecipe` / `load_rna_recipe` — the generic scoped pipeline's recipe format.
- `benchmark/methylprophet/config.py::RunConfig` / `load_config` — the MethylProphet-matched path's
  single-YAML format (`data:`/`model:`/`loss:`/`training:`/`tracking:` blocks), including
  `DataConfig`/`MatrixConfig`/`TableConfig` which exist only for that path.

When adding a config field, decide first whether it belongs in the shared dataclasses (read by
both trainers) or in one of the two path-specific loaders — don't add benchmark-only fields to the
shared `config.py`.

### The MethylProphet benchmark is isolated, not central

`benchmark/methylprophet/` (+ `scripts/benchmark_methylprophet/` + `configs/benchmark_methylprophet/`)
is a frozen, exact reproduction of the MethylProphet Table-5 chr1 benchmark — kept for paper-parity
validation, not as the repo's main architecture. It has its own `cache.py`/`feature_store.py`/
`probe.py`/`tracking.py`, deliberately not shared with the generic pipeline's own
`storage.py`/wandb wiring, because they evolved independently and merging them would be a behavior
change. `MethylProphetTrainer` (`benchmark/methylprophet/trainer.py`) is a complete, self-contained
Cartesian-block trainer — don't route generic-pipeline changes through it.

### Run/search output layout

`runs/<model>/<train-scope>/<run-id>/` and `searches/<model>/<scope>/<search-id>/` are the only
places training/tuning write to; both are gitignored (along with `artifacts/`, `checkpoints/`,
`wandb/`, `logs/`). Layout and provenance fields are defined in `run_store.py`. Only small
machine-readable reference numbers are version-controlled, under `results/reference/`
(`{cpg_statistics,rna_methylation}/{chr1,chr123,genomewide}.yaml` + `ablations.yaml`) —
`docs/BENCHMARKS.md` is the narrative index into them.

### Model compatibility note

`RNAMethylationPredictor` (in `models.py`) is a zero-diff subclass of `VarianceNormalizedResidualModel`,
kept so historical checkpoints load with identical state-dict keys — don't rename or add parameters
to it without checking checkpoint compatibility. `RNA2DNAmModel` (the flat, non-variance-normalized
residual model) is still live production code for `MethylProphetTrainer`, not dead/legacy.

### No legacy fallback path

There is no older training entrypoint left in this repo (the pre-refactor `data.py`/`trainer.py`/
`cli.py` Cartesian-batch path, and the ad-hoc `full_suite/` cache/probe helpers it depended on,
were removed). `scripts/{prepare,train,tune,evaluate}.py` are the only entrypoints; treat any
future one-off/experiment-specific script or config as something to delete once the experiment
concludes, not something to keep around as a second workflow.
