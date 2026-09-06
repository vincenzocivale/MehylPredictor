# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Research framework for reconstructing DNA methylation from bulk RNA using frozen NTv3 CpG
representations. Two canonical trainable models share one genomic-scope axis:

- `CpGStatisticsPredictor` (`src/methylation_predictor/cpg_statistics/`): frozen NTv3 CpG embedding -> locus mean `mu` and logit-scale `sigma`. Its `target_mu` output also feeds the RNA model's mean-branch proxy task below.
- `FeatureFusionArchitectureVariantModel` with `model.encoder.kind=locus_attention` (`src/methylation_predictor/models.py`, trained via `rna_training/locus_cls_trainer.py`, `scripts/train.py --engine matched_chr1_shared_backbone`, recipe `configs/models/rna_methylation_locus_attention.yaml`): the **primary/reference RNA-methylation architecture** as of 2026-09-05 — a single-stage shared backbone: RNA + CpG embedding -> two late-fused branches (a locus-only mean-prediction branch, a locus-conditioned RNA-attention branch) -> `beta_hat` directly, no explicit prior/residual composition, no post-fusion trunk. Superseded the `encoder.kind=linear` reference (`FeatureFusionLocusCLSModel`, selected 2026-09-03 via the `shared_backbone_locus_cls_2026_09` ablation ladder) once a follow-up chr1 isolation ladder showed locus-conditioned RNA attention is a decisive, consistent win over that linear-encoder model on every official view; see `docs/RNA_METHYLATION.md`'s "2026-09-05 update" section for the full evidence and open questions (a trunk-depth control, row C, was interrupted, not yet rerun). A multi-stream Hyper-Connections/Manifold-Constrained-HC (mHC) trunk kind was also tried as part of the architecture-novelty suite and **removed from the codebase entirely** (its contribution was small, within single-seed-noise range, and judged not worth the added complexity) — see git history before this removal if it needs revisiting; `TrunkConfig`/`FeatureFusionArchitectureVariantModel` now only support `kind="none"`/`"plain"`. **The RNA encoder is now the repo's primary axis of ongoing exploration** — see that section's "Forward direction" for the encoder-comparison harness and candidate directions (data-derived gene programs, pathway-informed projections, frozen foundation-model embeddings, deeper Perceiver-style designs) new work should target through `EncoderConfig.kind`/`build_rna_encoder()`, evaluated against the locus-attention reference above. `FeatureFusionLocusCLSModel` (`encoder.kind` hardcoded to `linear`) remains supported for old-checkpoint compatibility and as the frozen previous-reference recipe (`configs/models/rna_methylation_shared_backbone.yaml`), not where new recipes should target. The earlier two-stage frozen-prior + residual architecture (`RNAMethylationPredictor`/`VarianceNormalizedResidualModel`/`RNA2DNAmModel`/`ArchitectureVariantModel`, trained via the retired `MethylProphetTrainer`/`ScopedRNATrainer`) has been **removed**, including old-checkpoint compatibility for it; its frozen numbers remain under `results/reference/methylprophet_comparison/` and `results/reference/rna_methylation/chr1.yaml`'s `legacy_two_stage` field as historical provenance. See "Model compatibility note" below.
- Scopes: `chr1`, `chr123` (`chr1 ∪ chr2 ∪ chr3`), `genomewide`. `chr1` is the MethylProphet-matched
  comparison scope (official Array split independently verified against the released MethylProphet
  evaluation artifact, see `docs/BENCHMARK_METHYLPROPHET.md`); `genomewide` is the primary general
  benchmark, and the roadmap goes chr1 → genomewide directly. `chr123`'s CpG split is now verified
  exact against the released MethylProphet chr123 evaluation artifact (2026-09-02); its Array
  sample split remains a reconstruction (chr1's split, reused) rather than a literal release-ID
  match — the release's own sample_idx values don't fully overlap this repo's canonical Array
  bundle (306 IDs absent either way, a genuine snapshot/content difference), so the release split
  can't be applied here. No retrain was needed — the existing `cpg_statistics/chr123` and
  `rna_methylation/chr123` checkpoints already use the (unchanged) chr1-reused split. See
  `docs/BENCHMARK_METHYLPROPHET.md`'s "chr123: verified" section.

**The paper reports exactly three MethylProphet-matched settings — TCGA chr1 (done, verified),
TCGA chr1-3 (in progress, split verification pending), and ENCODE all-chromosome (not yet
built) — plus one further-out, not-yet-formalized TCGA-whole-genome+ENCODE-merged setting that is
explicitly not a paper-comparison result. `docs/PAPER_EXPERIMENTS.md` is the authoritative,
single source of truth for this: read it before adding any new baseline model or reporting a new
"vs MethylProphet" number — every new baseline must be trained/optimized independently in each of
the three settings, not tuned once and reused.**

Read `README.md`, `docs/WORKFLOWS.md`, `docs/RNA_METHYLATION.md`, `docs/CPG_STATISTICS.md`,
`docs/BENCHMARKS.md`, and `docs/PAPER_EXPERIMENTS.md` before making architectural changes — they
hold the current design rationale and frozen reference numbers, not just usage instructions.

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
pytest tests/test_architecture_variants.py -q                # one file
pytest tests/test_architecture_variants.py::test_name -q     # one test
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
python scripts/train.py --model rna_methylation --scope chr123 --engine shared_backbone \
  --recipe configs/models/rna_methylation_locus_attention.yaml --cpg-targets-dir ... ...
python scripts/tune.py --model rna_methylation --scope chr123 --cpg-targets-dir ... \
  --lrs 2e-5,5e-5,8e-5 --schedulers constant,cosine_warmup ...
python scripts/evaluate.py --model rna_methylation --checkpoint /path/to/best.pt --eval-scope genomewide --cpg-targets-dir ... ...
```

A fifth, read-only diagnostic entrypoint explains a trained `rna_methylation` checkpoint's
predictions rather than training/evaluating one -- see `docs/EXPLAINABILITY.md`:

```bash
python scripts/explain.py --checkpoint /path/to/best.pt --canonical-root ... --feature-cache ... \
  --rna-cache ... --sample-idx 1234 --cpg-idx-file candidate_cpg_ids.npy --auto-top-loci 20
```

Reference shared-backbone architecture, chr1 matched MethylProphet (primary architecture, see
`docs/RNA_METHYLATION.md`):

```bash
python scripts/train.py --model rna_methylation --scope chr1 --engine matched_chr1_shared_backbone \
  --prepared-root ... --canonical-root ... --feature-cache ... --rna-cache ... \
  --registry ... --cpg-targets-dir ... --recipe configs/models/rna_methylation_shared_backbone.yaml \
  --mode final --output-root ...

python scripts/evaluate.py --model rna_methylation --engine matched_chr1_shared_backbone \
  --checkpoint /path/to/last.pt --eval-scope chr1 --prepared-root ... --canonical-root ... \
  --feature-cache ... --rna-cache ... --registry ... --cpg-targets-dir ... --output ...
```

Both engines above consume caches built by `scripts/benchmark_methylprophet/prepare.py` (see
`docs/BENCHMARK_METHYLPROPHET.md`). The earlier two-stage architecture's exact `--engine
matched_chr1` reproduction path has been retired along with that architecture generation; its
frozen chr1 numbers remain under `results/reference/methylprophet_comparison/`.

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
`TrackingConfig`. `rna_training/config.py::RNARecipe` / `load_rna_recipe` is the sole recipe
loader on top of these, used by `LocusCLSJointTrainer` for both the chr1 matched preparation and
the scope-agnostic chr123/genome-wide path.

When adding a config field, decide first whether it belongs in the shared dataclasses (also read
by `cpg_statistics/`) or in `rna_training/config.py` — don't add RNA-methylation-only fields to
the shared `config.py`.

### The MethylProphet chr1 data preparation is isolated, not central

`benchmark/methylprophet/` (+ `scripts/benchmark_methylprophet/prepare.py` +
`configs/benchmark_methylprophet/`) builds the exact chr1 MethylProphet-matched cache
(`cache.py`/`feature_store.py`/`probe.py`/`protocol.py`), deliberately not shared with the
generic pipeline's own `storage.py`/wandb wiring, because they evolved independently and merging
them would be a behavior change. The `LocusCLSJointTrainer` engine (`--engine
matched_chr1_shared_backbone`) consumes this cache but is otherwise defined in `rna_training/`,
not here. The earlier exact two-stage-architecture reproduction path (`MethylProphetTrainer` and
its own `scripts/benchmark_methylprophet/{run_experiment,analyze_context,resolve_final_epoch_budget}.py`/
`run.sh`) has been retired along with that architecture generation; its frozen numbers remain
under `results/reference/methylprophet_comparison/`.

### The foundation-model masked-CpG comparison is isolated too

`benchmark/foundation_models/` (+ `scripts/benchmark_foundation_models/` +
`configs/benchmark_foundation_models/`) feeds the same `val_cpg_x_train_sample` view
to external pretrained masked-methylation models (CpGPT, MethylGPT, DeepCpG) — see
`docs/PAPER_EXPERIMENTS.md`'s foundation-model section for status/known blockers,
including a correction of an earlier (wrong) assumption about how DeepCpG was used
as a baseline in the reference papers — read that section before assuming any of
this pipeline's numbers reproduce a specific published comparison.
Vendored external repos + their checkpoints live in `external/` (gitignored,
`scripts/benchmark_foundation_models/setup.sh` recreates it) — each model gets its
own isolated environment there rather than adding its framework deps
(hydra/lightning/torchtext/legacy tensorflow+keras/...) to this repo's own
`requirements*.txt`; the three models' dependency stacks are themselves mutually
incompatible (MethylGPT's `torchtext` has no build compatible with this repo's main
torch install; DeepCpG needs a legacy `python=3.7`/`tensorflow==1.13.1` conda env,
not a venv), which is
exactly the kind of divergence this isolation pattern exists to contain.

### Run/search output layout

`runs/<model>/<train-scope>/<run-id>/` and `searches/<model>/<scope>/<search-id>/` are the only
places training/tuning write to; both are gitignored (along with `artifacts/`, `checkpoints/`,
`wandb/`, `logs/`). Layout and provenance fields are defined in `run_store.py`. Only small
machine-readable reference numbers are version-controlled, under `results/reference/`, split by
role: `{cpg_statistics,rna_methylation}/{chr1,chr123,genomewide}.yaml` are the two models' own
headline results; `baselines/<name>/{chr1,chr123,genomewide}.yaml` (`cpg_prior`, `global_rna_shift`,
`bilinear_rna_cpg`, `mlp_rna_cpg` — see `docs/PAPER_EXPERIMENTS.md`'s "Baseline models" section)
are simplified-architecture baselines trained/tuned independently per setting, schema-compatible
with the `rna_methylation/` files; `methylprophet_comparison/` holds head-to-head comparisons
against the published MethylProphet paper (Table 5 mixed-source, Table 7 per-source rows, and the
baseline comparison tables) — a paper-comparison claim, not an internal ablation; `ablations.yaml`
holds internal design/hyperparameter ablations only (prior choice, training search,
architecture-simplification sweeps). `docs/BENCHMARKS.md` is the narrative index into all four.

### Model compatibility note

`FeatureFusionArchitectureVariantModel` with `encoder.kind=locus_attention` is the reference
RNA-methylation architecture going forward (see "What this repo is" above) — new work should
target it, not `FeatureFusionLocusCLSModel` below. `FeatureFusionLocusCLSModel` (`encoder.kind`
hardcoded to `linear`) is the previous (2026-09-03 to 2026-09-05) reference, kept live for
old-checkpoint compatibility and as a frozen comparison point, not for new work.

The earlier two-stage frozen-prior + residual architecture generation
(`RNAMethylationPredictor`, `VarianceNormalizedResidualModel`, `RNA2DNAmModel`,
`ArchitectureVariantModel`, `DirectPredictionModel`, and their trainers `MethylProphetTrainer`/
`ScopedRNATrainer`) has been **removed entirely, with no old-checkpoint compatibility retained**.
Checkpoints from that generation cannot be loaded by current code. Its frozen paper-comparison
numbers remain under `results/reference/methylprophet_comparison/` and
`results/reference/rna_methylation/chr1.yaml`'s `legacy_two_stage` field as historical provenance.
Three things that depended on it were ported to the current architecture rather than removed:
the three simplified baselines other than CpG Prior (Global RNA Shift/Bilinear RNA-CpG/MLP
RNA-CpG, now expressed via `FeatureFusionArchitectureVariantModel`'s `use_mean_branch`/
`include_raw_rna`/`include_raw_cpg`/`use_raw_product` constructor kwargs — see
`docs/PAPER_EXPERIMENTS.md`), explainability (`scripts/explain.py`, now attributing
`residual_logit` instead of the old `raw_delta` — see `docs/EXPLAINABILITY.md`), and
hyperparameter tuning (`scripts/tune.py`, now built on `LocusCLSJointTrainer`).

### No legacy fallback path

There is no older training entrypoint left in this repo (the pre-refactor `data.py`/`trainer.py`/
`cli.py` Cartesian-batch path, and the ad-hoc `full_suite/` cache/probe helpers it depended on,
were removed; more recently, the entire two-stage frozen-prior + residual architecture generation
and its `--engine generic`/`matched_chr1` CLI paths were removed too — see "Model compatibility
note" above). `scripts/{prepare,train,tune,evaluate,explain}.py` are the only entrypoints; treat
any future one-off/experiment-specific script or config as something to delete once the
experiment concludes, not something to keep around as a second workflow.
