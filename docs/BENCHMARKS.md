# Benchmarks

Machine-readable numbers live in `results/reference/`; this document is the
narrative index into them.

See [`PAPER_EXPERIMENTS.md`](PAPER_EXPERIMENTS.md) for the three canonical MethylProphet-matched
paper settings (TCGA chr1, TCGA chr123, ENCODE genomewide) and their status — read that first if
you're adding a new baseline or a new "vs MethylProphet" comparison.

## Models and scopes

Two trainable models, three genomic scopes:

- `CpGStatisticsPredictor` → `results/reference/cpg_statistics/{chr1,chr123,genomewide}.yaml`
- RNA-methylation model → `results/reference/rna_methylation/{chr1,chr123,genomewide}.yaml`. As of
  2026-09-03, chr1's file records the new primary/reference architecture
  (`FeatureFusionLocusCLSModel`, shared-backbone -- see `docs/RNA_METHYLATION.md`); chr123 and
  genomewide still record the earlier two-stage architecture (`RNAMethylationPredictor`, since
  removed from the codebase -- see CLAUDE.md's "Model compatibility note") pending a
  shared-backbone rerun at those scopes. chr1's file keeps the two-stage number too, under
  `legacy_two_stage`, for comparison.

`chr1` is the MethylProphet-matched comparison scope -- its official Array
split is independently verified against the actual released MethylProphet
evaluation artifact, exact ID-set match (see
[`BENCHMARK_METHYLPROPHET.md`](BENCHMARK_METHYLPROPHET.md)). `genomewide` is
the primary general benchmark; the roadmap goes chr1 → genomewide directly.
`chr123` (`chr1 ∪ chr2 ∪ chr3`) remains a usable general scope in the
pipeline, but is **not currently presented as a verified MethylProphet-matched
comparison**: its CpG-axis split provenance has not been through the same
direct verification as chr1's, and access to a candidate source dataset for
that verification is still being pursued (see
[`BENCHMARK_METHYLPROPHET.md`](BENCHMARK_METHYLPROPHET.md)'s "chr123: not
verified" note).

## Current reference results

### `rna_methylation`

| scope | train-CpG × val-sample | val-CpG × train-sample | val-CpG × val-sample | architecture |
|---|---:|---:|---:|---|
| chr1 | 0.6431 / 0.01351 | 0.6081 / 0.01613 | **0.5627** / 0.01702 | shared-backbone (primary) |
| chr1 (legacy) | 0.6327 / 0.01278 | 0.5984 / 0.01874 | 0.5613 / 0.01941 | two-stage |
| chr123 | 0.5468 / 0.01592 | 0.5492 / 0.02198 | 0.5034 / 0.02287 | two-stage |
| genomewide | 0.5889 / 0.01482 | 0.5800 / 0.02099 | 0.5244 / 0.02223 | two-stage |

Cells are `MAS-PCC / MSE`. chr1's shared-backbone row (`FeatureFusionLocusCLSModel`, the new
primary architecture, see `docs/RNA_METHYLATION.md`) is a single-seed run stopped at epoch 47/80
by user request, not yet at convergence -- a lower bound, not this architecture's final number
(`results/reference/rna_methylation/chr1.yaml`). chr123/genomewide are still on the earlier
two-stage architecture (`RNAMethylationPredictor`, since removed) pending a shared-backbone
rerun. Genome-wide
per-chromosome mean MAS-PCC (two-stage): 0.5196 (observed range 0.459–0.579). LR 5e-5, constant
scheduler, 80 epochs, seed 17 for all rows. chr123 (2026-08-31) is not a MethylProphet-matched
result -- see the scope note above -- and required extending the generic engine's feature cache to
the full multi-technology CpG universe (see `results/reference/PROVENANCE.md`).

### `cpg_statistics`

Joint mu/sigma ensemble model (retrained 2026-08-28 after fixing a real bug where
`cpg_statistics/{trainer,evaluator,export}.py` read the wrong NTv3 embedding HDF5 key --
see `results/reference/PROVENANCE.md`). chr123 retrained 2026-08-31 (replaces a lost
2026-08-19 checkpoint).

| scope | heldout mu beta MSE | heldout mu PCC | heldout mu R² | heldout sigma PCC |
|---|---:|---:|---:|---:|
| chr1 | 0.00782 | 0.9668 | 0.9337 | 0.7761 |
| chr123 | 0.00873 | 0.9635 | 0.9269 | 0.7657 |
| genomewide | 0.00822 | 0.9660 | 0.9322 | 0.7581 |

## Baseline models

Four simplified baselines (CpG Prior, Global RNA Shift, Bilinear RNA–CpG, MLP RNA–CpG) isolate
why the reference architecture works, each trained/tuned independently per setting — see
[`PAPER_EXPERIMENTS.md`](PAPER_EXPERIMENTS.md#baseline-models) for what each tests, its current
recipe (`configs/models/baselines/`), and **how to run the still-missing retrain under the
current shared-backbone engine**. Results land under
`results/reference/baselines/<name>/<scope>.yaml` (schema-compatible with
`rna_methylation/<scope>.yaml`); the head-to-head table against MethylProphet at
[`baselines_chr1.md`](../results/reference/methylprophet_comparison/baselines_chr1.md) —
**complete for chr1** (2026-09-01): Bilinear RNA–CpG (0.5273 val-CpG × val-sample MAS-PCC) and MLP
RNA–CpG (0.5198) both landed close to the two-stage architecture's 0.5613 and clearly ahead of
MethylProphet (0.3904); Global RNA Shift (0.2343) and CpG Prior (MSE-only, MAS-PCC undefined for a
constant-per-CpG predictor) were well below — **but these numbers were produced by the
now-removed two-stage engine and are frozen historical provenance, not yet reproduced under the
current architecture** (see `PAPER_EXPERIMENTS.md`'s "Baseline models" section for the pending
retrain). chr123/genomewide follow once those settings are themselves verified/built.

## Head-to-head comparison with MethylProphet (published SOTA)

We beat the published MethylProphet numbers decisively on every row/view
tested so far. Full tables (including the historical V0→V1 architecture
progression that motivated the current design, and the known 8,260/918 vs
8,258/920 split caveat) live under `results/reference/methylprophet_comparison/`:

- [`table5_chr1.md`](../results/reference/methylprophet_comparison/table5_chr1.md) —
  Table 5 (single mixed-source model, Array+EPIC+WGBS).
- [`table7_source_ablation.md`](../results/reference/methylprophet_comparison/table7_source_ablation.md) —
  Table 7 (per-training-source rows T(A), T(A+W), T(A+E); TCGA only, ENCODE
  rows still pending).

See [`BENCHMARK_METHYLPROPHET.md`](BENCHMARK_METHYLPROPHET.md) for how these
are reproduced and the protocol caveats.

## Ablation summary

`results/reference/ablations.yaml` holds the machine-readable outcomes of
internal design/hyperparameter ablations against the chr1 `rna_methylation`
reference (val-CpG × val-sample MAS-PCC 0.5613 baseline) — distinct from the
MethylProphet paper-comparison tables above. **Historical note**: every study
below except `shared_backbone_locus_cls_2026_09` and `architecture_novelty_2026_09`
ran against the now-removed two-stage architecture's 0.5613 number (see
CLAUDE.md's "Model compatibility note"); their qualitative conclusions are
still the design record, but their code is not reproducible as-is.

- **`prior_headroom`** — an oracle mu (true per-locus mean instead of the
  predicted prior) reaches MAS-PCC 0.5675, a delta of only 0.0062. Conclusion:
  the static mean prior is not the main MAS-PCC bottleneck.
- **`prior_replacement`** — a marginally-better standalone mean predictor
  (MAS-PCC 0.5598 vs the canonical prior's 0.5613 on its own held-out metric)
  did not improve downstream reconstruction when substituted in. Conclusion:
  better standalone mean prediction alone does not transfer to better joint
  reconstruction.
- **`training_search`** — LR/scheduler search on chr123 (see
  [`WORKFLOWS.md`](WORKFLOWS.md#hyperparameter-search) for the search
  mechanics): LR 5e-5
  constant, 80 epochs was selected (inner double-OOD MAS-PCC 0.5712 at epoch
  80, vs 0.5704 at epoch 57 with the same LR and 0.5639 at epoch 39 with
  LR 4e-5).
- **`interaction_concat_and_latent_dim_2026_08`** — chr1, single seed:
  4-arm sweep over which pieces (`rna`/`cpg`/`product`) feed the interaction
  MLP, plus a 256/512/1024 RNA-latent-width sweep. The product term matters
  (-0.027 MAS-PCC if dropped); once present, raw rna/cpg concatenation and a
  wider latent give only marginal (≤0.002) gains. Conclusion: no permanent
  change to the canonical architecture — see
  [`RNA_METHYLATION.md`](RNA_METHYLATION.md) for the full note.

- **`structured_loss_objective_variants_2026_08`** — three chr1 variants of
  the locus-Pearson structured-loss objective (`large_sample_pcc`: wider
  Array Cartesian blocks for more patients per per-CpG Pearson estimate;
  `tail_aware_pcc`: split the locus-Pearson weight between the mean and the
  lower-60%-correlation tail; `array_only_structured`: restrict Pearson-family
  structured objectives to Array only, since WGBS's 32 measurements are too
  few for a stable patient-dynamic gradient) all underperform the canonical
  objective (0.548/0.545/0.540 vs 0.5613 val-CpG × val-sample MAS-PCC). None
  adopted. Their exact configurations remain in git history
  (`configs/tcga_chr1/experiments/` before the unified-scopes cleanup) if any
  needs to be reproduced.

## Reproduce

- [`WORKFLOWS.md`](WORKFLOWS.md) — the four generic entrypoints
  (`prepare.py`/`train.py`/`tune.py`/`evaluate.py`) across any model × scope.
- [`BENCHMARK_METHYLPROPHET.md`](BENCHMARK_METHYLPROPHET.md) — the exact
  MethylProphet chr1 split/data-preparation path. Its own two-stage
  `MethylProphetTrainer` was removed in the 2026-09-06 refactor (see that
  doc and CLAUDE.md's "Model compatibility note") -- what remains runnable is
  data prep (`scripts/benchmark_methylprophet/prepare.py`) plus training the
  *current* shared-backbone architecture on that data, not a still-runnable
  exact-reproduction trainer.
