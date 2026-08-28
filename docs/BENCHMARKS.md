# Benchmarks

Machine-readable numbers live in `results/reference/`; this document is the
narrative index into them.

## Models and scopes

Two trainable models, three genomic scopes:

- `CpGStatisticsPredictor` → `results/reference/cpg_statistics/{chr1,chr123,genomewide}.yaml`
- `RNAMethylationPredictor` → `results/reference/rna_methylation/{chr1,chr123,genomewide}.yaml`

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

| scope | train-CpG × val-sample | val-CpG × train-sample | val-CpG × val-sample |
|---|---:|---:|---:|
| chr1 | 0.6327 / 0.01278 | 0.5984 / 0.01874 | 0.5613 / 0.01941 |
| chr123 | *pending — see `results/reference/rna_methylation/chr123.yaml`* | | |
| genomewide | 0.5889 / 0.01482 | 0.5800 / 0.02099 | 0.5244 / 0.02223 |

Cells are `MAS-PCC / MSE`. Genome-wide per-chromosome mean MAS-PCC: 0.5196
(observed range 0.459–0.579). LR 5e-5, constant scheduler, 80 epochs, seed 17
for both frozen chr1 and genome-wide runs.

### `cpg_statistics`

Joint mu/sigma ensemble model (retrained 2026-08-28 after fixing a real bug where
`cpg_statistics/{trainer,evaluator,export}.py` read the wrong NTv3 embedding HDF5 key --
see `results/reference/PROVENANCE.md`). chr123 is deferred, not yet retrained.

| scope | heldout mu beta MSE | heldout mu PCC | heldout mu R² | heldout sigma PCC |
|---|---:|---:|---:|---:|
| chr1 | 0.00782 | 0.9668 | 0.9337 | 0.7761 |
| genomewide | 0.00822 | 0.9660 | 0.9322 | 0.7581 |

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
MethylProphet paper-comparison tables above:

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
- [`BENCHMARK_METHYLPROPHET.md`](BENCHMARK_METHYLPROPHET.md) — the frozen,
  exact MethylProphet chr1 reproduction path.
