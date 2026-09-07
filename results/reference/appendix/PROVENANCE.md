# Provenance: recorded numbers → raw run directories

> **Relocated 2026-09-06** into `results/reference/appendix/` when `results/reference/` was
> reorganized around `docs/PAPER_ROADMAP.md` (see `../README.md`). Every path below that starts
> `rna_methylation/...`, `baselines/...`, etc. refers to this file's *pre-reorganization* location
> (`results/reference/<that path>`) — kept as-is, historical, not current. Current numbers for the
> paper live in `../paper/` (MethylProphet published) and `../ours/` (our experiments, with
> checkpoint paths inline in each entry — this file's indexing role is now secondary to those).

Every number in `results/reference/` traces back to a raw run under
`results/experiments/` (gitignored, not version-controlled). This file is the index from one to the other, so a
paper table/figure can always be traced back to its exact checkpoint. Update
it whenever a `results/reference/**` file is added or a referenced run is
superseded/deleted.

## 2026-09-06 pipeline: B.1 chr1 close-out, chr123-ref, cross-scope, chr123 baselines

| file | run path | checkpoint sha256 | status |
|---|---|---|---|
| `ours/01_final_chr1_model.yaml`, `ours/02_rna_encoder_comparison.yaml` (cross_attention arm), `ours/03_mean_contribution.yaml` (full_reference arm), `ours/05_chromosome_generalization.yaml` (chr1→chr1 cell) | `MethylPredictionData/experiments/runs/locus_cls_joint/chr1/reference-locus_attention_no_product-seed17/checkpoints/best.pt` | `0504ecd7d330bd295f4baa2543da07b5ddd3218b0c9bceb4a21c548c0fb81095` | **live**, confirmed official 2026-09-06 21:39 UTC, 80/80 epochs |
| `ours/05_chromosome_generalization.yaml` (chr123→chr123 cell) | `.../runs/locus_cls_joint/chr123/chr123-ref/checkpoints/best.pt` | `ee2836b9c7e2e46997fb611235535b288f7fd3297ae11e401a54c1e6c2e4ba8a` | **live**, confirmed official 2026-09-06 22:11 UTC, 80/80 epochs |
| `ours/05_chromosome_generalization.yaml` (chr123→chr1 cell) | `.../runs/locus_cls_joint/chr123/chr123-ref/checkpoints/best.pt`, eval-only against chr1 data | `ee2836b9...` (same as above) | **live**, confirmed official 2026-09-06 22:12 UTC |
| `ours/05_chromosome_generalization.yaml` (chr1→chr123 cell) | `.../runs/locus_cls_joint/chr1/reference-locus_attention_no_product-seed17/checkpoints/best.pt`, eval-only against chr123 data | `0504ecd7...` (same as chr1 row above) | **live**, confirmed official 2026-09-06 22:12 UTC |
| `ours/04_baselines.yaml` (chr123 columns) | `.../experiments/baselines/cpg_prior/chr123/metrics.json/metrics.json` (no checkpoint, zero-parameter) | n/a | **live**, confirmed official 2026-09-06 22:54 UTC |
| `ours/04_baselines.yaml` (chr123 columns) | `.../runs/locus_cls_joint/chr123/baseline-global-shift-chr123/checkpoints/best.pt` | `33145552eee27f07de471a5973f52c5158a2992199d0297a445937265091ad19` | **live**, confirmed official 2026-09-07 01:05 UTC |
| `ours/04_baselines.yaml` (chr123 columns) | `.../runs/locus_cls_joint/chr123/baseline-bilinear-chr123/checkpoints/best.pt` | `7d365ca56cc4f57bf7f5b335de54e96af5bea82c5a09a00a9b905c6439cb5696` | **live**, confirmed official 2026-09-07 03:16 UTC |
| `ours/04_baselines.yaml` (chr123 columns) | `.../runs/locus_cls_joint/chr123/baseline-mlp-chr123/checkpoints/best.pt` | `a0071baa0127b754306a40d025a9f33dbe7e3ad063fa9b513f71f674cc0f4967` | **live**, confirmed official 2026-09-07 05:55 UTC |

B.4 (baselines) is now fully closed, chr1 + chr123, all 8 cells. See
`logs/b1_final_reference/queue_chr123_baselines.log` for the full run history.

Layout note: new runs from `scripts/{train,tune,evaluate}.py` should land under
`results/experiments/runs/` (the `RunStore` layout,
`runs/<model>/<scope>/<run-id>/`) by passing `--output-root
results/experiments` (RunStore appends the `runs/` segment itself -- do not also
put `/runs` in `--output-root` or it double-nests).
Everything below predates that convention and lives under older ad hoc paths;
entries are updated to `experiments/runs/...` as each is redone.

## `rna_methylation`

| file | run path | checkpoint sha256 | status |
|---|---|---|---|
| `rna_methylation/chr1.yaml` | `experiments/MethylPredictor/tcga_chr1/e1_v1_mixed_source_lr5e-5_constant_80ep_seed17/checkpoints/final.pt` | `9cc73aa6...753b60` | live, verified numbers match headline.json exactly |
| `rna_methylation/genomewide.yaml` | `experiments/_legacy_pre_refactor/e7_v1_array_genomewide_seed17/checkpoints/final.pt` | `f2d7aae2...142e627` | **provenance flag open** — pre-refactor trainer, chr1-labeled embeddings path in its resolved config despite genome-wide-looking eval; not yet re-derived under the current pipeline. Roadmap priority: redo under `scripts/train.py --scope genomewide` once `cpg_statistics` genomewide is retrained (below), then replace this row. |
| `rna_methylation/chr123.yaml` | `experiments/runs/rna_methylation/chr123/rna-methylation-chr123-retrain-2026-08-31/checkpoints/best.pt` | `d07dd24f...3851818f` | **live**, first completed run for this scope (2026-08-31), generic engine over the full tcga_mix_chr123 auxiliary universe (see `derived/rna_feature_cache/chr123/` note below); not a verified MethylProphet comparison (see `methylprophet_comparison/`) |

## `baselines/`

| file | run path | checkpoint sha256 | status |
|---|---|---|---|
| `baselines/cpg_prior/chr1.yaml` | n/a (zero-parameter, `CpGPriorEvaluator`, no checkpoint) | n/a | **live**, evaluated 2026-09-01 via `scripts/evaluate.py --model cpg_prior` against `derived/methylprophet_table5_tcga_chr1/features` |
| `baselines/global_rna_shift/chr1.yaml` | `experiments/runs/rna_methylation/chr1/baseline-global-rna-shift-chr1-2026-09-01/checkpoints/final.pt` | `e108fd49...fdb76fd` | **live**, first completed run (2026-09-01), `--engine matched_chr1`, 80 epochs |
| `baselines/bilinear_rna_cpg/chr1.yaml` | `experiments/runs/rna_methylation/chr1/baseline-bilinear-rna-cpg-chr1-2026-09-01/checkpoints/final.pt` | `06e69775...cffc03` | **live**, first completed run (2026-09-01), `--engine matched_chr1`, 80 epochs |
| `baselines/mlp_rna_cpg/chr1.yaml` | `experiments/runs/rna_methylation/chr1/baseline-mlp-rna-cpg-chr1-2026-09-01/checkpoints/final.pt` | `6a98e1ce...ce9f965` | **live**, completed 2026-09-01, `--engine matched_chr1`, 80 epochs. First attempt failed (`interaction.include_product` rejected as a retired field by `benchmark/methylprophet/config.py::load_config`; fixed by re-permitting that field specifically for this baseline). Retry was killed mid-run at epoch 7 by an external process (not a code/data error; GPU/process state showed no OOM) and resumed via `--resume` from `checkpoints/latest.pt` to completion |

Add a row per `<name>/<scope>.yaml` as each baseline's training/eval run completes, per
`docs/PAPER_EXPERIMENTS.md`'s "Baseline models" section (chr1 first; chr123/genomewide follow once
those settings are themselves verified/built).

## `cpg_statistics`

| file | run path | checkpoint sha256 | status |
|---|---|---|---|
| `cpg_statistics/chr1.yaml` | `experiments/runs/cpg_statistics/chr1/cpg-statistics-chr1-retrain-2026-08-28/checkpoints/best.pt` | `b1b6c64a...360d43c52` | **live**, joint mu/sigma, retrained 2026-08-28 after the NTv3-embedding-key bugfix (commit `c8ea106`) |
| `cpg_statistics/genomewide.yaml` | `experiments/runs/cpg_statistics/genomewide/cpg-statistics-genomewide-retrain-2026-08-28/checkpoints/best.pt` | `265e3bf1...71c21b87a` | **live**, joint mu/sigma, retrained 2026-08-28 |
| `cpg_statistics/chr123.yaml` | `experiments/runs/cpg_statistics/chr123/cpg-statistics-chr123-retrain-2026-08-31/checkpoints/best.pt` | `cdaa4a50...42c47e5ccf` | **live**, joint mu/sigma, retrained 2026-08-31, replaces the lost 2026-08-19 checkpoint |

Superseded (kept only as historical/provenance reference, no longer the source of any recorded
number): `experiments/_legacy_pre_refactor/cpg_mean_predictor/{chr1_baseline,genomewide_baseline}/final/model.pt`
— mean-only architecture (`heads.N.*` state dict), incompatible with the current joint mu/sigma
`CpGStatisticsPredictor` (`mu_heads.*`/`sigma_heads.*`). Also lost: the 2026-08-19 joint mu/sigma
checkpoints (`runs/runs/cpg_statistics/{chr1,chr123,genomewide}/20260819T*/checkpoints/best.pt`,
sha256 recorded in `results-reference-taxonomy` session memory) that produced the `prior.npy`/
`sigma.npy` arrays still baked into `derived/rna_feature_cache/{chr1,chr123,genomewide}/` — those
files no longer exist on disk. `derived/rna_feature_cache/{chr1,genomewide}/` need regenerating
from the 2026-08-28 checkpoints above via `scripts/prepare.py --model rna_methylation` before the
next `rna_methylation` retrain uses them.

### chr123 feature-cache universe extension (2026-08-31)

`scripts/prepare.py --model cpg_statistics` builds its `cpg_idx.npy` universe from only the official
Array train/val CpGs (`cpg_statistics/targets.py`), same as chr1. But `rna_methylation`'s generic
training engine resolves the `tcga_mix_chr123` protocol, whose auxiliary axis needs 5,314,693
additional EPIC/WGBS-only loci beyond that Array universe (5,407,797 total) — training crashed with
`LocusFeatureCache` missing those IDs until this was fixed. Unlike chr1 (which has a dedicated
`matched_chr1` engine/cache for this exact problem), no `matched_chr123` equivalent exists.

Fix used (not `scripts/prepare.py` itself — a one-off padding step, not yet promoted to a script):
built `derived/cpg_statistics/chr123_rna_universe/` = the official `derived/cpg_statistics/chr123/`
targets, padded with the 5,314,693 auxiliary-only CpGs (`official_train_mask`/`official_val_mask`
both `false`, placeholder `target_mu`/`target_sigma` = 0 — never read, since `export_feature_cache`
only substitutes empirical targets where `official_train_mask` is `true`). All 5,407,797 required
CpGs were already present in the NTv3 atlas (`ntv3_cpg_atlas_v1.h5`, 5,723,092 rows), so no new NTv3
inference was needed. Re-ran `scripts/prepare.py --model rna_methylation` against this padded
targets dir to regenerate the full `derived/rna_feature_cache/chr123/` (5,407,797 rows, supersedes
the smaller/orphaned 2026-08-19 cache). The official `cpg_statistics/chr123.yaml` reference metrics
are unaffected — those are still computed from the pure-Array `derived/cpg_statistics/chr123/`.

**How to apply:** if chr123 `rna_methylation` is retrained again, reuse
`derived/cpg_statistics/chr123_rna_universe/` (or rebuild it the same way from a newer
`cpg_statistics/chr123` checkpoint) rather than pointing `--targets` at the plain
`derived/cpg_statistics/chr123/` dir, or training will hit the same missing-IDs crash. Consider
promoting this padding step into `scripts/prepare.py` if chr123 `rna_methylation` becomes a recurring
retrain target.

### chr123 Array sample split investigation (2026-09-02) — CpG axis verified, sample axis not reproducible from our data

Attempted the same released-artifact ID verification used for chr1, against
`MethylProphet/eval-tcga-mix-chr123-bs_512-32xl40s-aws-eval_on_tcga_chr123`. The CpG axis
(`note1 ∪ note4`) matched exactly, confirming `protocols/tcga_mix_chr123`'s existing CpG split as-is
(no change). The Array *sample* split does not: the release's own sample_idx values include 306
samples that don't exist anywhere in this repo's canonical Array HDF5 (`tcga_array_official_full.h5`,
241231 snapshot) — a real content difference, not an ID-extraction bug (the release's sample_idx
range extends to 10,915; our bundle's only to 10,702, and the CpG axis extracted via the exact same
code path from the exact same rows matched exactly). The release's chr123 sample split is therefore
**not reproducible** against our current canonical bundle. `array_train_sample_idx.npy`/
`array_val_sample_idx.npy` were briefly overwritten with the release IDs, found to break
`cpg_statistics/targets.py` (`KeyError: sample_idx value not found`) on the very first CpG,
and reverted to the chr1-reused 8,260/918 split — the only one internally consistent with our own
data, and numerically identical to what `derived/cpg_statistics/chr123/` and
`derived/rna_feature_cache/chr123/` were already built from. **No retrain is needed**: the existing
`cpg_statistics/chr123.yaml` and `rna_methylation/chr123.yaml` checkpoints/caches are unaffected by
this investigation. See
`results/reference/methylprophet_comparison/chr1_official_split_verification.md`'s "chr123:
verified 2026-09-02" section for the full record, including the missing/extra sample_idx ranges.

## `ablations.yaml`

| entry | run path | status |
|---|---|---|
| `prior_headroom` | *(raw run not retained)* | result recorded, closed, no code/config change |
| `prior_replacement` | *(raw run not retained)* | result recorded, closed, no code/config change |
| `training_search` | *(deleted 2026-08-28 — cleanup pass, closed ablation)* | was `experiments/training_search/tcga_chr1_v1*` |
| `interaction_concat_and_latent_dim_2026_08` | *(deleted 2026-08-28 — cleanup pass, closed ablation)* | was `runs/runs/rna_methylation/chr1/ablation-*` (this repo's local `runs/`, not `/dune`) |
| `structured_loss_objective_variants_2026_08` | `experiments/MethylPredictor/tcga_chr1/{tail_aware_pcc,large_sample_pcc,array_only_structured}/` | live, `.done` + `evaluation/headline.json` present in each |

## `ablations/<study>/` (per-study directories, distinct from `ablations.yaml` above)

| study | run path(s) | status |
|---|---|---|
| `architecture_novelty_2026_09` (`enc_bottleneck_mlp`, `enc_frozen_embedding_bulkrnabert`) | `results/experiments/runs/locus_cls_joint/chr1/arch-architecture_novelty_2026_09-shared-{enc_bottleneck_mlp,enc_frozen_embedding_bulkrnabert}-seed17/checkpoints/best.pt` | **live**, collected 2026-09-06 via `scripts/experiments/collect_arch_results.py`; both underperform the `locus_attention` reference (row B) and are not promoted — see that study's `summary.md`/`README.md` and `docs/RNA_METHYLATION.md`'s "Forward direction" section. Remaining arms in the suite (`seed_variance_reference` etc.) not yet collected. |
| `query_representation_2026_09` (`q0_ntv3`/`q1_mean_only`/`q2_hybrid_detached`/`q3_hybrid_joint`) | `results/experiments/runs/locus_cls_joint/chr1/` (query-representation run ids, seed17) | **live**, collected via `scripts/experiments/collect_query_representation.py`; winner `q0_ntv3` (0.5352) on the development split — see that study's `README.md`/`summary.md`. |

## `methylprophet_comparison/`

| file | run path | status |
|---|---|---|
| `table5_chr1.md` (current row) | same as `rna_methylation/chr1.yaml` above | live |
| `table5_chr1.md` (historical V0→V1 progression) | `experiments/_legacy_pre_refactor/final_model/methylprophet_table5_tcga_chr1/{seed17_epoch4_baseline,seed17_epoch25_oldprior_baseline,seed17_epoch25_v3_fixedprior,seed17,seed17_v1}/` | archived, narrative-only, not reproducible with current code |
| `table7_source_ablation.md` | `experiments/MethylPredictor/tcga_chr1/{table7_train_array_only,table7_train_array_wgbs,table7_train_array_epic}/` | live |
| `chr1_official_split_verification.md` | `methylprophet_official/eval-tcga_mix_chr1-bs_512-c2b2/` (downloaded HF artifact, not a training run) | live; own checksums recorded in that file |
