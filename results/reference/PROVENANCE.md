# Provenance: recorded numbers → raw run directories

Every number in `results/reference/` traces back to a raw run under
`/dune/DATASETS/MethylPredictionData/experiments/` (this machine; gitignored,
not version-controlled). This file is the index from one to the other, so a
paper table/figure can always be traced back to its exact checkpoint. Update
it whenever a `results/reference/**` file is added or a referenced run is
superseded/deleted.

Layout note (2026-08-28 reorg): new runs from `scripts/{train,tune,evaluate}.py`
should land under `experiments/runs/` (the `RunStore` layout,
`runs/<model>/<scope>/<run-id>/`) by passing `--output-root
/dune/DATASETS/MethylPredictionData/experiments` (RunStore appends the `runs/`
segment itself -- do not also put `/runs` in `--output-root` or it double-nests).
Everything below predates that convention and lives under older ad hoc paths;
entries are updated to `experiments/runs/...` as each is redone.

## `rna_methylation`

| file | run path | checkpoint sha256 | status |
|---|---|---|---|
| `rna_methylation/chr1.yaml` | `experiments/MethylPredictor/tcga_chr1/e1_v1_mixed_source_lr5e-5_constant_80ep_seed17/checkpoints/final.pt` | `9cc73aa6...753b60` | live, verified numbers match headline.json exactly |
| `rna_methylation/genomewide.yaml` | `experiments/_legacy_pre_refactor/e7_v1_array_genomewide_seed17/checkpoints/final.pt` | `f2d7aae2...142e627` | **provenance flag open** — pre-refactor trainer, chr1-labeled embeddings path in its resolved config despite genome-wide-looking eval; not yet re-derived under the current pipeline. Roadmap priority: redo under `scripts/train.py --scope genomewide` once `cpg_statistics` genomewide is retrained (below), then replace this row. |
| `rna_methylation/chr123.yaml` | *(none yet)* | — | pending; not a verified MethylProphet comparison (see `methylprophet_comparison/`) |

## `cpg_statistics`

| file | run path | checkpoint sha256 | status |
|---|---|---|---|
| `cpg_statistics/chr1.yaml` | `experiments/runs/cpg_statistics/chr1/cpg-statistics-chr1-retrain-2026-08-28/checkpoints/best.pt` | `b1b6c64a...360d43c52` | **live**, joint mu/sigma, retrained 2026-08-28 after the NTv3-embedding-key bugfix (commit `c8ea106`) |
| `cpg_statistics/genomewide.yaml` | `experiments/runs/cpg_statistics/genomewide/cpg-statistics-genomewide-retrain-2026-08-28/checkpoints/best.pt` | `265e3bf1...71c21b87a` | **live**, joint mu/sigma, retrained 2026-08-28 |
| `cpg_statistics/chr123.yaml` | *(not yet redone)* | — | deferred with `rna_methylation/chr123.yaml` (chr1 → genomewide is the current roadmap priority); an earlier joint mu/sigma checkpoint for this scope (2026-08-19) is lost, see below |

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

## `ablations.yaml`

| entry | run path | status |
|---|---|---|
| `prior_headroom` | *(raw run not retained)* | result recorded, closed, no code/config change |
| `prior_replacement` | *(raw run not retained)* | result recorded, closed, no code/config change |
| `training_search` | *(deleted 2026-08-28 — cleanup pass, closed ablation)* | was `experiments/training_search/tcga_chr1_v1*` |
| `interaction_concat_and_latent_dim_2026_08` | *(deleted 2026-08-28 — cleanup pass, closed ablation)* | was `runs/runs/rna_methylation/chr1/ablation-*` (this repo's local `runs/`, not `/dune`) |
| `structured_loss_objective_variants_2026_08` | `experiments/MethylPredictor/tcga_chr1/{tail_aware_pcc,large_sample_pcc,array_only_structured}/` | live, `.done` + `evaluation/headline.json` present in each |

## `methylprophet_comparison/`

| file | run path | status |
|---|---|---|
| `table5_chr1.md` (current row) | same as `rna_methylation/chr1.yaml` above | live |
| `table5_chr1.md` (historical V0→V1 progression) | `experiments/_legacy_pre_refactor/final_model/methylprophet_table5_tcga_chr1/{seed17_epoch4_baseline,seed17_epoch25_oldprior_baseline,seed17_epoch25_v3_fixedprior,seed17,seed17_v1}/` | archived, narrative-only, not reproducible with current code |
| `table7_source_ablation.md` | `experiments/MethylPredictor/tcga_chr1/{table7_train_array_only,table7_train_array_wgbs,table7_train_array_epic}/` | live |
| `chr1_official_split_verification.md` | `methylprophet_official/eval-tcga_mix_chr1-bs_512-c2b2/` (downloaded HF artifact, not a training run) | live; own checksums recorded in that file |
