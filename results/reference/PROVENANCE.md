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
| `cpg_statistics/chr1.yaml` | `experiments/_legacy_pre_refactor/cpg_mean_predictor/chr1_baseline/final/model.pt` | `05444649...96877b9e` | **stale** — mean-only architecture (`heads.N.*` state dict), incompatible with the current joint mu/sigma `CpGStatisticsPredictor` (`mu_heads.*`/`sigma_heads.*`). Superseded once the roadmap retrain lands. |
| `cpg_statistics/genomewide.yaml` | `experiments/_legacy_pre_refactor/cpg_mean_predictor/genomewide_baseline/final/model.pt` | `bc391cb3...24ab693488b56` | same caveat as above |
| `cpg_statistics/chr123.yaml` | *(checkpoint lost)* | — | a joint mu/sigma checkpoint was trained 2026-08-19 (`runs/runs/cpg_statistics/chr123/20260819T141250Z_.../checkpoints/best.pt`, sha256 `06ddf93d...5731e80c8f1`) but the file no longer exists anywhere on this filesystem; only derived `prior.npy`/`sigma.npy` survive in `derived/rna_feature_cache/chr123/`. Needs a redo. |

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
