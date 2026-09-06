# mean_contribution_2026_09

Paper-facing results ledger for the locus-mean proxy task in the current
locus-attention RNA-methylation model.

Raw checkpoints and prediction matrices do **not** belong here. The collector
writes only compact, reproducible paper results.

Protocol and commands:
[`docs/MEAN_CONTRIBUTION_EXPERIMENTS.md`](../../../docs/MEAN_CONTRIBUTION_EXPERIMENTS.md).

Expected generated files:

- `summary.yaml` — aggregate metrics, paired effects and protocol metadata.
- `summary.md` — manuscript-facing summary.
- `paper_table.csv` — one arm × seed × official-view row.
- `variance_decile_effects.csv` — full-vs-ablation MSE/bias effect across CpG variance deciles.
- `dataset_variance.json` — missing-aware variance decomposition.
- `runs/*.json` — complete per-run provenance and diagnostics.

Primary causal contrast: `full_reference` vs `no_mean_supervision`.

Primary mean-sensitive endpoints: MSE and locus-level bias on official
Val-CpG views. MAS-PCC is secondary because a CpG-specific constant shift
does not affect within-CpG Pearson correlation.
