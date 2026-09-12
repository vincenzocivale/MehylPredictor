# Repository cleanup inventory

The repository is being cleaned before the final architecture is selected.
The goal is to remove dead research-history material while keeping the active
J-series fully resumable.

The machine-readable policy is
[`configs/repository_cleanup_inventory.yaml`](../configs/repository_cleanup_inventory.yaml).

## Removed now

This phase removes only material that cannot affect an active experiment:

- temporary refactor patch scripts;
- the local architecture-result audit utility from the previous exploratory step;
- Git-tracked `.eval_runs` / diagnostic outputs;
- the stale `legacy benchmark index` overview;
- the stale genomic-feature-store document that still described FM embeddings
  as a current RNA-model dependency.

The actual external experiment directories under `MethylPredictionData` are
not touched.

## Protected until architecture selection

All J1-J10 architecture-search recipes and launchers remain frozen. Their
supporting predictor implementations and factory selectors remain too.

## After the final architecture is chosen

The selected architecture is promoted to `configs/models/main.yaml`. Then:

1. delete non-selected J-series recipes and launchers;
2. delete predictor classes/selectors used only by those recipes;
3. delete architecture-only tests that no longer exercise retained code;
4. remove shared-backbone-era config compatibility fields;
5. remove `locus_cls_trainer.py` and legacy aliases;
6. optionally migrate new runs away from the historical `locus_cls_joint`
   storage key.

## Results policy

Current `results/reference/` files are historical/frozen ledgers. They are
useful during development but are not the intended final paper result tree.

Once the definitive architecture is selected, paper-facing experiments should
be rerun cleanly under the standardized pipeline. Only after those runs are
verified should `results/reference/` be replaced by a compact final
`results/paper/` summary generated from the fresh runs.

## External data policy

No external dataset or cache is deleted during repository cleanup.

After the fresh paper runs exist, a separate dry-run dependency scanner should
classify external artifacts as KEEP / DELETE / REVIEW from the actual final
run manifests. This prevents accidental deletion of a cache still required for
reproduction.
