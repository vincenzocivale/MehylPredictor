# Paper dependency inventory

`configs/data/paper_dependencies_v2.yaml` is the exact dependency matrix for
the frozen final paper studies.

The matrix separates:

- runtime inputs required by training/evaluation;
- comparator-specific RNA caches;
- repository resources;
- reproducibility-only assets.

Run the read-only scanner with:

```bash
python scripts/inventory_paper_dependencies.py   --data-root "$METHYL_DATA_ROOT"   --output /tmp/methylpredictor-paper-deps   --no-sizes
```

The scanner emits:

```text
dependency_inventory.json
dependency_matrix.csv
artifact_inventory.csv
summary.json
```

No output may be written inside `METHYL_DATA_ROOT`.

## Important prior-cache distinction

The live RNA model reads only:

```text
cpg_idx.npy
prior.npy
```

through `LocusPriorCache`.

The current zero-parameter CpG-prior evaluator still constructs a
`LocusFeatureCache`, so its present contract additionally requires:

```text
embeddings.f16.npy
sigma.npy
```

The dependency inventory records that stricter consumer-specific requirement.
A later cleanup optimization may refactor the CpG-prior evaluator to
`LocusPriorCache`, after which those two files can be reconsidered.

## Artifact classifications

Top-level external artifacts are classified as:

- `REQUIRED_RUNTIME`: directly needed by at least one frozen paper arm;
- `REQUIRED_REPRODUCIBILITY`: retained for reproducible preparation;
- `UNREFERENCED_CLEANUP_TARGET`: explicitly named by Data/Results V2 as a
  historical target and not required by any frozen arm;
- `REVIEW`: not required, but not yet explicitly authorized for cleanup.

This phase remains read-only. `UNREFERENCED_CLEANUP_TARGET` is not a deletion
command.
