# External data and results cleanup inventory

Phase 5e prepares external cleanup without deleting anything.

The scanner is:

```text
scripts/inventory_external_artifacts.py
```

and its policy is:

```text
configs/external_cleanup_policy.yaml
```

## Classifications

Every discovered artifact receives one label:

- `KEEP`: required by the paper profile, explicitly retained by policy, or
  referenced by a standardized final `paper/record.json`;
- `PROTECT`: run/search output that must remain while architecture selection
  or fresh-paper verification is incomplete;
- `REVIEW`: currently unreferenced but not proven obsolete;
- `DELETE_CANDIDATE`: unreferenced, matching an explicit historical/scratch
  pattern, and only after both safety gates are open.

`DELETE_CANDIDATE` is not a deletion instruction.

The scanner has no delete command.

## Dependency evidence

The scanner uses:

1. `configs/data/paper_chr1.yaml`;
2. standardized `paper/record.json` files written by
   `scripts/paper_experiment.py`.

A parent of a referenced path is retained, and a child inside an explicitly
retained artifact is retained.

## Inventory units

The scanner treats these as atomic audit units:

- each first-level `derived/*` artifact;
- each `<storage>/<scope>/<run-id>` under `experiments/runs`;
- each search directory under `experiments/searches`;
- first-level `model_weights_cache/*` and `reference/*`;
- the canonical TCGA bundle.

## Current pre-selection scan

Run now:

```bash
python scripts/inventory_external_artifacts.py \
  --data-root "$METHYL_DATA_ROOT" \
  --output /tmp/methylpredictor-cleanup-inventory
```

Fast structural scan:

```bash
python scripts/inventory_external_artifacts.py \
  --data-root "$METHYL_DATA_ROOT" \
  --output /tmp/methylpredictor-cleanup-inventory-fast \
  --no-sizes
```

Outputs:

```text
inventory.json
inventory.csv
summary.json
```

The report directory must be outside `MethylPredictionData`.

## After architecture selection

```bash
python scripts/inventory_external_artifacts.py \
  --data-root "$METHYL_DATA_ROOT" \
  --output /tmp/methylpredictor-cleanup-after-selection \
  --architecture-selected
```

Historical runs still remain protected because fresh final runs have not yet
been verified.

## After fresh final paper runs

```bash
python scripts/inventory_external_artifacts.py \
  --data-root "$METHYL_DATA_ROOT" \
  --output /tmp/methylpredictor-cleanup-final \
  --architecture-selected \
  --fresh-paper-runs-verified
```

Only then can unreferenced artifacts matching explicit candidate patterns be
labelled `DELETE_CANDIDATE`.

A later destructive cleanup must consume a reviewed/frozen inventory; it must
never scan and delete in one operation.
