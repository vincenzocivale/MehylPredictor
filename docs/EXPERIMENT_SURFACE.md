# Experiment surface freeze

This file separates the paper/reproducibility surface from the architecture
search machinery that is still present on the refactor branch.

The machine-readable source of truth is
[`configs/experiment_surface.yaml`](../configs/experiment_surface.yaml).
Every model recipe and every file under `scripts/experiments/` is classified
exactly once and content-hashed. The regression test fails if an active file
is silently edited, added, removed, or left unclassified.

## Paper-facing surface

### Primary reference

- `configs/models/main.yaml`

`main.yaml` is the only primary RNA reference recipe. The historical
`functional_fusion/j0_final.yaml` name is retained only as a migration alias.

### Claim-driven ablations

The paper-facing mean-proxy ablation is:

- full reference (`main.yaml`);
- no mean supervision;
- no mean branch.

These isolate the contribution of the auxiliary mean-proxy task without
changing the rest of J0.

### Functional-matched baselines

Retained baseline recipes are:

- CpG Prior;
- Global RNA Shift;
- pooled-RNA + functional-CpG MLP;
- bilinear RNA-CpG interaction.

The three trainable baselines use the same functional locus information as the
reference model and differ in how patient RNA interacts with the locus.

### RNA encoder comparison

Retained RNA-encoder arms are:

- our program-token encoder;
- MethylProphet-style bottleneck MLP;
- gene/pathway encoder;
- frozen BulkFormer-147M embedding;
- frozen BulkRNABert embedding.

These are claim-driven comparator experiments, not architecture-search rungs.

## Support surface

`configs/models/cpg_statistics.yaml` remains because the repository still
contains the auxiliary/static CpG-statistics workflow and the mean-proxy target
preparation path.

## Compatibility surface

The following remain only during migration:

- `functional_fusion/base.yaml`;
- `functional_fusion/j0_final.yaml`;
- the historical `locus_cls_joint` run/checkpoint storage key.

They are not additional public model choices.

## Protected architecture-selection surface

J1 and J2-J10 are currently classified as **protected research**, not
paper-facing models. They include iterative retrieval, attention-depth,
residual/gated, efficient single-attention, functional-branch-depth, and
FFN-fusion experiments.

They remain in the branch because they may still be running, resumable, or
needed to select the final architecture. Their recipe and launcher hashes are
frozen by regression test. Do not refactor or delete these files until the
architecture decision is explicitly locked.

Once that decision is made:

1. promote at most the selected architecture/control needed by the paper;
2. update `main.yaml` only if the selected architecture replaces J0;
3. retain only claim-relevant ablations;
4. delete the remaining J-series recipes/launchers from the paper branch;
5. then remove config fields/classes that are used only by deleted recipes.

## Known reproducibility gap

The functional baseline registry exists, but unlike the mean-proxy and RNA
encoder studies it currently has no dedicated paper-facing run/collect wrapper.
That should be filled during the reproducibility phase. It should not be solved
by restoring historical shared-backbone baseline tooling.

## Why hashes are frozen

While active experiments are resumable, a recipe can be scientifically
identified only if its content remains stable. The manifest therefore records
SHA-256 digests for the current recipe and experiment-script surface.

An intentional edit requires an explicit manifest refresh in the same commit,
making the change visible in review instead of silently changing the meaning of
an existing run ID.
