# chr123 shared-backbone training optimizations

This note records the input-pipeline changes introduced on 2026-09-04/05 for
pair-complete `chr123` training. They are performance optimizations, not a reduced
training protocol: the schedule still covers the full Cartesian product for Array,
EPIC and WGBS, uses the same loss, and performs the same number of optimizer updates.

## Why chr123 was slow

One pair-complete chr123 epoch contains 2,237 optimizer steps and approximately 1.11
billion sample-CpG pair slots:

| source | rows | CpGs | steps/epoch | pairs/epoch |
| --- | ---: | ---: | ---: | ---: |
| Array | 8,260 | 78,211 | 1,599 | 646,022,860 |
| EPIC | 1,706 | 172,723 | 374 | 294,665,438 |
| WGBS | 32 | 5,396,437 | 264 | 172,685,984 |

The canonical HDF5 files are genome-wide and protocol IDs are sparse in their physical
column axes. A logical 640 x 640 Array block therefore triggered expensive HDF5 fancy
selection instead of one dense slice. Profiling the first implementation showed about
886 seconds waiting for data versus 203 seconds of GPU computation per epoch. The GPU
was idle for most samples despite the data being resident in the operating-system page
cache: the bottleneck was HDF5 selection/decode and CPU block preparation, not disk
bandwidth or model computation.

## Changes

### Bounded parallel prefetch

`LocusCLSJointTrainer` supports two recipe fields:

```yaml
training:
  prefetch_depth: 8
  prefetch_workers: 4
```

Workers prepare HDF5 blocks, resolve RNA/features, scan finite values, convert arrays and
pin host memory concurrently. Futures are consumed in submission order, so GPU optimizer
updates retain the scheduled order. The queue is bounded: at most `prefetch_depth`
prepared or in-flight blocks exist, including large WGBS feature blocks.

`prefetch_workers <= prefetch_depth` is validated when loading a recipe. The defaults
remain one worker and depth two for old recipes.

### Locality-aware pool ordering

For `schedule_layout: contiguous_blocks`, training pools are ordered once by physical
HDF5 row and column position. This avoids treating a shuffled protocol-ID list as if it
were physically contiguous. Pair-complete coverage is unchanged, although minibatch
ordering/grouping is not bit-identical to the old sparse-I/O path (like changing a
deterministic shuffle).

### Protocol-ordered compact methylation cache

The main speedup comes from materializing the immutable canonical methylation targets in
the axes repeatedly used by the chr123 protocol:

```bash
python scripts/prepare_chr123_compact.py \
  --canonical-root /path/to/MethylPredictionData/datasets/methylprophet_repro_v1 \
  --output /path/to/MethylPredictionData/derived/methylprophet_compact_chr123
```

The output contract is:

```text
methylprophet_compact_chr123/
  array.h5       # (9,178, 93,104): train+validation axes
  epic.h5        # (1,706, 172,723)
  wgbs.h5        # (32, 5,396,437), duplicate patient measurement retained
  manifest.json  # shapes and SHA-256 hashes of both ID axes
```

Each file contains `beta`, `sample_idx`, `cpg_idx` and, where available,
`measurement_idx`. The cache stores targets only; RNA and 1,536-dimensional NTv3
features remain in their existing caches. Source files are read-only and never changed.
The builder is restartable: completed source caches are validated and reused.

WGBS requires a separate column compactor because its 32 measurements contain 31 unique
patient IDs. The repeated measurement is intentionally retained rather than deduplicated.

Pass the compact directory through the shared-backbone `--prepared-root` on the standard
entrypoint (the one-off sequential-comparison wrapper that used to run this automatically,
`scripts/experiments/run_chr123_shared_backbone_comparison.py`, was removed 2026-09-05 --
it compared the now-superseded `linear` encoder as "primary" against a Hyper-Connections/mHC
arm that has since been removed from the codebase entirely, see
`results/reference/ablations/architecture_novelty_2026_09/README.md`'s "What was tried and
removed" section):

```bash
python scripts/train.py --model rna_methylation --scope chr123 \
  --engine matched_chr1_shared_backbone --prepared-root /path/to/compact_chr123_dir \
  --canonical-root ... --feature-cache ... --rna-cache ... --registry ... \
  --cpg-targets-dir derived/cpg_statistics/chr123 \
  --recipe configs/models/rna_methylation_locus_attention.yaml --mode final --output-root ...
```

The feature cache may use the expanded `chr123_rna_universe`, but `--cpg-targets-dir`
must point to `derived/cpg_statistics/chr123`. The expanded/imputed universe is needed to
look up auxiliary CpG features; it must not supervise the mean-head auxiliary loss.

## Correctness checks

Before the production restart, representative Array, EPIC and WGBS blocks were compared
against the canonical sources with exact equality, including NaN masks. Axis order and
hashes are persisted in `manifest.json`, and `load_compact_scope_sources` refuses caches
that cannot resolve every protocol CpG or official Array sample.

Measured representative block reads:

| source | canonical read | compact read | block-read speedup |
| --- | ---: | ---: | ---: |
| Array 640 x 640 | 5.9701 s | 0.0046 s | 1,290x |
| EPIC 160 x 5,120 | 0.8268 s | 0.0076 s | 109x |
| WGBS 32 x 20,480 | 0.0175 s | 0.0160 s | 1.1x |

WGBS was already efficient because the canonical file is column-major. End-to-end epoch
time is the relevant result: 1,076 seconds fell to 146 seconds (7.4x), data wait fell
from about 886 seconds to 0.28 seconds, throughput rose to 7.62 million pair slots/s, and
GPU utilization rose from intermittent/near-zero to approximately 89-93%. The compact
cache occupies 5.1 GiB. These figures describe the primary model on this server and are
not guaranteed constants for other storage or architectures.

## Can this be reused elsewhere?

Yes, when a fixed protocol repeatedly selects a stable subset of a larger dense matrix.
The benefit is highest when logical batches are sparse in the canonical HDF5 axes and
training revisits them for many epochs.

- **chr1:** already uses the same idea. The matched chr1 preparation contains compact
  Array and EPIC caches; canonical WGBS is kept direct because 32-row column-major reads
  are already fast. Rebuilding another chr1 compact cache is unlikely to help materially.
- **Other chromosome subsets:** generalizable by materializing their protocol sample and
  CpG axes and passing that directory to the shared-backbone loader. The current builder
  is intentionally chr123-specific; factor its scope/axis selection into arguments before
  using it as a general production command.
- **Genome-wide:** technically possible, but pair-complete training is usually the wrong
  compute policy at that scale. Benchmark compact-cache benefit against
  `axis_full_coverage` before paying the additional storage/build cost.
- **Sparse or one-pass datasets:** usually a poor fit. A compact dense copy can waste
  space when observations are sparse or each cell is read only once.

Do not assume the block-read microbenchmark is the model speedup. Always record
`data_wait_seconds`, `cpu_prepare_seconds`, `compute_seconds` and
`pair_slots_per_second` from the first full epoch, and keep the canonical-vs-compact
exact-equality check for every new scope.
