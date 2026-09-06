# Planning: expanding NTv3 CpG coverage (true multi-technology genomewide, + ENCODE)

**Status: planning only. Nothing in this document has been executed — no NTv3 inference has
been run, no atlas file has been modified, no scope has been redefined in code.** This is the
write-up of an exploration session (2026-08-28) so the next session can pick this up without
re-deriving the numbers below.

## The ask

Two related but distinct goals surfaced in the same conversation:

1. Compare our RNA-methylation model against MethylProphet's **ENCODE** benchmark (Table 7
   E-rows), analogous to the already-completed TCGA T(A)/T(A+W)/T(A+E) comparison
   ([`table7_source_ablation.md`](../results/reference/methylprophet_comparison/table7_source_ablation.md)).
2. Redefine this repo's own `genomewide` scope so it means what the name implies: **the union
   of Array + EPIC + WGBS CpGs across all chromosomes**, not just the Array-probe universe
   (current `genomewide` = the `array_genomewide` protocol, 408,399 CpGs). This should
   **replace** the existing `genomewide` scope, not sit alongside it as a second one — the
   current Array-only numbers in `results/reference/{cpg_statistics,rna_methylation}/genomewide.yaml`
   are to be treated as superseded once the new numbers exist, not kept as a parallel scope.

Both goals bottleneck on the same resource: NTv3 CpG embeddings for chromosomes/CpGs the
current atlas doesn't densely cover. It makes sense to plan them together.

## Why the current `genomewide` isn't what the name suggests

`ntv3_cpg_atlas_v1.h5` (`/dune/DATASETS/MethylPredictionData/datasets/methylprophet_repro_v1/cpg/ntv3/ntv3_cpg_atlas_v1.h5`,
5,723,092 CpGs total) has two origins, and they explain the current scope design exactly:

- `origin_0 = array_genomewide_408399` — all 408,399 Array probes, genuinely spanning every
  chromosome (chr1-22; the bundle has no chrX/Y/M CpGs at all in Array/EPIC/WGBS).
- `origin_1 = epic_wgbs_additional_chr1_3` — 5,314,693 CpGs, but **restricted to chr1-3**:
  chr1=2,004,538, chr2=1,925,669, chr3=1,477,590 (near-complete WGBS/EPIC density). chr4-22 get
  only 3K-31K CpGs each (Array-probe density only, already counted in origin_0).

So the atlas was purpose-built for exactly the repo's existing three scopes: `chr1`/`chr123`
get dense WGBS/EPIC + Array, `genomewide` gets Array-only-but-everywhere. **This is not a bug in
today's published `genomewide.yaml` numbers** — they are internally correct for what `genomewide`
was defined to mean. It's a naming/scope-design gap relative to what "genomewide" should mean
going forward.

## The real CpG universe per technology (measured, not estimated)

Read directly from the canonical bundle HDF5s
(`/dune/DATASETS/MethylPredictionData/datasets/methylprophet_repro_v1/methylation/*.h5`):

| source | CpGs, all chromosomes | already NTv3-covered |
|---|---:|---|
| Array | 408,399 | 100% (all of `origin_0`) |
| EPIC | 740,296 | only the chr1-3 subset |
| WGBS | 23,047,052 | only the chr1-3 subset (~5.3M) |

WGBS dominates: it's near-complete CpG-dinucleotide density across the whole genome, same as
what a "true genomewide" scope would need. The **exact deduplicated union** of Array ∪ EPIC ∪
WGBS CpG positions has not been computed yet (EPIC and WGBS likely overlap substantially on
shared positions) — see "First concrete step" below; ~23M (WGBS's own count) is the right order
of magnitude to plan against until that's computed.

## ENCODE's CpG universe, for comparison

From the downloaded raw ENCODE dataset (`cpg_names_cpg_location.txt`,
`MethylProphet/250116_191844-241213-encode-raw` on HF, 28,301,739 CpGs total, hg38, hosted
locally at `/dune/DATASETS/MethylPredictionData/methylprophet_official/raw-encode/`):

| ENCODE scope | CpGs | already NTv3-covered | new CpGs needed |
|---|---:|---:|---:|
| chr1 | 2,329,538 | 2,004,535 (86.0%) | ~325,003 |
| chr1-3 | 6,156,289 | 5,407,793 (87.8%) | ~748,496 |
| genomewide (chr1-22) | 28,301,739 | 5,722,914 (20.2%) | ~22,578,825 |

ENCODE's genomewide gap (22.6M) and the TCGA multi-tech genomewide gap (~17.7M, see below) are
the same order of magnitude and likely overlap heavily in genomic position (both are WGBS,
both hg38) — a combined TCGA+ENCODE genomewide NTv3 job would probably cost meaningfully less
than the naive sum of the two, but this has **not been quantified** (needs the same
position-set-intersection analysis as the ENCODE-vs-atlas check, applied to
ENCODE-CpG-positions vs TCGA-WGBS-CpG-positions).

## TCGA multi-technology genomewide gap (this repo's own scope redefinition)

Using WGBS's own count (23,047,052) as the union proxy, minus the ~5.3M already densely covered
on chr1-3:

**≈ 17.7M new CpGs** need NTv3 inference to make `genomewide` a true Array+EPIC+WGBS,
all-chromosome scope. (Refine this with the exact dedup count before committing resources — see
below.)

## Compute cost: no measured GPU number exists yet

**No NTv3 atlas-generation script exists anywhere in this repo.** `cpg_statistics/{export,trainer,evaluator}.py`
only *read* the frozen atlas via `h5py`; nothing here does hg38-window extraction + NTv3 forward
pass + pooling. The atlas's own HDF5 attrs (`model=InstaDeepAI/NTv3_650M_post`,
`window_length_bp=32768`, `pooling=central_CG_mean`, `reference_genome=hg38`,
`embedding_dim=1536`, `created_utc=2026-08-12T10:52:27`) confirm it was built by an external,
not-checked-in script — this whole pipeline (model load, windowing, pooling, chunked/resumable
HDF5 writer) would need to be written from scratch in this repo.

A timing attempt (2026-08-28) could not get a clean GPU measurement — the single RTX PRO 5000
(48GB) had ~190MB actually free at the time, saturated by concurrent training runs on this
machine (including this repo's own in-progress `rna-methylation-genomewide-retrain-2026-08-28`
and a chr1 ablation run). Fell back to a real CPU measurement (48 cores): **8.66 s/CpG**,
fp32, batch=1, no batching speedup observed on CPU. Extrapolating typical CPU→GPU speedups for
this workload class (20-100x):

| scope | new CpGs | GPU-hours (20x, conservative) | GPU-hours (100x, optimistic) |
|---|---:|---:|---:|
| ENCODE chr1 | 325,003 | ~39h (~1.6 days) | ~7.8h |
| ENCODE chr1-3 | 748,496 | ~90h (~3.8 days) | ~18h |
| TCGA genomewide (all tech) | ~17.7M | ~2,130h (~89 days) | ~425h (~18 days) |
| ENCODE genomewide | ~22.6M | ~2,717h (~113 days) | ~543h (~23 days) |

**Treat every number above as order-of-magnitude, not a commitment** — it's extrapolated from a
CPU run, not measured on this GPU. The model is fp32-native (2.72GB weights); a naive `.to(bf16)`
cast breaks with an internal dtype mismatch, so real GPU throughput depends on getting
`torch.autocast`/mixed-precision handling right, which hasn't been attempted yet.

Good news for parallelizability: each CpG's window+forward pass is fully independent (no
cross-CpG state), and the atlas's flat sorted-array format (`cpg_idx`/`position`/`embedding`)
is naturally shardable by chromosome or index range — a chr-by-chr or chunked/resumable job is
architecturally easy to build, it just doesn't exist yet.

## Recommended priority order

Consistent with this repo's own documented roadmap (`chr1 → genomewide`, CLAUDE.md) and with
the much smaller/more tractable gap size:

1. **First concrete step, cheap, no GPU needed**: compute the exact deduplicated Array ∪ EPIC ∪
   WGBS CpG-position union for TCGA (currently only WGBS's own count is used as a proxy), and
   the exact TCGA-WGBS ∩ ENCODE-WGBS position overlap. This sharpens every estimate above before
   any GPU time is spent, and is pure `pandas`/`numpy` work over already-local data.
2. **ENCODE chr1** (~325K new CpGs, ~8-40 GPU-hours): smallest, matches the repo's own verified
   chr1 comparison scope, most directly extends the completed Table-7 TCGA-rows work.
3. **TCGA genomewide redefinition** (~17.7M new CpGs, ~18-89 GPU-days): the actual scope change
   requested. Large — should only start once (1) has sharpened the true gap size and a real GPU
   throughput number replaces the CPU-extrapolated one.
4. **ENCODE genomewide** (~22.6M new CpGs): similar scale to (3); if (1)'s overlap analysis shows
   heavy position sharing with TCGA WGBS, doing (3) and (4) together could be markedly cheaper
   than doing them separately.

Before starting (3) or (4), also required: write the atlas-generation pipeline itself (does not
exist yet — see above), and get a real GPU throughput measurement (needs a quiet GPU window, this
machine's single GPU is often busy with active training runs).

## What this document does NOT do

- Does not redefine `genomewide` in any config/protocol/code.
- Does not touch `results/reference/{cpg_statistics,rna_methylation}/genomewide.yaml` (still the
  Array-only numbers, still correct for what they currently claim to measure).
- Does not write, launch, or schedule any NTv3 inference job.
- Does not commit to a GPU-hours number as fact — every cost figure above is explicitly a rough
  estimate pending a real measurement.
