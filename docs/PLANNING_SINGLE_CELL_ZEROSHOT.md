# Planning: zero-shot single-cell RNA→methylation generalization (exploration only)

**Status: planning-only, nothing executed.** This mirrors
`docs/PLANNING_NTV3_GENOMEWIDE_EXPANSION.md` — a desk-research pass to decide whether a
formal experiment is worth building, not a committed roadmap item. No code, config, or
`results/reference/` changes accompany this document.

## Motivating question

`RNAMethylationPredictor` and MethylProphet are both trained and evaluated exclusively on
bulk TCGA RNA + bulk WGBS. Does either model generalize **zero-shot** (no fine-tuning) to
single-cell paired RNA+methylation data, at **per-cell** resolution? A positive result
would be strong evidence that the learned RNA→methylation relationship is real regulatory
biology rather than an artifact of bulk-tissue signal averaging.

## Candidate datasets evaluated

### snmC2T-seq / snmCAT-seq — human frontal cortex + cell lines (GEO **GSE140493**) — top candidate

Source: Luo et al., "Single nucleus multi-omics identifies human cortical cell regulatory
genome diversity" (bioRxiv 2019.12.11.873398 → Cell Genomics 2022,
[cell.com/cell-genomics/fulltext/S2666-979X(22)00027-1](https://www.cell.com/cell-genomics/fulltext/S2666-979X(22)00027-1)).

- **True per-nucleus pairing confirmed**: DNA methylation, chromatin accessibility, and
  RNA are captured from the *same* single nucleus (5-methyl-dCTP incorporation during
  reverse transcription molecularly partitions RNA from gDNA before separate library
  prep) — this is the property a genuine zero-shot per-cell test requires, and several
  other "multi-omic atlas" datasets found in the initial search (sn-m3C-seq Human Body
  Atlas, adipose tissue atlas GSE297267) were **not** confirmed to have this property —
  they may pair modalities per-tissue/donor rather than per-cell, which would not support
  this test. Deprioritized unless snmC2T-seq turns out inadequate.
- **Human, real scale**: 4,358 nuclei from postmortem human frontal cortex (two donors),
  spanning 63 cortical cell types — plus a cell-line validation set (H1 ESCs, HEK293T).
- **Species/tissue match to TCGA**: cortex is a different tissue from most TCGA cancer
  types, and it's normal (non-tumor) tissue — a real generalization test (out-of-tissue,
  out-of-modality), not a comfortable one. Worth flagging to the user as a feature, not a
  bug, for a "does it generalize at all" question, but means results won't be directly
  comparable to any TCGA-tumor-type-matched claim.
- **Open questions not resolved by desk research** — need direct inspection of the GEO
  series' supplementary files before committing to a formal experiment:
  - Per-nucleus CpG coverage (i.e., how many CpGs per cell fall inside our chr1/chr123
    NTv3-embedding scope) — not stated in the abstract/methods excerpts fetched. The
    paper's preprocessing bins methylation into 100kb genomic bins for downstream
    clustering, which suggests native per-nucleus CpG-level coverage may be sparse
    enough that the authors themselves work at bin resolution, not raw per-CpG calls —
    a signal worth taking seriously before assuming per-cell, per-CpG comparison is
    viable.
  - Genome build/reference used for methylation calls (needed to map onto our CpG
    identity space — see `tcga_canonical/`, `docs/DATA.md`).
  - Exact processed file format deposited under GSE140493 (per-cell methylation
    call files vs. only aggregated matrices) — determines how much parsing work is
    needed before any comparison is possible.
  - Gene identifier scheme for the paired RNA counts, and its overlap with what our
    RNA cache/RNA encoder expects.

### scNMT-seq (GEO GSE109262) — ruled out

Mouse embryonic stem cells only (70 + 43 cells across two datasets); no human data. Not
usable for a claim about generalization from a human-tumor-trained model. True per-cell
RNA+methylation+accessibility pairing is confirmed methodologically, but species mismatch
is disqualifying for this purpose.

### scM&T-seq (GEO GSE74535) — deprioritized

Small (61 cells total in the original Angermueller et al. 2016 study), and predominantly
mouse ESC-focused as a method-validation dataset. Even if a human subset exists, N is too
small to support a meaningful correlation analysis at single-cell resolution. Not worth
pursuing unless snmC2T-seq turns out completely inaccessible.

## Blockers identified, and whether they're hard

1. **Per-cell CpG sparsity vs. continuous `mu`/`sigma` predictions** — likely the single
   biggest risk to the "cell-per-cell" comparison mode the user specifically wants. Not
   yet quantified for snmC2T-seq specifically (see open questions above); this must be
   checked with real downloaded data before further planning, not assumed either way.
   *Not a hard blocker on its own* — if per-cell coverage is too sparse for a meaningful
   correlation, aggregating multiple CpGs per gene/region within a cell (still per-cell,
   not per-tissue) may rescue enough signal without abandoning the "same single cell"
   property the user asked for.
2. **RNA dropout / normalization mismatch** — solvable preprocessing (needs matching our
   RNA cache's normalization, not a redesign).
3. **CpG genome-build/identity mapping** — solvable with a liftover step if the paper
   used a different build than our canonical bundle; needs confirming which build.
4. **Tissue/domain mismatch (cortex vs. TCGA tumor types)** — not a blocker, arguably the
   point of a generalization test, but changes how any result should be framed (out-of-
   distribution generalization claim, not a benchmark-matched comparison).
5. **Access** — GSE140493 is a standard public GEO series, no DUA/registration expected;
   not yet confirmed the exact supplementary file sizes.

## Recommendation

Worth a next step, but **not yet ready to become a formal `docs/PAPER_EXPERIMENTS.md`
entry**. Before committing to build any pipeline:

1. Actually download GSE140493's processed supplementary files (not just the abstract)
   and directly measure: per-nucleus CpG coverage inside chr1, genome build, RNA count
   matrix gene ID scheme, and the exact human-cortex nucleus count with both modalities
   present and QC-passing.
2. Only after that inspection, decide whether the comparison mode is:
   - Pure per-cell, per-CpG (user's preferred mode) — viable only if raw coverage is
     high enough;
   - Per-cell but aggregated over CpGs-per-gene/region (still single-cell, more robust
     to sparsity) — likely fallback if (a) isn't viable;
   - Not viable at all with this dataset, in which case re-examine scM&T-seq's human
     subset or search for newer (2024–2025) higher-coverage single-cell WGBS-scale
     datasets (e.g. sciMETv3, GSE-level atlases) explicitly for per-cell RNA pairing.

This document should be updated (not superseded) once the actual GSE140493 files have
been inspected, before any formal experiment plan is written.
