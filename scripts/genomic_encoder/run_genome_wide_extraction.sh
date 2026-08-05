#!/usr/bin/env bash
# Fase 4: genome-wide NTv3 embedding extraction, forward-only, one chromosome
# per invocation. Long-running (~22-25 GPU-hours) -- meant to run unattended
# in the background. Stops on first failure (set -e) so a partial run is
# never silently treated as complete; per-chromosome outputs already written
# before a failure are left in place for inspection/resume.
#
# Targets are built per-chromosome (--chromosomes chrN), not as one combined
# genome-wide read: a single 408,399-row x 8,260-column me.parquet read got
# SIGKILLed twice (~90s in, both times) -- almost certainly a sandbox memory
# cgroup limit, not a harness/session issue (dmesg shows no host-level OOM,
# free -h had 55GB free at launch). The largest chromosome (chr1, 40,627
# CpGs) was already validated standalone at ~85s with no memory issue, so
# per-chromosome keeps peak memory bounded throughout.
set -euo pipefail
cd /home/oem/projects/methylation/MethylProphetTest

PYTHON=/home/oem/miniconda3/envs/methil-predictor/bin/python
SCRATCH=/data/dataset/methylation/genomic_encoder_genome_wide_scratch
FASTADIR="$SCRATCH/reference/per_chromosome"
BASE=/data/dataset/methylation/MethylProphetData/parquet/241231-tcga_array/metadata/cpg_split/index_files
CPGMAP=third_party/MethylProphet/data/examples/tcga_mix_chr1/cpg_chr_pos_df.parquet
REF=artifacts/genomic_encoder/static_prior/reference

mkdir -p "$SCRATCH/targets_per_chromosome"

CHROMOSOMES="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22"

for chrom in $CHROMOSOMES; do
  TARGETS="$SCRATCH/targets_per_chromosome/targets_${chrom}.parquet"
  FEATURES="$SCRATCH/features_${chrom}.parquet"
  EMBED_DIR="$SCRATCH/embeddings_${chrom}"
  DONE_MARKER="$EMBED_DIR/.done"

  if [ -f "$DONE_MARKER" ]; then
    echo "[genome-wide] $(date -u +%FT%TZ) $chrom already done, skipping" >&2
    continue
  fi

  if [ ! -f "$TARGETS" ]; then
    echo "[genome-wide] $(date -u +%FT%TZ) === $chrom: build_genome_wide_targets ===" >&2
    "$PYTHON" -m methylation_predictor.genomic_encoder.build_genome_wide_targets \
      --cpg-map "$CPGMAP" \
      --me-parquet /data/dataset/methylation/MethylProphetData/parquet/241231-tcga_array/me.parquet \
      --train-manifest artifacts/genomic_encoder/static_prior/train_sample_manifest.parquet \
      --sample-metadata artifacts/diagnostics/methylprophet/locus_dominance/tcga_metadata/sample_idx_cancer_type.parquet \
      --chromosomes "$chrom" \
      --official-train-cpg "$BASE/train.parquet" \
      --official-val-cpg "$BASE/val.parquet" \
      --output "$TARGETS"
  fi

  echo "[genome-wide] $(date -u +%FT%TZ) === $chrom: build-features ===" >&2
  "$PYTHON" -m methylation_predictor.genomic_encoder.cli build-features \
    --prior "$TARGETS" \
    --cpg-map "$CPGMAP" \
    --fasta "$FASTADIR/${chrom}.fa.gz" \
    --cpg-islands "$REF/cpgIslandExt_hg38.txt.gz" \
    --gencode-gtf "$REF/gencode.v41.annotation.gtf.gz" \
    --within-stats "$TARGETS" \
    --output "$FEATURES"

  echo "[genome-wide] $(date -u +%FT%TZ) === $chrom: extract-ntv3 ===" >&2
  "$PYTHON" -m methylation_predictor.genomic_encoder.cli extract-ntv3 \
    --input "$FEATURES" \
    --fasta "$FASTADIR/${chrom}.fa.gz" \
    --output-dir "$EMBED_DIR" \
    --checkpoint InstaDeepAI/NTv3_650M_post \
    --lengths 32768 \
    --batch-size 4 \
    --device cuda \
    --bf16 \
    --orientations forward

  touch "$DONE_MARKER"
  echo "[genome-wide] $(date -u +%FT%TZ) === $chrom: done ===" >&2
done

echo "[genome-wide] $(date -u +%FT%TZ) ALL CHROMOSOMES COMPLETE" >&2
