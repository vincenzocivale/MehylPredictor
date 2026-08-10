#!/usr/bin/env bash
# Full current-model experiment suite:
#   E2 = Array+EPIC+WGBS chr1, all canonical auxiliary loci
#   E3 = Array+EPIC+WGBS chr1-3, all canonical auxiliary loci
#   E4 = Array genome-wide (408,399 CpGs; official train/heldout CpG pools)
#
# E1 is deliberately NOT touched.  Run this beside an existing E1 process by
# assigning only free GPUs, e.g. GPUS=1,2 MAX_GPUS=2.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

CANONICAL_ROOT="${TCGA_CANONICAL_ROOT:-/raid/DATASETS/MethylPredictionData/methylprophet_official/official_training_data}"
BASE_EMBEDDINGS="${LOCUS_EMBEDDINGS:-/raid/DATASETS/MethylPredictionData/locus_embeddings.h5}"
BASE_FEATURES="${LOCUS_FEATURES:-/raid/DATASETS/MethylPredictionData/locus_features.parquet}"
GENOMEWIDE_CPG_SPLIT="${GENOMEWIDE_CPG_SPLIT:-/raid/DATASETS/MethylPredictionData/cpg_split_manifest.parquet}"
BASE_CONFIG="${BASE_CONFIG:-configs/train.yaml}"
HG38_FASTA="${HG38_FASTA:-}"
SEED="${SEED:-17}"
DATA_ROOT="${DATA_ROOT:-/raid/DATASETS/MethylPredictionData}"
FEATURE_ROOT="${FEATURE_ROOT:-$DATA_ROOT/genomic_features}"
DERIVED_ROOT="${DERIVED_ROOT:-$DATA_ROOT/derived}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$DATA_ROOT/experiments}"
NTV3_STORE="${NTV3_STORE:-$FEATURE_ROOT/ntv3_650m_post/hg38_L32768_forward_cpg_center}"
PRIOR_STORE="${PRIOR_STORE:-$FEATURE_ROOT/methylation_prior/ntv3_650m_post/official_array_train_distilled_current_features}"
CACHE_ROOT="${CACHE_ROOT:-$DERIVED_ROOT/tcga_canonical}"
RUN_ROOT="${RUN_ROOT:-$EXPERIMENT_ROOT/current_model/full_e2_e4/seed${SEED}}"
GPUS_CSV="${GPUS:-1,2}"
MAX_GPUS="${MAX_GPUS:-99}"
SOURCE_POLICY="${SOURCE_POLICY:-equal_source}"
HOLDOUT_POLICY="${HOLDOUT_POLICY:-mp_matched}"
MIXED_STEPS_PER_EPOCH="${MIXED_STEPS_PER_EPOCH:-128}"
NTV3_BATCH_SIZE="${NTV3_BATCH_SIZE:-4}"
NTV3_SHARD_SIZE="${NTV3_SHARD_SIZE:-25000}"
NTV3_STORAGE_DTYPE="${NTV3_STORAGE_DTYPE:-float32}"
RUN_E2="${RUN_E2:-1}"
RUN_E3="${RUN_E3:-1}"
RUN_E4="${RUN_E4:-1}"
RUN_STRICT_OOD_ABLATION="${RUN_STRICT_OOD_ABLATION:-0}"
AUTO_DOWNLOAD_MP_EVAL="${AUTO_DOWNLOAD_MP_EVAL:-1}"
MP_EVAL_REPO="${MP_EVAL_REPO:-MethylProphet/eval-tcga_array_chr1-32xl40s-c2b2}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/runs" "$NTV3_STORE" "$PRIOR_STORE" "$CACHE_ROOT"
exec > >(tee -a "$RUN_ROOT/logs/full_suite.log") 2>&1

echo "=== E2-E4 full suite ==="
date -Is
echo "repo=$REPO"
echo "git_head=$(git rev-parse HEAD)"
echo "run_root=$RUN_ROOT"
echo "ntv3_store=$NTV3_STORE"
echo "prior_store=$PRIOR_STORE"
echo "cache_root=$CACHE_ROOT"
echo "source_policy=$SOURCE_POLICY holdout_policy=$HOLDOUT_POLICY"

if [[ -z "$HG38_FASTA" || ! -f "$HG38_FASTA" ]]; then
  echo "FATAL: set HG38_FASTA to the indexed GRCh38/hg38 FASTA used for NTv3 extraction." >&2
  echo "Example: HG38_FASTA=/path/to/hg38.fa" >&2
  exit 2
fi
for path in "$CANONICAL_ROOT" "$BASE_EMBEDDINGS" "$BASE_FEATURES" "$GENOMEWIDE_CPG_SPLIT" "$BASE_CONFIG"; do
  [[ -e "$path" ]] || { echo "FATAL missing required path: $path" >&2; exit 2; }
done

python - <<'PY'
mods = ["pyarrow", "h5py", "torch", "transformers", "huggingface_hub", "pyfaidx"]
for name in mods:
    try:
        __import__(name)
    except Exception as exc:
        raise SystemExit(f"FATAL: missing/incompatible {name}: {exc}. Install requirements-genomics.txt without replacing torch.")
import torch
if not torch.cuda.is_available():
    raise SystemExit("FATAL: CUDA unavailable")
print("python dependency/CUDA preflight: PASS")
PY

IFS=',' read -r -a ALL_GPUS <<< "$GPUS_CSV"
GPUS=()
for g in "${ALL_GPUS[@]}"; do
  [[ -n "$g" ]] || continue
  GPUS+=("$g")
  [[ "${#GPUS[@]}" -ge "$MAX_GPUS" ]] && break
done
if [[ "${#GPUS[@]}" -lt 1 ]]; then echo "FATAL: no GPUs selected" >&2; exit 2; fi
for g in "${GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="$g" python - <<'PY'
import torch
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
x=torch.randn((1024,1024),device='cuda'); (x@x).sum().item()
print('CUDA smoke PASS')
PY
done

echo "selected physical GPUs: ${GPUS[*]}"

# Build the FASTA index exactly once in a writable run-local directory.  The
# symlink keeps the FASTA bytes untouched while pyfaidx writes .fai/.gzi next
# to the link instead of inside a potentially read-only reference directory.
REF_CACHE="$CACHE_ROOT/reference_index/hg38"
mkdir -p "$REF_CACHE"
FASTA_LOCAL="$REF_CACHE/$(basename "$HG38_FASTA")"
if [[ ! -e "$FASTA_LOCAL" ]]; then
  ln -s "$(readlink -f "$HG38_FASTA")" "$FASTA_LOCAL"
fi
python - "$FASTA_LOCAL" <<'PY'
from pyfaidx import Fasta
import sys
g = Fasta(sys.argv[1], as_raw=True, sequence_always_upper=True, rebuild=True)
print(f"FASTA index ready: records={len(g.keys())}")
g.close()
PY
HG38_FASTA="$FASTA_LOCAL"

BASE_CACHE="$CACHE_ROOT/base_array_features_${NTV3_STORAGE_DTYPE}"
RNA_CACHE="$CACHE_ROOT/rna_official_array_train_zscore"
UNIVERSE="$NTV3_STORE/universe/missing_tcga_mix_chr123.h5"
SHARDS="$NTV3_STORE/shards"
EXPANDED_EMBEDDINGS="$NTV3_STORE/merged"
PROBE="$PRIOR_STORE/probe"
EXPANDED_FEATURES="$PRIOR_STORE/expanded"
COMPACT="$CACHE_ROOT/compact_methylation/tcga_mix_chr123"
mkdir -p "$SHARDS" "$EXPANDED_EMBEDDINGS" "$PROBE" "$EXPANDED_FEATURES" "$COMPACT"

# 1. Pure representation caches; no model/data semantics change.
echo "=== 1/8 base + RNA caches ==="
python scripts/full_suite.py prepare-base-cache --embeddings "$BASE_EMBEDDINGS" --features "$BASE_FEATURES" --output "$BASE_CACHE" --storage-dtype "$NTV3_STORAGE_DTYPE"
python scripts/full_suite.py prepare-rna-cache --canonical-root "$CANONICAL_ROOT" --output "$RNA_CACHE"

# 2. Build E3 universe once; E2 is its chr1 subset.
echo "=== 2/8 required new CpG universe ==="
python scripts/full_suite.py prepare-universe --canonical-root "$CANONICAL_ROOT" --embeddings "$BASE_EMBEDDINGS" \
  --protocol tcga_mix_chr123 --shard-size "$NTV3_SHARD_SIZE" --output "$UNIVERSE"

# 3. Persistent NTv3 worker per selected GPU.  Each process takes every Nth shard.
echo "=== 3/8 multi-GPU NTv3-650M-post extraction ==="
WORLD="${#GPUS[@]}"
PIDS=()
for ((rank=0; rank<WORLD; rank++)); do
  gpu="${GPUS[$rank]}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python scripts/full_suite.py extract-ntv3-worker \
      --universe "$UNIVERSE" --fasta "$HG38_FASTA" --output "$SHARDS" \
      --rank "$rank" --world-size "$WORLD" --batch-size "$NTV3_BATCH_SIZE" --device cuda --storage-dtype "$NTV3_STORAGE_DTYPE"
  ) > "$RUN_ROOT/logs/ntv3_gpu${gpu}.log" 2>&1 &
  PIDS+=("$!")
  echo "NTv3 worker rank=$rank physical_gpu=$gpu pid=$!"
done
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
[[ "$FAIL" == 0 ]] || { echo "FATAL: at least one NTv3 worker failed; inspect logs/ntv3_gpu*.log" >&2; exit 3; }

# 4. Merge once into mmap-friendly storage.
echo "=== 4/8 merge expanded embeddings ==="
python scripts/full_suite.py merge-ntv3 --universe "$UNIVERSE" --shards "$SHARDS" --output "$EXPANDED_EMBEDDINGS" --storage-dtype "$NTV3_STORAGE_DTYPE"

# 5. Original upstream probe checkpoints were not available on this host.
# Distil the exact existing frozen feature map; base CpGs stay bit-identical.
echo "=== 5/8 fit + apply frozen feature-extension probe ==="
PROBE_GPU="${GPUS[0]}"
CUDA_VISIBLE_DEVICES="$PROBE_GPU" python scripts/full_suite.py fit-feature-probe \
  --base-cache "$BASE_CACHE" --output "$PROBE" --device cuda --cpg-split "$GENOMEWIDE_CPG_SPLIT"
CUDA_VISIBLE_DEVICES="$PROBE_GPU" python scripts/full_suite.py infer-expanded-features \
  --embeddings "$EXPANDED_EMBEDDINGS" --probe "$PROBE/feature_extension_probe.pt" \
  --output "$EXPANDED_FEATURES" --device cuda

# 6. Build small row-friendly Array/EPIC chr1-3 caches once.  Do this after NTv3
# so it does not compete for methylation I/O with an E1 that may still be running.
echo "=== 6/8 compact Array/EPIC training caches ==="
python scripts/full_suite.py build-compact-cache --canonical-root "$CANONICAL_ROOT" --protocol tcga_mix_chr123 --output "$COMPACT"
ARRAY_CACHE="$COMPACT/array_tcga_mix_chr123.h5"
EPIC_CACHE="$COMPACT/epic_tcga_mix_chr123.h5"

train_one () {
  local protocol="$1" output="$2" holdout="$3" gpu="$4"
  local expansion_args=()
  [[ "$protocol" != "array_genomewide" ]] && expansion_args=(
    --expanded-embeddings "$EXPANDED_EMBEDDINGS"
    --expanded-features "$EXPANDED_FEATURES"
  )
  local genome_args=()
  [[ "$protocol" == "array_genomewide" ]] && genome_args=(--genomewide-cpg-split "$GENOMEWIDE_CPG_SPLIT")
  local cache_args=()
  if [[ "$protocol" != "array_genomewide" ]]; then cache_args=(--array-cache "$ARRAY_CACHE" --epic-cache "$EPIC_CACHE"); fi
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/full_suite.py train \
    --canonical-root "$CANONICAL_ROOT" --protocol "$protocol" --base-config "$BASE_CONFIG" \
    --base-cache "$BASE_CACHE" "${expansion_args[@]}" --rna-cache "$RNA_CACHE" --output "$output" \
    --source-policy "$SOURCE_POLICY" --holdout-policy "$holdout" --seed "$SEED" \
    --steps-per-epoch "$MIXED_STEPS_PER_EPOCH" "${genome_args[@]}" "${cache_args[@]}"
}

# 7. Saturate available GPUs without touching any unlisted device.
echo "=== 7/8 E2/E3/E4 training + full evaluation ==="
TRAIN_PIDS=()
if [[ "$WORLD" -ge 3 ]]; then
  if [[ "$RUN_E2" == 1 ]]; then
    (train_one tcga_mix_chr1 "$RUN_ROOT/runs/E2_mix_chr1_${SOURCE_POLICY}_${HOLDOUT_POLICY}" "$HOLDOUT_POLICY" "${GPUS[0]}") > "$RUN_ROOT/logs/E2.log" 2>&1 &
    TRAIN_PIDS+=("$!")
  fi
  if [[ "$RUN_E3" == 1 ]]; then
    (train_one tcga_mix_chr123 "$RUN_ROOT/runs/E3_mix_chr123_${SOURCE_POLICY}_${HOLDOUT_POLICY}" "$HOLDOUT_POLICY" "${GPUS[1]}") > "$RUN_ROOT/logs/E3.log" 2>&1 &
    TRAIN_PIDS+=("$!")
  fi
  if [[ "$RUN_E4" == 1 ]]; then
    (train_one array_genomewide "$RUN_ROOT/runs/E4_array_genomewide" strict_global "${GPUS[2]}") > "$RUN_ROOT/logs/E4.log" 2>&1 &
    TRAIN_PIDS+=("$!")
  fi
elif [[ "$WORLD" -eq 2 ]]; then
  # Queue E3 directly behind E2 on GPU[0]; E4 owns GPU[1].
  (
    [[ "$RUN_E2" != 1 ]] || train_one tcga_mix_chr1 "$RUN_ROOT/runs/E2_mix_chr1_${SOURCE_POLICY}_${HOLDOUT_POLICY}" "$HOLDOUT_POLICY" "${GPUS[0]}"
    [[ "$RUN_E3" != 1 ]] || train_one tcga_mix_chr123 "$RUN_ROOT/runs/E3_mix_chr123_${SOURCE_POLICY}_${HOLDOUT_POLICY}" "$HOLDOUT_POLICY" "${GPUS[0]}"
  ) > "$RUN_ROOT/logs/E2_E3_gpu${GPUS[0]}.log" 2>&1 & TRAIN_PIDS+=("$!")
  if [[ "$RUN_E4" == 1 ]]; then
    (train_one array_genomewide "$RUN_ROOT/runs/E4_array_genomewide" strict_global "${GPUS[1]}") > "$RUN_ROOT/logs/E4_gpu${GPUS[1]}.log" 2>&1 &
    TRAIN_PIDS+=("$!")
  fi
else
  (
    [[ "$RUN_E2" != 1 ]] || train_one tcga_mix_chr1 "$RUN_ROOT/runs/E2_mix_chr1_${SOURCE_POLICY}_${HOLDOUT_POLICY}" "$HOLDOUT_POLICY" "${GPUS[0]}"
    [[ "$RUN_E3" != 1 ]] || train_one tcga_mix_chr123 "$RUN_ROOT/runs/E3_mix_chr123_${SOURCE_POLICY}_${HOLDOUT_POLICY}" "$HOLDOUT_POLICY" "${GPUS[0]}"
    [[ "$RUN_E4" != 1 ]] || train_one array_genomewide "$RUN_ROOT/runs/E4_array_genomewide" strict_global "${GPUS[0]}"
  ) > "$RUN_ROOT/logs/E2_E3_E4_gpu${GPUS[0]}.log" 2>&1 & TRAIN_PIDS+=("$!")
fi
FAIL=0
for pid in "${TRAIN_PIDS[@]}"; do wait "$pid" || FAIL=1; done
[[ "$FAIL" == 0 ]] || { echo "FATAL: one or more training queues failed; inspect $RUN_ROOT/logs" >&2; exit 4; }

# Optional strict-global OOD ablation: intentionally separate from the primary
# mp_matched run because it changes what "val CpG" means relative to MP training.
if [[ "$RUN_STRICT_OOD_ABLATION" == 1 && "$HOLDOUT_POLICY" != "strict_global" ]]; then
  echo "=== strict-global OOD ablation ==="
  train_one tcga_mix_chr1 "$RUN_ROOT/runs/E2_mix_chr1_${SOURCE_POLICY}_strict_global" strict_global "${GPUS[0]}"
  train_one tcga_mix_chr123 "$RUN_ROOT/runs/E3_mix_chr123_${SOURCE_POLICY}_strict_global" strict_global "${GPUS[0]}"
fi

# 8. Exact released MP paired comparison for E2 chr1.  The evaluation adapter
# uses the original base Array features because all official chr1 eval loci are
# already in the 408,399-row store; mixed training only changes model weights.
echo "=== 8/8 E2 exact MethylProphet paired comparison (optional) ==="
E2_RUN="$RUN_ROOT/runs/E2_mix_chr1_${SOURCE_POLICY}_${HOLDOUT_POLICY}"
if [[ "$RUN_E2" == 1 && -f "$E2_RUN/final_refit/best.pt" ]]; then
  ADAPTER="$E2_RUN/mp_eval_adapter"
  python scripts/full_suite.py write-chr1-eval-adapter --canonical-root "$CANONICAL_ROOT" --base-config "$BASE_CONFIG" \
    --embeddings "$BASE_EMBEDDINGS" --features "$BASE_FEATURES" --output "$ADAPTER"
  MP_DIR="${MP_EVAL_DIR:-}"
  if [[ -z "$MP_DIR" && "$AUTO_DOWNLOAD_MP_EVAL" == 1 ]]; then
    MP_DIR="$CACHE_ROOT/methylprophet_released_eval"
    mkdir -p "$MP_DIR"
    if command -v hf >/dev/null 2>&1; then
      hf download "$MP_EVAL_REPO" --repo-type dataset --local-dir "$MP_DIR" || MP_DIR=""
    elif command -v huggingface-cli >/dev/null 2>&1; then
      huggingface-cli download "$MP_EVAL_REPO" --repo-type dataset --local-dir "$MP_DIR" || MP_DIR=""
    else
      MP_DIR=""
    fi
  fi
  ARGS=(--config "$ADAPTER/eval_config.yaml" --checkpoint "$E2_RUN/final_refit/best.pt" --output "$E2_RUN/evaluation/vs_methylprophet.json")
  [[ -z "$MP_DIR" ]] || ARGS+=(--mp-eval "$MP_DIR" --mp-label "$MP_EVAL_REPO")
  python scripts/evaluate_current_model_vs_methylprophet.py "${ARGS[@]}"
fi

echo "=== COMPLETE ==="
date -Is
echo "Results under: $RUN_ROOT/runs"
echo "Reusable NTv3 embeddings: $EXPANDED_EMBEDDINGS"
echo "Reusable prior/variability: $EXPANDED_FEATURES"
