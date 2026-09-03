#!/usr/bin/env bash
# Vendor CpGPT + MethylGPT + DeepCpG and download their released pretrained
# checkpoints into external/ (gitignored). Network/disk only -- no GPU needed, safe
# to run with the GPU busy. Re-running is idempotent (git clone/pip install/hf/gdown
# all skip already-present content).
#
# THREE SEPARATE environments are created, not one shared env -- each model's
# dependency stack is mutually incompatible with the others:
#   - CpGPT: fine against this repo's main torch install (2.13).
#   - MethylGPT depends on `torchtext` (discontinued upstream after torch 2.3 --
#     see external/MethylGPT/docs/troubleshooting.md's version table), ABI-
#     incompatible with torch 2.13. Confirmed 2026-09-02: `torchtext==0.18.0`
#     only loads cleanly against `torch==2.3.1`, not torch>=2.9.
#   - DeepCpG needs a legacy `python=3.7`/`tensorflow==1.13.1`/`keras==1.2.2`
#     stack (Kipoi's own model.yaml pin) -- a **conda** env, not a venv (no
#     python3.7 interpreter available on this machine for venv to wrap).
# Do not try to unify any of these three.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root
mkdir -p external/checkpoints

# --- 1. Vendor all three repos -------------------------------------------------
[ -d external/CpGPT ]     || git clone --depth 1 https://github.com/lucascamillomd/CpGPT.git external/CpGPT
[ -d external/MethylGPT ] || git clone --depth 1 https://github.com/albert-ying/MethylGPT.git external/MethylGPT

# --- 2. A small venv just to drive the downloads (huggingface_hub + gdown) ----
[ -d external/.dl-venv ] || python3 -m venv external/.dl-venv
external/.dl-venv/bin/pip install --quiet huggingface_hub gdown

# --- 3. CpGPT: HuggingFace-hosted. Only the two *pretrained masked* sizes ------
#    (small=CpGPT-2M, large=CpGPT-100M) -- the other named checkpoints on
#    lucascamillomd/cpgpt-models (age/cancer/mortality/...) are task-finetuned,
#    out of scope for masked-CpG recovery.
external/.dl-venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
out = "external/checkpoints/cpgpt"
for model in ["small", "large"]:
    for filename in [f"weights/{model}.ckpt", f"config/{model}.yaml", f"vocab/{model}.json"]:
        hf_hub_download(repo_id="lucascamillomd/cpgpt-models", filename=filename, local_dir=out)
# DNA-embedding + Illumina-metadata dependencies needed for real genomic-locations inference (~6.2GB).
snapshot_download(repo_id="lucascamillomd/cpgpt-human-dependencies", local_dir="external/checkpoints/cpgpt_human_dependencies")
PY

# --- 4. MethylGPT: Google-Drive-hosted, all three released sizes --------------
#    NOTE: the internal .pt filenames don't match their folder/variant name
#    (leftover training-run naming) -- resolve_resources() in the adapter takes
#    the first *.pt in each folder and check_variant_identity() cross-checks
#    args.json's layer_size against the README table rather than trusting names.
declare -A GDRIVE_IDS=(
  [base]=1kWdmkkVQpU17uzUC6-wpNR_4UEdxGx6k
  [medium]=14M4wdS83el9PAgh9TdfjSCeEcDPbz34f
  [large]=1lt8SF9MvoytPN3DeaxIss_ED9zNpf_Le
)
mkdir -p external/checkpoints/methylgpt
for variant in base medium large; do
  dest="external/checkpoints/methylgpt/${variant}"
  if [ ! -d "$dest" ] || [ -z "$(ls -A "$dest" 2>/dev/null)" ]; then
    external/.dl-venv/bin/gdown --folder "https://drive.google.com/drive/folders/${GDRIVE_IDS[$variant]}" -O "$dest"
  fi
done

# --- 5. Isolated per-model inference venvs ------------------------------------
if [ ! -d external/cpgpt-env ]; then
  python3 -m venv external/cpgpt-env
  external/cpgpt-env/bin/pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch torchvision
  external/cpgpt-env/bin/pip install --quiet \
    transformers lightning fastapi torchmetrics hydra-core hydra-colorlog rich loguru \
    pyfaidx sqlitedict biopython schedulefree rootutils boto3 ipywidgets lifelines \
    pyarrow huggingface-hub scikit-learn
fi

if [ ! -d external/methylgpt-env ]; then
  python3 -m venv external/methylgpt-env
  # This exact pin (not requirements.txt's torch==2.1.0/torchtext==0.16.0) is the
  # pair actually verified working in this session -- either compatible pair from
  # docs/troubleshooting.md's table is fine, just don't mix across the pair.
  external/methylgpt-env/bin/pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 torchtext==0.18.0
  external/methylgpt-env/bin/pip install --quiet \
    numpy pandas scipy scikit-learn lightning tqdm datasets scib ipython
fi

# --- 6. DeepCpG: DNA-only submodel, human variants (Hou et al. 2016) ----------
#    DeepCpG's own download host (and its own dcpg_download.py) is dead
#    (http://www.ebi.ac.uk/~angermue/deepcpg/alias/..., confirmed HTTP 500,
#    2026-09-02) -- Kipoi (kipoi.org) is the live mirror; its model.yaml files
#    point at a stable Zenodo record (1466079) for the actual weights/arch.
[ -d external/deepcpg ] || git clone --depth 1 https://github.com/cangermueller/deepcpg.git external/deepcpg

declare -A DEEPCPG_ZENODO_NAME=(
  [hou2016_hcc_dna]=Hou2016_HCC_dna
  [hou2016_hepg2_dna]=Hou2016_HepG2_dna
)
mkdir -p external/checkpoints/deepcpg
for variant in hou2016_hcc_dna hou2016_hepg2_dna; do
  dest="external/checkpoints/deepcpg/${variant}"
  mkdir -p "$dest"
  name="${DEEPCPG_ZENODO_NAME[$variant]}"
  [ -f "$dest/arch.json" ]    || curl -sL "https://zenodo.org/record/1466079/files/${name}-model?download=1" -o "$dest/arch.json"
  [ -f "$dest/weights.h5" ]   || curl -sL "https://zenodo.org/record/1466079/files/${name}-model_weights.h5?download=1" -o "$dest/weights.h5"
done

# DeepCpG needs a legacy stack (Kipoi's own model.yaml pin: python=3.7,
# tensorflow==1.13.1, keras==1.2.2) -- no python3.7 interpreter is available on
# this machine for a venv to wrap, so this one is a **conda** env, not a venv.
if ! conda env list 2>/dev/null | grep -q '^deepcpg-env '; then
  conda create -y -n deepcpg-env python=3.7
  conda run -n deepcpg-env pip install --quiet tensorflow==1.13.1 keras==1.2.2 "h5py==2.10.0" "protobuf==3.20"
fi

echo "Setup complete. Next: run scripts/benchmark_foundation_models/check_readiness.py"
echo "  external/cpgpt-env/bin/python scripts/benchmark_foundation_models/check_readiness.py --model cpgpt"
echo "  external/methylgpt-env/bin/python scripts/benchmark_foundation_models/check_readiness.py --model methylgpt"
echo "  conda run -n deepcpg-env python scripts/benchmark_foundation_models/check_readiness.py --model deepcpg [--fasta-path /path/to/hg38.fa]"
