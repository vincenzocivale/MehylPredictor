"""Adapter around DeepCpG's DNA-only submodel (Angermueller et al. 2017,
github.com/cangermueller/deepcpg).

Unlike CpGPT/MethylGPT, DeepCpG's own download host and its own `dcpg_download.py`
both point at a dead EBI host (`http://www.ebi.ac.uk/~angermue/deepcpg/alias/...`,
confirmed HTTP 500, 2026-09-02). The live mirror used here is **Kipoi**
(kipoi.org), whose `model.yaml` files point to a stable **Zenodo record (1466079)**
for the actual architecture JSON + weights HDF5.

Only the two **human, DNA-sequence-only** variants are used
(`hou2016_hcc_dna`, `hou2016_hepg2_dna`, from Hou et al. 2016 scRRBS-seq data) --
Kipoi's own model card documents that these CpG positions were already lifted from
GRCh37 to **GRCh38**, matching this repo's own hg38-based canonical bundle (no
liftover needed). The CpG-neighbor and Joint DeepCpG modules are deliberately not
used: they're hard-wired to the exact number and identity of cells in their
original training dataset, so they don't transfer to arbitrary new samples the way
the DNA-only module (sequence window in, methylation probability out) does.

**This model is inherently patient-agnostic** -- it only sees a CpG's local hg38
DNA sequence, never any per-sample input (RNA, other CpGs, ...). Its prediction is
the same for every one of this repo's patients at a given CpG, conceptually closer
to this repo's own `CpG Prior` baseline (`rna_training/evaluator.py`'s
`CpGPriorEvaluator`) than to CpGPT/MethylGPT's per-patient masked-recovery framing
-- report it as such, not as a like-for-like comparison. It was also trained on
single-cell data; scoring bulk TCGA methylation zero-shot may simply transfer
poorly, which would be a legitimate result, not a pipeline bug.

Import this module with `external/deepcpg-env`'s python (a **conda** env, not a
venv -- no python3.7 interpreter is available on this machine for venv to wrap;
Kipoi's own model.yaml pins `python=3.7`/`tensorflow==1.13.1`/`keras==1.2.2`, a
legacy stack that cannot share an environment with either the main repo's torch
install or CpGPT/MethylGPT's venvs).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

VALID_VARIANTS = ("hou2016_hcc_dna", "hou2016_hepg2_dna")
DNA_WINDOW_LEN = 1001  # fixed by the released architecture's input shape (1001, 4)
_BASE_TO_ROW = {"A": 0, "C": 1, "G": 2, "T": 3}


@dataclass
class DeepCpGResources:
    variant: str
    arch_path: Path
    weights_path: Path


def resolve_resources(checkpoints_dir: str | Path, variant: str) -> DeepCpGResources:
    if variant not in VALID_VARIANTS:
        raise ValueError(f"Unknown DeepCpG variant '{variant}', expected one of {VALID_VARIANTS}")
    root = Path(checkpoints_dir) / variant
    resources = DeepCpGResources(variant=variant, arch_path=root / "arch.json", weights_path=root / "weights.h5")
    missing = [p for p in (resources.arch_path, resources.weights_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing DeepCpG '{variant}' resource file(s): {missing}")
    return resources


def one_hot_dna_window(fasta_path: str | Path, chrom: str, pos: int, wlen: int = DNA_WINDOW_LEN) -> np.ndarray:
    """One-hot-encode the `wlen`-bp hg38 window centered on `pos` (1-based, as this
    repo's CpG registry stores it -- `pyfaidx` is 0-based half-open, converted here).

    Pure numpy + `pyfaidx` (already a repo dependency, `requirements-genomics.txt`)
    -- no DeepCpG/Keras/TensorFlow import, so this is testable in the main repo env
    without the legacy `deepcpg-env`. Returns shape `(wlen, 4)`, column order A/C/G/T
    (matching the released model's `schema.inputs.dna` convention), all-zero rows for
    N or out-of-bounds bases.
    """
    import pyfaidx

    half = wlen // 2
    start0 = pos - 1 - half  # pyfaidx is 0-based
    end0 = start0 + wlen

    fasta = pyfaidx.Fasta(str(fasta_path), one_based_attributes=False)
    chrom_len = len(fasta[chrom])
    clip_start = max(start0, 0)
    clip_end = min(end0, chrom_len)
    seq = str(fasta[chrom][clip_start:clip_end]).upper() if clip_start < clip_end else ""

    encoded = np.zeros((wlen, 4), dtype=np.float32)
    offset = clip_start - start0
    for i, base in enumerate(seq):
        row = _BASE_TO_ROW.get(base)
        if row is not None:
            encoded[offset + i, row] = 1.0
    return encoded


def build_random_model(arch_path: str | Path):
    """Reconstruct the Keras architecture from its saved JSON, random-initialized."""
    from keras.models import model_from_json

    arch_json = Path(arch_path).read_text()
    return model_from_json(arch_json)


def load_checkpoint_into(model, weights_path: str | Path) -> None:
    """Load DeepCpG's `.h5` weights into an existing `model` in place."""
    model.load_weights(str(weights_path))


def dna_forward_probe(model, *, batch_size: int = 2, wlen: int = DNA_WINDOW_LEN):
    """A tiny sanity forward pass on random one-hot DNA windows (no FASTA needed)."""
    rng = np.random.default_rng(0)
    bases = rng.integers(0, 4, size=(batch_size, wlen))
    dummy = np.eye(4, dtype=np.float32)[bases]

    def probe(m):
        out = m.predict(dummy, verbose=0)
        # Multi-head models (one output per training cell) return a list of arrays;
        # stack into one array so `np.allclose` in verify.py works on either shape.
        return np.stack(out, axis=-1) if isinstance(out, list) else out

    return probe
