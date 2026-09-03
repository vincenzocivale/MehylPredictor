"""Adapter around the vendored MethylGPT repo (`external/MethylGPT`).

MethylGPT (Ying et al., biorxiv 2024.10.30.621013, github.com/albert-ying/MethylGPT)
has a **fixed vocabulary of 49,156 Illumina-probe-ID CpG sites** (confirmed by
inspecting the downloaded `vocab.json` for all three released sizes: 49,159 tokens
total = 49,156 sites + `<pad>`/`<cls>`/`<eoc>`; 49,135 of the 49,156 are standard
`cgXXXXXXXX` probe IDs, the rest are Illumina `ch.`-prefixed probes). Only the
subset of our chr1/chr123 val-CpGs whose hg38 position falls in that vocabulary can
be scored -- `vocabulary_overlap()` in `verify.py` computes this, but needs an
external probe-id -> hg38-position crosswalk this repo does not build yet (our own
canonical bundle keys CpGs by (chrom, pos), not Illumina probe ID -- see
`docs/PAPER_EXPERIMENTS.md`).

Import this module with `external/methylgpt-env`'s python (pinned torch==2.3.1 /
torchtext==0.18.0 -- MethylGPT depends on `torchtext`, which has no build compatible
with this machine's main torch 2.13 install; see `scripts/benchmark_foundation_models/
setup.sh` for why this needed its own isolated venv, separate from CpGPT's).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch

VALID_VARIANTS = ("base", "medium", "large")
# README "Pretrained Models" table -- cross-checked against each downloaded
# args.json's own layer_size (2026-09-02: all three matched on download).
EXPECTED_LAYER_SIZE = {"base": 64, "medium": 128, "large": 256}


@dataclass
class MethylGPTResources:
    variant: str
    args_path: Path
    checkpoint_path: Path
    vocab_path: Path


def resolve_resources(checkpoints_dir: str | Path, variant: str) -> MethylGPTResources:
    if variant not in VALID_VARIANTS:
        raise ValueError(f"Unknown MethylGPT variant '{variant}', expected one of {VALID_VARIANTS}")
    root = Path(checkpoints_dir) / variant
    args_path = root / "args.json"
    vocab_path = root / "vocab.json"
    checkpoints = list(root.glob("*.pt"))
    if not args_path.is_file() or not vocab_path.is_file() or not checkpoints:
        raise FileNotFoundError(f"Missing MethylGPT '{variant}' resource file(s) under {root}")
    return MethylGPTResources(variant=variant, args_path=args_path, checkpoint_path=checkpoints[0], vocab_path=vocab_path)


def check_variant_identity(resources: MethylGPTResources) -> dict:
    """Cross-check args.json's layer_size against the README table for this variant name.

    MethylGPT's Google-Drive-hosted checkpoints have internal filenames that don't
    match their folder/variant name (e.g. the `large` folder's file is named
    `medium-best_model_epoch6.pt`, `medium`'s is `small-best_model_epoch6.pt`) --
    this is leftover training-run naming, not a mislabeled download, but it's worth
    verifying explicitly rather than trusting the folder name blindly.
    """
    args = json.loads(resources.args_path.read_text())
    expected = EXPECTED_LAYER_SIZE[resources.variant]
    actual = args.get("layer_size")
    return {
        "variant": resources.variant,
        "checkpoint_filename": resources.checkpoint_path.name,
        "expected_layer_size": expected,
        "actual_layer_size": actual,
        "matches": actual == expected,
    }


def load_vocab_from_json(vocab_path: str | Path, pad_token: str = "<pad>"):
    """Reconstruct a `torchtext` vocab from a saved `vocab.json` (stoi dict).

    `methylgpt.model.methyl_vocab.MethylVocab` always rebuilds from a probe-id CSV,
    which we don't have; the downloaded `vocab.json` already *is* the built
    vocabulary (`MethylVocab._save_vocab`'s `get_stoi()` dump), so this reconstructs
    the same `torchtext.vocab.Vocab` object directly from it instead.
    """
    from torchtext._torchtext import Vocab as VocabPybind
    from torchtext.vocab import Vocab

    stoi = json.loads(Path(vocab_path).read_text())
    tokens = [tok for tok, _idx in sorted(stoi.items(), key=lambda kv: kv[1])]
    vocab = Vocab(VocabPybind(tokens, None))
    vocab.set_default_index(vocab[pad_token])
    vocab.pad_token = pad_token
    return vocab


def build_model(resources: MethylGPTResources, *, load_model: bool = False):
    """Instantiate `MethylGPTModel` from `args.json` + `vocab.json`.

    `load_model=False` returns a randomly initialized model (for the
    before/after fingerprint in `verify.py`); `load_model=True` loads the
    checkpoint via the model's own `from_pretrained` (exercising its exact
    production loading path, non-strict fallback included).
    """
    from methylgpt.model.methyl_model import MethylGPTModel

    args = json.loads(resources.args_path.read_text())
    vocab = load_vocab_from_json(resources.vocab_path)
    config = dict(args)
    config["load_model"] = load_model
    config["pretrained_file"] = str(resources.checkpoint_path)
    model = MethylGPTModel.from_pretrained(config, vocab)
    model.eval()
    return model, vocab, config


def _remap_flash_attn_qkv_keys(state_dict: dict) -> dict:
    """Rename `self_attn.Wqkv.{weight,bias}` -> `self_attn.in_proj_{weight,bias}`.

    The released checkpoints were trained with `fast_transformer=True` (this repo's
    `FlashTransformerEncoderLayer`, `methylgpt/modules/scGPT/scgpt/model/model.py`),
    whose fused QKV linear layer is named `Wqkv`. Without flash-attn installed, the
    model falls back to plain `nn.TransformerEncoderLayer`/`nn.MultiheadAttention`,
    whose equivalent fused QKV projection is named `in_proj_weight`/`in_proj_bias`
    -- confirmed by direct comparison (2026-09-02) to have the **identical** shape
    `(3*d_model, d_model)`/`(3*d_model,)` for all three released sizes, so this is a
    pure key rename, not an architecture change requiring flash-attn.

    Correctness relies on both implementations concatenating Q/K/V as three
    contiguous `d_model`-sized row blocks in that order -- the standard convention
    both `torch.nn.MultiheadAttention` and flash-attn's `MHA`/`FlashSelfAttention`
    follow. This has not been cross-checked numerically against an actual
    flash-attn forward pass on this checkpoint (no flash-attn install on this CPU
    dev machine) -- do that once flash-attn is available on a GPU host, before
    trusting exact reproduction of the paper's own reported numbers.
    """
    remapped = {}
    for key, value in state_dict.items():
        if key.endswith(".self_attn.Wqkv.weight"):
            key = key.replace(".self_attn.Wqkv.weight", ".self_attn.in_proj_weight")
        elif key.endswith(".self_attn.Wqkv.bias"):
            key = key.replace(".self_attn.Wqkv.bias", ".self_attn.in_proj_bias")
        remapped[key] = value
    return remapped


def load_checkpoint_into(model, checkpoint_path: str | Path) -> dict:
    """Load a MethylGPT `.pt` state dict into an existing `model` in place.

    Replicates `MethylGPTModel.from_pretrained`'s try-strict/fall-back-to-compatible-
    subset logic (so this exercises the same production behavior), but applied
    in-place to `model` rather than via that classmethod's "always builds a new
    instance" shape -- `verify.py`'s before/after fingerprint needs the same object.
    Applies `_remap_flash_attn_qkv_keys` first so a flash-attn-free environment
    doesn't spuriously drop the attention QKV weights (see that function's docstring).
    Returns `{"mode": "strict"|"partial", "n_skipped": int, "skipped": [...]}`.
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = _remap_flash_attn_qkv_keys(state_dict)
    try:
        model.load_state_dict(state_dict)
        return {"mode": "strict", "n_skipped": 0, "skipped": []}
    except RuntimeError:
        model_dict = model.state_dict()
        compatible = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        skipped = sorted(set(state_dict.keys()) - set(compatible.keys()))
        model_dict.update(compatible)
        model.load_state_dict(model_dict)
        return {"mode": "partial", "n_skipped": len(skipped), "skipped": skipped}


def embedding_forward_probe(model, vocab, *, batch_size: int = 2, seq_len: int = 8):
    """A tiny sanity forward pass through `get_cell_embeddings` on random real tokens."""
    torch.manual_seed(0)
    non_special_ids = [i for tok, i in vocab.get_stoi().items() if not tok.startswith("<")]
    gene_ids = torch.tensor(
        [non_special_ids[:seq_len] for _ in range(batch_size)], dtype=torch.long
    )
    values = torch.rand(batch_size, seq_len)

    def probe(m):
        with torch.no_grad():
            return m.get_cell_embeddings(gene_ids, values)

    return probe
