"""Adapter around the vendored CpGPT repo (`external/CpGPT`).

CpGPT (Lima et al., biorxiv 2024.10.24.619766, github.com/lucascamillomd/CpGPT) is
site-agnostic: each CpG is represented by a DNA-sequence embedding (a frozen
Nucleotide-Transformer-v2 embedding of its +/-1000bp window, `d_dna_embedding=1024`
for both released pretraining sizes) plus its genomic position, not a fixed
per-model vocabulary -- so it can in principle score arbitrary chr1/chr123 loci,
unlike MethylGPT (see `methylgpt_adapter.py`).

This module only wraps the two released *pretrained masked* checkpoints
(`small` = CpGPT-2M, `large` = CpGPT-100M -- the README's other named checkpoints,
e.g. `age`/`cancer`/`mortality`, are task-finetuned, not masked-recovery models,
and are out of scope here). Import this module with `external/cpgpt-env`'s
python (see `scripts/benchmark_foundation_models/setup.sh`) -- it needs CpGPT's
own dependency stack (hydra, lightning, torchmetrics, ...), which is deliberately
not added to this repo's own requirements.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

VALID_VARIANTS = ("small", "large")


@dataclass
class CpGPTResources:
    variant: str
    config_path: Path
    checkpoint_path: Path
    vocab_path: Path


def resolve_resources(checkpoints_dir: str | Path, variant: str) -> CpGPTResources:
    if variant not in VALID_VARIANTS:
        raise ValueError(f"Unknown CpGPT variant '{variant}', expected one of {VALID_VARIANTS}")
    root = Path(checkpoints_dir)
    resources = CpGPTResources(
        variant=variant,
        config_path=root / "config" / f"{variant}.yaml",
        checkpoint_path=root / "weights" / f"{variant}.ckpt",
        vocab_path=root / "vocab" / f"{variant}.json",
    )
    missing = [p for p in (resources.config_path, resources.checkpoint_path, resources.vocab_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing CpGPT '{variant}' resource file(s): {missing}")
    return resources


def build_random_model(config_path: str | Path):
    """Instantiate the CpGPT architecture from its hydra config, random-initialized."""
    import hydra
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    model = hydra.utils.instantiate(config.model)
    model.eval()
    return model, config


def load_checkpoint_into(model, checkpoint_path: str | Path, *, strict: bool = True, device: str = "cpu") -> None:
    """Load a CpGPT `.ckpt` into `model` in place -- mirrors `CpGPTInferencer.load_cpgpt_model`."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]
    cleaned = {k.replace("net._orig_mod.", "net."): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=strict)


def dna_embedding_forward_probe(model, config, *, batch_size: int = 2, seq_len: int = 4):
    """A tiny, architecture-level sanity forward pass: `model.net.encode_sequence`.

    Deliberately does not build the full genomic-locations preprocessing pipeline
    (dna_embeddings lookup by position, methylation-value masking, LightningModule
    batch dict) -- that belongs to the eventual full-split `evaluate.py`. This only
    proves the loaded network is wired up and produces checkpoint-dependent output,
    using the exact `d_dna_embedding` the config declares (1024, NTv2-500m, for both
    released sizes) rather than a made-up dimension.
    """
    d_dna_embedding = config.model.net.d_dna_embedding
    torch.manual_seed(0)
    dummy = torch.randn(batch_size, seq_len, d_dna_embedding)

    def probe(m):
        with torch.no_grad():
            return m.net.encode_sequence(dummy)

    return probe
