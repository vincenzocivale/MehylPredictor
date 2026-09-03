"""Checkpoint-loaded-not-randomly-initialized verification.

`torch.load_state_dict(..., strict=True)` (the convention used everywhere else in
this repo, see `rna_training/evaluator.py`) only catches key/shape mismatches. It
does not catch a checkpoint that shape-matches but is the wrong file, or a loader
that silently falls back to a partial/no-op load (both MethylGPT's own
`from_pretrained` and CpGPT's `load_cpgpt_model` have such non-strict fallback
paths in their upstream code). This module adds an independent, framework-agnostic
check: fingerprint the model before and after loading, and assert every tensor and
the forward-pass output actually changed.

Only depends on `torch` for the torch-specific functions (`fingerprint_state_dict`,
`verify_checkpoint_loaded`) and `numpy` for everything else -- `torch` is imported
lazily inside those two functions (not at module level) so this module also imports
cleanly in `external/deepcpg-env`, which has no torch (legacy `tensorflow==1.13.1`/
`keras==1.2.2` stack instead; see `verify_checkpoint_loaded_keras`). No
CpGPT/MethylGPT/DeepCpG imports here either way, so this module works from any of
the three isolated per-model environments without needing this repo installed as a
package (scripts add `src/` to `sys.path`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Hash a file's bytes (checkpoint provenance, matches `run_store.sha256_file`)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_state_dict(state_dict: dict[str, "torch.Tensor"]) -> dict[str, tuple[float, float, int]]:
    """Per-tensor (L2 norm, mean, numel) fingerprint of a state dict.

    Cheap and order-independent -- good enough to detect "nothing changed" or
    "only some tensors changed" after a load, without hashing full tensor bytes.
    """
    out: dict[str, tuple[float, float, int]] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().float()
        out[name] = (t.norm().item(), t.mean().item(), t.numel())
    return out


@dataclass
class VerificationResult:
    passed: bool
    checkpoint_path: str
    checkpoint_sha256: str
    n_params_total: int
    n_params_changed: int
    n_params_unchanged: int
    unchanged_param_names: list[str] = field(default_factory=list)
    forward_output_changed: bool | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "n_params_total": self.n_params_total,
            "n_params_changed": self.n_params_changed,
            "n_params_unchanged": self.n_params_unchanged,
            "unchanged_param_names": self.unchanged_param_names,
            "forward_output_changed": self.forward_output_changed,
            "notes": self.notes,
        }


def verify_checkpoint_loaded(
    model: "torch.nn.Module",
    checkpoint_path: str | Path,
    load_fn,
    *,
    forward_probe=None,
    tol: float = 1e-6,
) -> VerificationResult:
    """Fingerprint `model` before/after `load_fn(model)`, assert weights actually moved.

    Args:
        model: a freshly, randomly constructed instance of the target architecture.
        checkpoint_path: the checkpoint file being loaded (hashed for provenance only;
            `load_fn` is responsible for actually reading it).
        load_fn: callable `load_fn(model) -> None` that loads `checkpoint_path`'s
            weights into `model` in place (e.g. via the model's own
            `load_state_dict`/`from_pretrained`, so this exercises the exact
            production loading code path, non-strict fallbacks included).
        forward_probe: optional callable `forward_probe(model) -> torch.Tensor`,
            called once before and once after loading on the same fixed input; if
            given, the result also asserts the output actually changed (a second,
            independent signal beyond parameter fingerprints -- catches loaders
            whose non-strict fallback silently loads zero tensors).
        tol: minimum L2-norm delta to count a tensor as "changed".

    Returns:
        VerificationResult. `passed` is False if any tensor is unchanged, or if
        `forward_probe` is given and its output does not change.
    """
    import torch  # lazy: keeps this module importable in torch-free envs (deepcpg-env)

    checkpoint_path = Path(checkpoint_path)
    before = fingerprint_state_dict(model.state_dict())
    before_out = forward_probe(model) if forward_probe is not None else None

    load_fn(model)

    after = fingerprint_state_dict(model.state_dict())
    after_out = forward_probe(model) if forward_probe is not None else None

    unchanged = []
    for name, (norm_before, mean_before, _n) in before.items():
        norm_after, mean_after, _n2 = after.get(name, (None, None, None))
        if norm_after is None:
            continue
        if abs(norm_after - norm_before) < tol and abs(mean_after - mean_before) < tol:
            unchanged.append(name)

    forward_changed = None
    notes: list[str] = []
    if before_out is not None:
        forward_changed = not torch.allclose(before_out, after_out, atol=tol)
        if not forward_changed:
            notes.append("forward-pass output identical before/after load_fn")

    passed = len(unchanged) == 0 and (forward_changed is not False)
    if unchanged:
        notes.append(
            f"{len(unchanged)}/{len(before)} tensors unchanged after load "
            "(likely a silent no-op/partial load)"
        )

    return VerificationResult(
        passed=passed,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=sha256_file(checkpoint_path) if checkpoint_path.is_file() else "",
        n_params_total=len(before),
        n_params_changed=len(before) - len(unchanged),
        n_params_unchanged=len(unchanged),
        unchanged_param_names=unchanged,
        forward_output_changed=forward_changed,
        notes=notes,
    )


def fingerprint_keras_model(model) -> dict[str, tuple[float, float, int]]:
    """Per-weight-array (L2 norm, mean, size) fingerprint of a Keras model (DeepCpG).

    Keras 1.x has no single `state_dict()`-like accessor with stable names, so this
    walks `model.layers` and names each layer's weight arrays `<layer_name>.<index>`
    -- same shape/semantics as `fingerprint_state_dict`, just keyed differently, so
    `verify_checkpoint_loaded_keras` below can reuse the same before/after-diff logic.
    """
    out: dict[str, tuple[float, float, int]] = {}
    for layer in model.layers:
        for i, array in enumerate(layer.get_weights()):
            arr = np.asarray(array, dtype=np.float64)
            out[f"{layer.name}.{i}"] = (float(np.linalg.norm(arr)), float(arr.mean()), int(arr.size))
    return out


def verify_checkpoint_loaded_keras(
    model,
    checkpoint_path: str | Path,
    load_fn,
    *,
    forward_probe=None,
    tol: float = 1e-6,
) -> VerificationResult:
    """Keras/TensorFlow counterpart of `verify_checkpoint_loaded` (for DeepCpG).

    Same before/after fingerprint + optional forward-probe-diff contract; only the
    weight extraction (`fingerprint_keras_model`) and the forward-probe comparison
    (`numpy.allclose` instead of `torch.allclose`) differ, since DeepCpG's Keras 1.x
    model has no `state_dict()`/torch-tensor API.
    """
    checkpoint_path = Path(checkpoint_path)
    before = fingerprint_keras_model(model)
    before_out = forward_probe(model) if forward_probe is not None else None

    load_fn(model)

    after = fingerprint_keras_model(model)
    after_out = forward_probe(model) if forward_probe is not None else None

    unchanged = []
    for name, (norm_before, mean_before, _n) in before.items():
        norm_after, mean_after, _n2 = after.get(name, (None, None, None))
        if norm_after is None:
            continue
        if abs(norm_after - norm_before) < tol and abs(mean_after - mean_before) < tol:
            unchanged.append(name)

    forward_changed = None
    notes: list[str] = []
    if before_out is not None:
        forward_changed = not np.allclose(before_out, after_out, atol=tol)
        if not forward_changed:
            notes.append("forward-pass output identical before/after load_fn")

    passed = len(unchanged) == 0 and (forward_changed is not False)
    if unchanged:
        notes.append(
            f"{len(unchanged)}/{len(before)} weight arrays unchanged after load "
            "(likely a silent no-op/partial load)"
        )

    return VerificationResult(
        passed=passed,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=sha256_file(checkpoint_path) if checkpoint_path.is_file() else "",
        n_params_total=len(before),
        n_params_changed=len(before) - len(unchanged),
        n_params_unchanged=len(unchanged),
        unchanged_param_names=unchanged,
        forward_output_changed=forward_changed,
        notes=notes,
    )


def vocabulary_overlap(
    query_positions: list[tuple[str, int]],
    probe_to_position: dict[str, tuple[str, int]],
) -> dict:
    """Overlap between a set of (chrom, pos) loci and a probe-id -> (chrom, pos) vocabulary.

    Used for MethylGPT, whose pretrained vocabulary is a fixed set of ~49,156 Illumina
    probe IDs -- only the subset of our chr1/chr123 val-CpGs whose hg38 position is in
    that vocabulary can be scored at all. `probe_to_position` is an external
    probe-id -> hg38-position crosswalk (e.g. an Illumina manifest); this repo's own
    canonical bundle only stores (chrom, pos), not Illumina probe IDs, so that crosswalk
    is not built yet -- see `docs/PAPER_EXPERIMENTS.md`'s foundation-model section.

    Returns `{"n_query", "n_vocab", "n_overlap", "coverage_fraction", "overlap_probe_ids"}`.
    """
    position_to_probe: dict[tuple[str, int], str] = {
        pos: probe for probe, pos in probe_to_position.items()
    }
    overlap_probe_ids = [
        position_to_probe[pos] for pos in query_positions if pos in position_to_probe
    ]
    n_query = len(query_positions)
    return {
        "n_query": n_query,
        "n_vocab": len(probe_to_position),
        "n_overlap": len(overlap_probe_ids),
        "coverage_fraction": (len(overlap_probe_ids) / n_query) if n_query else 0.0,
        "overlap_probe_ids": overlap_probe_ids,
    }
