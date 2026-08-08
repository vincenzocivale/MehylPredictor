"""Config loading for the canonical bundle root and per-protocol sampling knobs.

No `/raid/...` path is hardcoded in this module -- the default lives only in
`configs/data/tcga_canonical.yaml`, itself overridable by environment
variable or an explicit argument.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ENV_ROOT_VAR = "TCGA_CANONICAL_ROOT"
DEFAULT_BUNDLE_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "data" / "tcga_canonical.yaml"


def resolve_bundle_root(root: str | Path | None = None, config_path: str | Path | None = None) -> Path:
    """Resolve the canonical bundle root: explicit arg > env var > YAML config."""
    if root is not None:
        return Path(root)
    env_value = os.environ.get(ENV_ROOT_VAR)
    if env_value:
        return Path(env_value)
    config_path = Path(config_path) if config_path is not None else DEFAULT_BUNDLE_CONFIG
    raw = yaml.safe_load(config_path.read_text())
    return Path(raw["root"])


@dataclass
class ProtocolRunConfig:
    protocol: str
    source_sampling_policy: str
    source_weights: dict[str, float]
    batch_sample_size: int
    batch_cpg_size: int


def load_protocol_run_config(path: str | Path) -> ProtocolRunConfig:
    """Load a `configs/protocols/*.yaml` run config.

    Expects a `source_sampling: {policy: ..., weights: {...}}` block --
    `policy` names one of `Protocol.SOURCE_SAMPLING_POLICIES`
    (`explicit_balanced`, `proportional_to_measurements`); `weights` is only
    meaningful for `explicit_balanced` and defaults to equal representation
    across the protocol's configured sources when omitted. See
    docs/data/METHYLPROPHET_PROTOCOLS.md's "two-level comparison policy" for
    why this is a named, swappable ablation axis rather than a claimed
    reproduction of MethylProphet's own internal mixing ratio.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    batch = raw.get("batch", {})
    source_sampling = raw.get("source_sampling", {})
    return ProtocolRunConfig(
        protocol=raw["protocol"],
        source_sampling_policy=source_sampling.get("policy", "explicit_balanced"),
        source_weights=dict(source_sampling.get("weights", {})),
        batch_sample_size=int(batch.get("sample_size", 32)),
        batch_cpg_size=int(batch.get("cpg_size", 256)),
    )
