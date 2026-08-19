"""Named source-sampling policies (explicit_balanced / proportional_to_measurements)
and the YAML configs that drive them -- see docs/data/METHYLPROPHET_PROTOCOLS.md's
"two-level comparison policy".

The weight-override fixtures here are synthetic (written to tmp_path), not the
historical source-mixing-ablation configs -- those were closed experiments and
have been removed; this file only needs to exercise the loader/policy contract
itself.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from methylation_predictor.tcga_canonical import SOURCE_SAMPLING_POLICIES, load_protocol, load_protocol_run_config

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs" / "protocols"


def _write_run_config(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(dedent(body))
    return path


def test_no_policy_is_named_methylprophet_exact():
    assert "methylprophet_exact" not in SOURCE_SAMPLING_POLICIES
    assert set(SOURCE_SAMPLING_POLICIES) == {"explicit_balanced", "proportional_to_measurements"}


def test_reference_config_parses_with_valid_policy():
    config = load_protocol_run_config(CONFIGS_DIR / "tcga_mix_chr1.yaml")
    assert config.protocol == "tcga_mix_chr1"
    assert config.source_sampling_policy in SOURCE_SAMPLING_POLICIES


@pytest.mark.parametrize(
    ("policy", "weights_block"),
    [
        ("explicit_balanced", "weights: {array: 1.0, epic: 1.0, wgbs: 1.0}"),
        ("explicit_balanced", "weights: {array: 4.0, epic: 1.0, wgbs: 1.0}"),
        ("proportional_to_measurements", ""),
    ],
)
def test_synthetic_configs_parse_with_valid_policy(tmp_path, policy, weights_block):
    path = _write_run_config(
        tmp_path,
        "run_config.yaml",
        f"""
        protocol: tcga_mix_chr1
        source_sampling:
          policy: {policy}
          {weights_block}
        """,
    )
    config = load_protocol_run_config(path)
    assert config.protocol == "tcga_mix_chr1"
    assert config.source_sampling_policy in SOURCE_SAMPLING_POLICIES


def test_explicit_balanced_default_is_equal_weight_regardless_of_pool_size(bundle):
    protocol = load_protocol("tcga_mix_chr1", bundle=bundle)
    pools = protocol._source_pools("explicit_balanced", weights=None)
    weights = {pool.name: pool.weight for pool in pools}
    assert weights == {"array": 1.0, "epic": 1.0, "wgbs": 1.0}
    # pool sizes are wildly different (8,260 vs 1,706 vs 32 rows) -- weights must not be
    assert len({len(pool.row_positions) for pool in pools}) > 1


def test_array_heavy_config_overrides_weights(tmp_path, bundle):
    protocol = load_protocol("tcga_mix_chr1", bundle=bundle)
    path = _write_run_config(
        tmp_path,
        "array_heavy.yaml",
        """
        protocol: tcga_mix_chr1
        source_sampling:
          policy: explicit_balanced
          weights: {array: 4.0, epic: 1.0, wgbs: 1.0}
        """,
    )
    config = load_protocol_run_config(path)
    pools = protocol._source_pools(config.source_sampling_policy, config.source_weights)
    weights = {pool.name: pool.weight for pool in pools}
    assert weights["array"] > weights["epic"] == weights["wgbs"]


def test_proportional_to_measurements_weights_by_pool_row_count(bundle):
    protocol = load_protocol("tcga_mix_chr1", bundle=bundle)
    pools = protocol._source_pools("proportional_to_measurements", weights=None)
    weights = {pool.name: pool.weight for pool in pools}
    sizes = {pool.name: len(pool.row_positions) for pool in pools}
    assert weights == {name: float(size) for name, size in sizes.items()}
    # array (8,260 train measurements) must dominate wgbs (32)
    assert weights["array"] > weights["wgbs"]


def test_proportional_to_measurements_rejects_explicit_weights(bundle):
    protocol = load_protocol("tcga_mix_chr1", bundle=bundle)
    with pytest.raises(ValueError):
        protocol._source_pools("proportional_to_measurements", weights={"array": 1.0})


def test_unknown_policy_rejected(bundle):
    protocol = load_protocol("tcga_mix_chr1", bundle=bundle)
    with pytest.raises(ValueError):
        protocol._source_pools("methylprophet_exact", weights=None)
