"""Named source-sampling policies (explicit_balanced / proportional_to_measurements)
and the YAML configs that drive them -- see docs/data/METHYLPROPHET_PROTOCOLS.md's
"two-level comparison policy"."""
from __future__ import annotations

from pathlib import Path

import pytest

from methylation_predictor.tcga_canonical import SOURCE_SAMPLING_POLICIES, load_protocol, load_protocol_run_config

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs" / "protocols"


def test_no_policy_is_named_methylprophet_exact():
    assert "methylprophet_exact" not in SOURCE_SAMPLING_POLICIES
    assert set(SOURCE_SAMPLING_POLICIES) == {"explicit_balanced", "proportional_to_measurements"}


@pytest.mark.parametrize(
    "config_name",
    [
        "tcga_mix_chr1.yaml",
        "ablations/tcga_mix_chr1_equal_source.yaml",
        "ablations/tcga_mix_chr1_array_heavy.yaml",
        "ablations/tcga_mix_chr1_proportional_to_measurements.yaml",
    ],
)
def test_ablation_configs_parse_with_valid_policy(config_name):
    config = load_protocol_run_config(CONFIGS_DIR / config_name)
    assert config.protocol == "tcga_mix_chr1"
    assert config.source_sampling_policy in SOURCE_SAMPLING_POLICIES


def test_explicit_balanced_default_is_equal_weight_regardless_of_pool_size(bundle):
    protocol = load_protocol("tcga_mix_chr1", bundle=bundle)
    pools = protocol._source_pools("explicit_balanced", weights=None)
    weights = {pool.name: pool.weight for pool in pools}
    assert weights == {"array": 1.0, "epic": 1.0, "wgbs": 1.0}
    # pool sizes are wildly different (8,260 vs 1,706 vs 32 rows) -- weights must not be
    assert len({len(pool.row_positions) for pool in pools}) > 1


def test_array_heavy_config_overrides_weights(bundle):
    protocol = load_protocol("tcga_mix_chr1", bundle=bundle)
    config = load_protocol_run_config(CONFIGS_DIR / "ablations" / "tcga_mix_chr1_array_heavy.yaml")
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
