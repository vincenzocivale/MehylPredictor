from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from methylation_predictor.rna_training.locus_cls_trainer import (
    _retired_locus_compat,
)


ROOT = Path(__file__).resolve().parents[1]


def _raw(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def test_main_recipe_retired_controls_resolve_to_noop_compat_values():
    compat = _retired_locus_compat(
        _raw(ROOT / "configs/models/main.yaml")
    )
    assert compat == {
        "use_fusion_product": False,
        "use_raw_product": False,
        "product_mlp": False,
        "include_raw_rna": True,
        "include_raw_cpg": True,
        "fusion_init_std": 0.01,
        "query_source": "ntv3",
        "residual_aux_weight": 0.0,
        "raw_lr_multiplier": 1.0,
        "trunk_hidden_dim": 256,
        "bottleneck_dim": 64,
        "trunk_dropout": 0.1,
    }


def test_functional_base_recipe_is_compatible():
    compat = _retired_locus_compat(
        _raw(
            ROOT
            / "configs/models/functional_fusion/base.yaml"
        )
    )
    assert compat["use_fusion_product"] is False
    assert compat["use_raw_product"] is False
    assert compat["raw_lr_multiplier"] == 1.0


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("query_source", "hybrid_joint"),
        ("raw_lr_multiplier", 2.0),
        ("residual_aux_weight", 0.1),
        ("use_fusion_product", True),
    ],
)
def test_retired_nondefault_controls_fail_loudly(key, value):
    raw = _raw(ROOT / "configs/models/main.yaml")
    raw.setdefault("locus_cls", {})[key] = value
    with pytest.raises(ValueError, match="retired locus_cls field"):
        _retired_locus_compat(raw)
