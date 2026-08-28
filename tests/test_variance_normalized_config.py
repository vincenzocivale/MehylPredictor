import tempfile
from pathlib import Path

import pytest
import yaml

from methylation_predictor.benchmark.methylprophet.config import load_config


def test_reference_config_preserves_variance_normalized_flag():
    cfg=load_config("configs/benchmark_methylprophet/reference.yaml")
    assert cfg.model.variance_normalized_residual is True


def _write_config(overrides: dict) -> Path:
    base = yaml.safe_load(Path("configs/benchmark_methylprophet/reference.yaml").read_text())
    base.setdefault("model", {}).update(overrides)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(base, tmp)
    tmp.flush()
    return Path(tmp.name)


def test_use_prior_anchor_defaults_true():
    cfg = load_config("configs/benchmark_methylprophet/reference.yaml")
    assert cfg.model.use_prior_anchor is True


def test_no_prior_anchor_rejects_variance_normalized_residual():
    path = _write_config({"use_prior_anchor": False, "variance_normalized_residual": True, "zero_init_residual": False})
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_config(str(path))


def test_no_prior_anchor_requires_zero_init_residual_false():
    path = _write_config({"use_prior_anchor": False, "variance_normalized_residual": False, "zero_init_residual": True})
    with pytest.raises(ValueError, match="zero_init_residual=false"):
        load_config(str(path))
