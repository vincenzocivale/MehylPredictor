from methylation_predictor.benchmark.methylprophet.config import load_config


def test_reference_config_preserves_variance_normalized_flag():
    cfg=load_config("configs/benchmark_methylprophet/reference.yaml")
    assert cfg.model.variance_normalized_residual is True
