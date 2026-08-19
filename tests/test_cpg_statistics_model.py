import torch
from methylation_predictor.cpg_statistics.model import CpGStatisticsModelConfig, CpGStatisticsPredictor


def test_statistics_predictor_shapes_and_ranges():
    cfg=CpGStatisticsModelConfig(input_dim=12,hidden_dim=8,bottleneck_dim=4,ensemble_seeds=(1,2),sigma_floor=0.03)
    model=CpGStatisticsPredictor(cfg); out=model(torch.randn(7,12))
    assert out["mu"].shape == (7,)
    assert out["sigma"].shape == (7,)
    assert torch.all((out["mu"]>0)&(out["mu"]<1))
    assert torch.all(out["sigma"]>=0.03)
    assert len(model.mu_heads)==len(model.sigma_heads)==2
    assert model.mu_heads[0] is not model.sigma_heads[0]
