import numpy as np
from methylation_predictor.cpg_statistics.targets import SourceMoments, _combine_sample_weighted, _combine_technology_balanced


def _mom(count, beta_mean, logit_values):
    count=np.asarray(count,np.int64); beta_mean=np.asarray(beta_mean,float); vals=np.asarray(logit_values,float)
    return SourceMoments(count=count,beta_sum=count*beta_mean,logit_sum=count*vals,logit_sumsq=count*(vals**2+0.25))


def test_sample_weighted_and_technology_balanced_are_explicitly_different():
    a=_mom([100],[0.2],[0.0]); b=_mom([1],[0.8],[2.0]); sources={"array":a,"wgbs":b}
    mu_w,sig_w=_combine_sample_weighted(sources,epsilon=1e-4,sigma_floor=0.01)
    mu_b,sig_b=_combine_technology_balanced(sources,epsilon=1e-4,sigma_floor=0.01)
    assert mu_w[0] < 0.25
    assert np.isclose(mu_b[0],0.5)
    assert sig_w[0] > 0 and sig_b[0] > 0
