import numpy as np
import torch

from methylation_predictor.benchmark.methylprophet.feature_store import SortedIndex
from methylation_predictor.benchmark.methylprophet.probe import ScalarProbeNet


def test_sorted_index_preserves_query_order():
    idx = SortedIndex(np.array([50, 10, 30], dtype=np.int64))
    assert idx.positions_of(np.array([10, 50, 30])).tolist() == [1, 0, 2]
    assert idx.contains(np.array([10, 11, 30])).tolist() == [True, False, True]


def test_scalar_probe_net_shape():
    model = ScalarProbeNet(dim=8, dropout=0.0)
    x = torch.randn(5, 8)
    assert model(x).shape == (5,)
