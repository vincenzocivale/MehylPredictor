from __future__ import annotations

import numpy as np
import torch

from methylation_predictor.models import GenePathwayEncoder


def membership(path):
    np.savez_compressed(path, n_genes=np.asarray([6], dtype=np.int64), n_pathways=np.asarray([3], dtype=np.int64), gene_idx=np.asarray([0,1,1,2,3,4], dtype=np.int64), pathway_idx=np.asarray([0,0,1,1,2,2], dtype=np.int64), pathway_names=np.asarray(["p0","p1","p2"]), matched_genes_per_pathway=np.asarray([2,2,2], dtype=np.int64))


def test_gene_pathway_forward_and_grad(tmp_path):
    p=tmp_path/"m.npz"; membership(p)
    model=GenePathwayEncoder(6,5,str(p),dim1=2,dim2=3,dropout=0.0,layer_norm=False)
    x=torch.randn(4,6,requires_grad=True); out=model(x).global_vector
    assert out.shape==(4,5); assert torch.isfinite(out).all(); out.square().mean().backward()
    assert model.edge_weight.grad is not None; assert torch.isfinite(model.edge_weight.grad).all()


def test_disconnected_gene_has_no_effect_without_layernorm(tmp_path):
    p=tmp_path/"m.npz"; membership(p); torch.manual_seed(3)
    model=GenePathwayEncoder(6,4,str(p),dim1=2,dim2=3,dropout=0.0,layer_norm=False).eval()
    x1=torch.randn(2,6); x2=x1.clone(); x2[:,5]+=1000
    with torch.no_grad(): y1=model(x1).global_vector; y2=model(x2).global_vector
    assert torch.allclose(y1,y2,atol=1e-6)
