"""Genomic mean prior for unseen CpGs and fitted intercepts for observed CpGs.

The distinction is determined by training-locus identifiers. No query target
or query-population methylation statistic is needed at inference. This class
represents a fitted predictor; intercept estimation lives in the experiment
runner and uses training patients only.
"""
from __future__ import annotations
import torch
from .alphagenome_transfer import AlphaGenomeTwoStagePredictor


class AlphaGenomeHierarchicalPredictor(AlphaGenomeTwoStagePredictor):
    def __init__(self,input_dim,*,n_known=0,**kwargs):
        if n_known<0: raise ValueError('n_known must be nonnegative')
        super().__init__(input_dim,**kwargs)
        self.register_buffer('known_cpg_idx',torch.full((n_known,),-1,dtype=torch.long))
        self.register_buffer('known_bias_delta',torch.zeros(n_known))
        self.register_buffer('known_ready',torch.tensor(n_known==0))
        self.model_spec.update(variant='alphagenome_hierarchical_chr1_v3',n_known=n_known)

    @torch.no_grad()
    def set_known_offsets(self,cpg_ids,bias_delta):
        ids=torch.as_tensor(cpg_ids,device=self.known_cpg_idx.device)
        delta=torch.as_tensor(bias_delta,device=self.known_bias_delta.device,dtype=torch.float32)
        if ids.dtype not in (torch.int32,torch.int64) or ids.ndim!=1 or ids.shape!=self.known_cpg_idx.shape:
            raise ValueError('Training CpG identifiers must be an integer vector of length n_known')
        if delta.shape!=ids.shape or not torch.isfinite(delta).all():
            raise ValueError('Intercept offsets must be finite and match the training CpG identifiers')
        order=ids.argsort(); ids=ids[order]
        if (ids<0).any() or (ids[1:]<=ids[:-1]).any():
            raise ValueError('Training CpG identifiers must be unique and nonnegative')
        self.known_cpg_idx.copy_(ids)
        self.known_bias_delta.copy_(delta[order])
        self.known_ready.fill_(True)

    def encode_cpg(self,*,track_indices,offsets,dense,cpg_ids=None,calibrate=None):
        b,a=super().encode_cpg(track_indices=track_indices,offsets=offsets,dense=dense,
                              cpg_ids=cpg_ids,calibrate=False)
        use_calibration=not self.training if calibrate is None else bool(calibrate)
        if not use_calibration: return b,a
        if not bool(self.known_ready):
            raise RuntimeError('Fit the training-locus intercepts before inference')
        prior_bias=self.calibrated_intercept(b,a,dense[:,self.annotation_dim:])
        if not len(self.known_cpg_idx): return prior_bias,a
        if cpg_ids is None:
            raise ValueError('Global registry cpg_ids are required for hierarchical inference')
        ids=torch.as_tensor(cpg_ids,device=b.device)
        if ids.dtype not in (torch.int32,torch.int64) or ids.shape!=b.shape or (ids<0).any():
            raise ValueError('cpg_ids must be a nonnegative integer vector aligned with CpG features')
        at=torch.searchsorted(self.known_cpg_idx,ids)
        at=at.clamp_max(len(self.known_cpg_idx)-1)
        observed=self.known_cpg_idx[at]==ids
        bias=torch.where(observed,b+self.known_bias_delta[at],prior_bias)
        return bias,a
