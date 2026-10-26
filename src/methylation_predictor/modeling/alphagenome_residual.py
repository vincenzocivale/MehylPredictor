"""Small genomic correction heads for controlled frozen-backbone experiments."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class AlphaGenomeResidualHead(nn.Module):
    """Correct a frozen intercept and RNA interaction; zero at initialization.

    Sequences use A/C/G/T/N = 0/1/2/3/4. Reverse-complement averaging is
    appropriate for an even window centred on the CpG dyad.
    """
    def __init__(self, mode: str, *, rank=32, ag_dim=1536):
        super().__init__()
        if mode not in {'ag', 'sequence', 'hybrid'}:
            raise ValueError(f'Unknown correction mode {mode!r}')
        self.mode, self.rank = mode, rank
        self.model_spec=dict(mode=mode,rank=rank,ag_dim=ag_dim)
        if mode in {'ag','hybrid'}:
            self.ag=nn.Sequential(nn.LayerNorm(ag_dim),nn.Linear(ag_dim,32),nn.GELU(),nn.Linear(32,32),nn.GELU())
        if mode in {'sequence','hybrid'}:
            self.sequence=nn.Sequential(nn.Conv1d(4,8,9,padding=4),nn.GELU(),nn.AvgPool1d(4),
                                        nn.Conv1d(8,16,5,padding=2),nn.GELU(),nn.AdaptiveAvgPool1d(4),
                                        nn.Flatten(),nn.Linear(64,32),nn.GELU())
        self.output=nn.Linear(64 if mode=='hybrid' else 32,rank+1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self, alphagenome, sequence=None):
        pieces=[]
        if self.mode in {'ag','hybrid'}:
            pieces.append(self.ag(alphagenome))
        if self.mode in {'sequence','hybrid'}:
            if sequence is None or sequence.ndim!=2 or sequence.dtype not in (torch.uint8,torch.int64,torch.int32):
                raise ValueError('sequence must be a two-dimensional integer A/C/G/T/N array')
            x=F.one_hot(sequence.long(),5)[...,:4].transpose(1,2).to(alphagenome.dtype)
            rc=x.flip((1,2))
            pieces.append((self.sequence(x)+self.sequence(rc))*.5)
        result=self.output(torch.cat(pieces,1))
        return result[:,0],result[:,1:]
