import torch
import pytest
from methylation_predictor.optim import build_lr_scheduler


def test_constant_scheduler_stays_constant():
    p=torch.nn.Parameter(torch.tensor(1.0)); opt=torch.optim.AdamW([p],lr=1e-3); sched=build_lr_scheduler(opt,name="constant",total_steps=8)
    values=[]
    for _ in range(8): opt.step(); sched.step(); values.append(opt.param_groups[0]["lr"])
    assert values == pytest.approx([1e-3]*8)


def test_cosine_decays_to_floor():
    p=torch.nn.Parameter(torch.tensor(1.0)); opt=torch.optim.AdamW([p],lr=1.0); sched=build_lr_scheduler(opt,name="cosine",total_steps=10,min_lr_ratio=0.1)
    values=[]
    for _ in range(10): opt.step(); sched.step(); values.append(opt.param_groups[0]["lr"])
    assert values[-1] == pytest.approx(0.1)
    assert all(a >= b for a,b in zip(values,values[1:]))
