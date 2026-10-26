"""Secondary benchmark with approximately matched observed pair exposures.

The stopping rule is frozen before primary model selection/reference scoring.
This is not a matched-FLOPs experiment: pretrained feature extraction differs.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from train_alphagenome_mixed import OUT,STEPS,MixedData,positions,predictions,sha
from methylation_predictor.modeling.alphagenome import AlphaGenomePredictorFromCheckpoint
from validate_alphagenome_hypotheses import evaluate,t,event
from verify_alphagenome_capacity import CANON,dump
from compare_alphagenome_methylprophet import VIEWS,axes,paired_crossed


@torch.inference_mode()
def main():
    out=OUT;dest=out/'budget_matched';dest.mkdir(exist_ok=True)
    assert (out/'evaluation/complete.json').exists()
    protocol=json.loads((out/'fairness_supplement_protocol.json').read_text())
    recipe=json.loads((out/'frozen_recipe.json').read_text())['selected']
    available=[k for k in STEPS if k<=recipe['step']]
    candidates=[]
    for k in available:
        exposure=[]
        for seed in [17,18,19]:
            history=json.loads((out/f"official/{recipe['policy']}/seed{seed}/history.json").read_text())
            row=next(r for r in history if r['step']==k)
            exposure.append(sum(v['finite_pairs'] for v in row['exposure'].values()))
        candidates.append(dict(step=k,mean_finite_pairs=float(np.mean(exposure)),per_seed=exposure))
    reference_pairs=protocol['reference_configured_pair_exposures']
    selected=min(candidates,key=lambda r:(abs(r['mean_finite_pairs']-reference_pairs),r['step']))
    frozen=dict(selected=selected,candidates=candidates,reference_pair_exposures=reference_pairs,
                rule=protocol['rule'],no_test_metric_selection=True,code_sha256=sha(__file__))
    dump(dest/'frozen_budget.json',frozen)
    torch.set_num_threads(4);threadpool_limits(4);torch.backends.cuda.matmul.allow_tf32=False
    d=MixedData(out);d.configure('official')
    registry=pd.read_parquet(CANON/'cpg/registries/array_cpg_map.parquet').set_index('cpg_idx')
    records,contrasts=[],[]
    for seed in [17,18,19]:
        path=out/f"official/{recipe['policy']}/seed{seed}/step{selected['step']}.pt"
        adapter=AlphaGenomePredictorFromCheckpoint(path,device='cuda');model=adapter.model
        for label,(ss,cc) in zip(VIEWS,axes(out)):
            rr=positions(d.sources['array'].sample_ids,ss);cr=positions(d.sources['array'].cpg_ids,cc)
            y=d.sources['array'].y[t(rr)[:,None],t(cr)[None]]
            tissue=d.tissue[positions(d.sample_ids,ss)]
            p=t(predictions(model,d,ss,cc,calibrated=True))
            ref=t(np.load(out/f'reference/aligned/{label}.npy'))
            metric,per=evaluate(y,p,tissue)
            records.append(dict(view=label,seed=seed,step=selected['step'],**metric))
            np.savez_compressed(dest/f'{label}_seed{seed}_per_cpg.npz',cpg_idx=cc,**per)
            blocks=registry.loc[cc,'pos'].to_numpy()//5_000_000
            unique,inverse=np.unique(blocks,return_inverse=True);bi=t(inverse)
            mask=torch.isfinite(y)
            n=torch.zeros((len(ss),len(unique)),device='cuda',dtype=torch.float64)
            n.scatter_add_(1,bi[None].expand(len(ss),-1),mask.double())
            a=torch.zeros_like(n);b=torch.zeros_like(n)
            a.scatter_add_(1,bi[None].expand(len(ss),-1),torch.where(mask,p.double()-y.double(),0).square())
            b.scatter_add_(1,bi[None].expand(len(ss),-1),torch.where(mask,ref.double()-y.double(),0).square())
            contrasts.append(dict(view=label,seed=seed,step=selected['step'],**paired_crossed(a.cpu().numpy(),b.cpu().numpy(),n.cpu().numpy())))
            event(stage='budget_matched',view=label,seed=seed,step=selected['step'],metrics=metric)
        del model,adapter
    pd.DataFrame(records).to_csv(dest/'metrics.csv',index=False)
    pd.DataFrame(contrasts).to_csv(dest/'paired_contrasts.csv',index=False)
    dump(dest/'complete.json',dict(complete=True,frozen_budget=frozen,primary_recipe_unchanged=True))


if __name__=='__main__':
    main()
