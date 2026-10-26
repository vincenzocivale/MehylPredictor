"""Refit a frozen, dev-confirmed hierarchical recipe and audit official errors.

Refuses to refit if internal or dev guardrails failed. Official measurements
are reported once after the recipe is frozen and never feed model selection.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from methylation_predictor.modeling.factory import build_alphagenome_predictor
from methylation_predictor.modeling.alphagenome import AlphaGenomePredictorFromCheckpoint
from methylation_predictor.modeling.alphagenome_hierarchical import AlphaGenomeHierarchicalPredictor
from verify_alphagenome_capacity import dump,normalize,CANON,PREP
import train_alphagenome_v2 as v2
import search_alphagenome_architecture as architecture
from search_alphagenome_hierarchical import fit_known_offsets,SEEDS

OUT=ROOT/'artifacts/alphagenome_hierarchical_2026-10-18'


@torch.inference_mode()
def predict(m,d,ss,cc):
    u=m.encode_rna(d.rna[torch.as_tensor(ss,device=d.device)]).float()
    result=np.empty((len(ss),len(cc)),np.float32)
    for j in range(0,len(cc),512):
        c=torch.as_tensor(cc[j:j+512],device=d.device); f=d.functional(c)
        f['cpg_ids']=torch.as_tensor(d.cpg_idx[cc[j:j+512]],device=d.device)
        b,a=m.encode_cpg(**f)
        for i in range(0,len(ss),256):
            p=(b[None]+u[i:i+256]@a.T/m.rank**.5).sigmoid()
            result[i:i+len(p),j:j+len(c)]=p.cpu().numpy()
    assert np.isfinite(result).all()
    return result


def audit(m,d,dest):
    protocol=json.loads((PREP/'table5_protocol/protocol.json').read_text())
    views={'new_patients':(d.val_s.cpu().numpy(),d.train_c.cpu().numpy(),'train_cpg_x_val_sample'),
           'new_cpgs':(d.train_s.cpu().numpy(),d.val_c.cpu().numpy(),'val_cpg_x_train_sample'),
           'double_ood':(d.val_s.cpu().numpy(),d.val_c.cpu().numpy(),'val_cpg_x_val_sample')}
    results={}
    for view,(ss,cc,countkey) in views.items():
        y=d.block(torch.as_tensor(ss,device=d.device),torch.as_tensor(cc,device=d.device)).cpu().numpy()
        p=predict(m,d,ss,cc)
        metric,frame=v2.error_frame(d,ss,cc,y,p)
        assert metric['rows']==protocol['finite_pair_audit']['evaluation_observed'][countkey]
        frame.to_csv(dest/f'{view}_per_cpg.csv.gz',index=False)
        frame.sort_values('mse',ascending=False).head(100).to_csv(dest/f'{view}_worst100.csv',index=False)
        result=dict(metrics=metric,audit=v2.summarize_errors(frame))
        for reference,folder in [('v1','baseline_v1'),('v2',f"seed{dest.name.removeprefix('seed')}")]:
            basepath=v2.ROOT/'artifacts/alphagenome_v2_2026-10-18/official'/folder/f'{view}_per_cpg.csv.gz'
            base=pd.read_csv(basepath)
            assert np.array_equal(base.cpg_idx,frame.cpg_idx) and np.array_equal(base.n,frame.n)
            comp=frame.copy()
            for key in ['mse','mae','bias','bias2','centered_mse','corr','severe']:
                comp[f'{reference}_{key}']=base[key].to_numpy()
                comp[f'delta_{key}']=comp[key]-base[key].to_numpy()
            comp.to_csv(dest/f'{view}_vs_{reference}.csv.gz',index=False)
            result[f'vs_{reference}']=dict(mse_difference=float(np.average(comp.delta_mse,weights=comp.n)),
                                           fraction_cpg_mse_improved=float((comp.delta_mse<0).mean()))
        results[view]=result
        print(json.dumps(dict(official=str(dest),view=view,metrics=metric)),flush=True)
    dump(dest/'metrics.json',dict(views=results,precision='strict FP32; common observations with v1/v2'))


def main():
    frozen=json.loads((OUT/'frozen_recipe.json').read_text())
    confirmed=json.loads((OUT/'dev_confirmation.json').read_text())
    assert frozen['guardrails_passed'] and confirmed['dev_guardrails_passed'],'No promotion: internal or dev guardrails failed'
    recipe=frozen['selected']; source=recipe['source']
    assert recipe==confirmed['frozen']['selected']
    protocol=dict(recipe=recipe,seeds=SEEDS,recipe_sha256=hashlib.sha256((OUT/'frozen_recipe.json').read_bytes()).hexdigest(),
                  dev_confirmation_sha256=hashlib.sha256((OUT/'dev_confirmation.json').read_bytes()).hexdigest(),
                  selection_frozen_before_official=True,
                  code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [Path(__file__),ROOT/'src/methylation_predictor/modeling/alphagenome_hierarchical.py',
                                ROOT/'src/methylation_predictor/modeling/alphagenome_transfer.py',
                                ROOT/'src/methylation_predictor/modeling/alphagenome.py',
                                ROOT/'src/methylation_predictor/modeling/factory.py']})
    if (OUT/'refit_protocol.json').exists(): assert json.loads((OUT/'refit_protocol.json').read_text())==protocol
    else: dump(OUT/'refit_protocol.json',protocol)
    torch.set_num_threads(4); threadpool_limits(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.set_float32_matmul_precision('highest')
    with h5py.File(CANON/'rna/tcga_rna_official_full.h5','r') as h:
        genes=[g.decode() if isinstance(g,bytes) else str(g) for g in h['gene_ids'][:]]
    d=v2.prepare_official(OUT)
    ts,tc=d.train_s.cpu().numpy(),d.train_c.cpu().numpy()
    norm=normalize(d,ts,tc)
    parent_root=OUT/'refit/phase1'; dest=parent_root/'official'; dest.mkdir(parents=True,exist_ok=True)
    prior=v2.fit_prior(d,tc,v2.training_means(d,ts,tc),dest)
    graph=v2.response_graph(d,tc,dest)
    np.savez_compressed(dest/'normalization.npz',**norm)
    arm='baseline' if source=='v1_early' else 'local01'
    recipe_hash=hashlib.sha256((OUT/'frozen_recipe.json').read_bytes()).hexdigest()
    for seed in SEEDS:
        final=OUT/'official'/f'seed{seed}'; final.mkdir(parents=True,exist_ok=True)
        path=final/'checkpoint.pt'
        if path.exists():
            adapter=AlphaGenomePredictorFromCheckpoint(path,device='cuda'); m=adapter.model
            st=torch.load(path,map_location='cpu',weights_only=False)
            assert st['recipe_sha256']==recipe_hash
        else:
            phase1=dest/arm/f'seed{seed}'
            base,_=v2.train(d,ts,tc,arm,seed,prior,graph,phase1,norm,genes,frozen)
            if source in architecture.MODES:
                del base; torch.cuda.empty_cache()
                architecture.PARENT=parent_root
                spec=('official',ts,d.val_s.cpu().numpy(),tc,d.val_c.cpu().numpy())
                base=architecture.continue_fit(d,spec,source,seed,OUT/'refit/continuation',norm,genes)
            spec=dict(base.model_spec); spec.pop('variant',None)
            spec['mean_blend']=recipe['novel_blend']
            m=AlphaGenomeHierarchicalPredictor(n_known=len(tc),**spec).cuda().eval()
            loaded=m.load_state_dict(base.state_dict(),strict=False)
            assert set(loaded.missing_keys)=={'known_cpg_idx','known_bias_delta','known_ready'} and not loaded.unexpected_keys
            del base
            train_u=m.encode_rna(d.rna[torch.as_tensor(ts,device=d.device)]).detach().float()
            delta=[]
            with torch.inference_mode():
                for chunk in torch.as_tensor(tc,device=d.device).split(512):
                    b,a=m.encode_cpg(**d.functional(chunk),calibrate=False)
                    offsets=fit_known_offsets(b,a,train_u,d.block(torch.as_tensor(ts,device=d.device),chunk),[recipe['known_blend']])
                    delta.append(offsets[recipe['known_blend']])
            m.set_known_offsets(torch.as_tensor(d.cpg_idx[tc],device=d.device),torch.cat(delta))
            state=dict(schema='alphagenome_hierarchical_chr1_v3',model_spec=m.model_spec,model_state=m.state_dict(),
                       normalization=norm,gene_ids=genes,fit_sample_ids=d.sample_idx[ts],fit_cpg_ids=d.cpg_idx[tc],
                       recipe=recipe,recipe_sha256=recipe_hash,seed=seed,coordinate_convention='hg38 1-based cytosine',
                       inference_cpg_ids='global registry identifiers; never feature-matrix row numbers',
                       code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                    [Path(__file__),ROOT/'src/methylation_predictor/modeling/alphagenome_hierarchical.py']})
            torch.save(state,path)
            del train_u
        adapter=AlphaGenomePredictorFromCheckpoint(path,device='cuda')
        for label,cc in [('known',tc[:13]),('novel',d.val_c[:13].cpu().numpy())]:
            ss=d.val_s[:7].cpu().numpy(); f=d.functional(torch.as_tensor(cc,device=d.device))
            global_ids=d.cpg_idx[cc]
            with torch.inference_mode():
                expected=m(d.rna[torch.as_tensor(ss,device=d.device)],**dict(f,cpg_ids=torch.as_tensor(global_ids,device=d.device)))['beta']
                actual=adapter(d.raw_rna[ss],genes,f['track_indices'],f['offsets'],d.raw_dense[cc],d.raw_ag[cc],cpg_ids=global_ids)
            diff=float((expected-actual).abs().max()); assert diff<1e-6
            dump(final/f'inference_{label}_check.json',dict(passed=True,max_abs_error=diff))
        if not (final/'metrics.json').exists(): audit(m,d,final)
        dump(final/'checkpoint_sha256.json',dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        del m,adapter; torch.cuda.empty_cache()
    dump(OUT/'refit_complete.json',dict(complete=True,recipe=recipe,recipe_sha256=recipe_hash,
                                      official_results_used_for_selection=False,primary_checkpoint=str(OUT/'official/seed17/checkpoint.pt')))


if __name__=='__main__':
    main()
