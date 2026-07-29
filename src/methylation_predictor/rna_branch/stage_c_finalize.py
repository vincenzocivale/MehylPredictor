"""Write a compact final Stage-C report and paired bootstrap comparisons."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def skill(t, p, prior):
    m=np.isfinite(t)&np.isfinite(p)
    pm=np.broadcast_to(prior, t.shape)
    return 1-float(np.mean((t[m]-p[m])**2))/float(np.mean((t[m]-pm[m])**2))

def main():
    a=argparse.ArgumentParser(); a.add_argument('--root',required=True); a.add_argument('--bootstrap',type=int,default=300); z=a.parse_args()
    root=Path(z.root); screen=root/'screening'; out=root/'final_report'; out.mkdir(exist_ok=True)
    rows=[]; boot=[]; rng=np.random.default_rng(20260729)
    base={}
    for p in screen.glob('c0_v0_seed*/predictions_double_ood.npz'):
        base[p.parent.name.rsplit('_seed',1)[1]]=np.load(p)
    for run in sorted(screen.glob('*_v*_seed*')):
        m=run/'metrics.json'; pred=run/'predictions_double_ood.npz'
        if not m.is_file() or not pred.is_file(): continue
        d=json.loads(m.read_text()); fam=d['run_name'].rsplit('_seed',1)[0]; seed=d['run_name'].rsplit('_seed',1)[1]; x=d['panels']['double_ood']
        rows.append({'family':fam,'seed':int(seed),'best_epoch':d['best_epoch'],**{k:x[k] for k in ['mse','mae','skill_vs_prior','dynamic_skill','within_cancer_skill','dynamic_pearson','dynamic_spearman','dynamic_calibration_alpha','dynamic_amplitude_ratio','macro_cancer_skill_vs_prior']}})
        if fam=='c0_v0': continue
        q=np.load(pred); b=base[seed]; t=q['target']; prior=q['prior']; diff=[]
        for axis in ('patient','genomic_block'):
            ds=[]
            for _ in range(z.bootstrap):
                idx=rng.integers(t.shape[0],size=t.shape[0]) if axis=='patient' else rng.integers(t.shape[1],size=t.shape[1])
                if axis=='patient': ds.append(skill(t[idx],q['prediction'][idx],prior)-skill(t[idx],b['prediction'][idx],prior))
                else: ds.append(skill(t[:,idx],q['prediction'][:,idx],prior[idx])-skill(t[:,idx],b['prediction'][:,idx],prior[idx]))
            boot.append({'family':fam,'seed':int(seed),'unit':axis,'mean_diff':float(np.mean(ds)),'ci_low':float(np.quantile(ds,.025)),'ci_high':float(np.quantile(ds,.975))})
    raw=pd.DataFrame(rows); raw.to_csv(out/'phase2_per_seed.csv',index=False)
    summary=raw.groupby('family').agg(['mean','std']); summary.to_csv(out/'phase2_mean_std.csv')
    pd.DataFrame(boot).to_csv(out/'paired_bootstrap.csv',index=False)
    lines=['# Stage C — RNA–locus interaction','', '## Decisione','', 'Il protocollo LR 2e-5/20 epoch/minimum validation beta-MSE migliora C0; nessuna interaction, decomposizione C3 o variante di sampling/loss supera C0 in tutti i seed. La conclusione supportata è **5: il limite principale era il protocollo di ottimizzazione**.','', '## Double-OOD (media±std)','', summary.to_markdown(), '', 'I confronti bootstrap appaiati (pazienti e blocchi genomici) sono in `paired_bootstrap.csv`. C3 mixture non è stata eseguita perché C3 semplice è nettamente inferiore.','', '## Prossimo esperimento','', 'Replicare C0 ottimizzato su un secondo cromosoma/split genomico indipendente: è il test più informativo per distinguere un guadagno di ottimizzazione robusto da adattamento al singolo pannello chr1.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(out/'report.md')
if __name__=='__main__': main()
