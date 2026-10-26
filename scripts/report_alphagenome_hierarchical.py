"""Summarize hierarchical intercept cross-fitting without individual predictions."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/alphagenome_hierarchical_2026-10-18'
PREFIX=ROOT/'docs/experiments/alphagenome_hierarchical_2026-10-18'


def aggregate(source,novel,known):
    docs=[json.loads((OUT/'dev'/source/f'seed{s}'/'evaluation.json').read_text()) for s in [17,18,19]]
    return {view:{metric:float(np.mean([d[view][str(w)][metric] for d in docs])) for metric in docs[0][view][str(w)]}
            for view,w in [('novel',novel),('known',known)]}


def main():
    frozen=json.loads((OUT/'frozen_recipe.json').read_text())
    assert (OUT/'complete.json').exists()
    c=frozen['selected'] or frozen['best_balanced']
    dev={'v1':aggregate('v1_early',0.,0.),'v2':aggregate('v2_early',0.,0.),
         'candidate':aggregate(c['source'],c['novel_blend'],c['known_blend'])}
    bq,bk=dev['v1']['novel'],dev['v1']['known']; q,k=dev['candidate']['novel'],dev['candidate']['known']
    guard=dict(novel_mse=q['mse']<=bq['mse']*.995,known_mse=k['mse']<=bk['mse']*1.01,
               novel_mae=q['mae']<=bq['mae'],known_mae=k['mae']<=bk['mae']*1.01,
               novel_pcc=q['mas_pcc']>=bq['mas_pcc']-.002,known_pcc=k['mas_pcc']>=bk['mas_pcc']-.002,
               novel_within=q['within_tissue']>=bq['within_tissue']-.002,
               known_within=k['within_tissue']>=bk['within_tissue']-.002,
               severe=q['severe']<=bq['severe']+.001)
    frames=[]
    for seed in [17,18,19]:
        with np.load(OUT/'dev'/c['source']/f'seed{seed}'/f"novel_{c['novel_blend']:g}_per_cpg.npz") as z:
            f=pd.DataFrame({key:z[key] for key in z.files})
        for label,source in [('v1','v1_early'),('v2','v2_early')]:
            with np.load(OUT/'dev'/source/f'seed{seed}'/'novel_0_per_cpg.npz') as z:
                assert np.array_equal(z['cpg_idx'],f.cpg_idx)
                for metric in ['mse','mae','bias2','centered_mse','corr','severe']:
                    f[f'{label}_{metric}']=z[metric]
                    f[f'delta_{label}_{metric}']=f[metric]-z[metric]
        f['seed']=seed; frames.append(f)
    f=pd.concat(frames,ignore_index=True).groupby('cpg_idx').mean(numeric_only=True).drop(columns='seed').reset_index()
    with np.load(OUT/'features.npz') as z:
        ix=pd.Index(z['cpg_idx']).get_indexer(f.cpg_idx); assert (ix>=0).all()
        f['pos_1based']=z['position'][ix]
        f['context']=np.array(['island','shore','shelf','open_sea'])[z['dense'][ix,:4].argmax(1)]
    f['rmse']=np.sqrt(f.mse)
    f.to_csv(str(PREFIX)+'_errors.csv.gz',index=False)
    f.sort_values('mse',ascending=False).head(100).to_csv(str(PREFIX)+'_worst100.csv',index=False)
    selection=pd.read_csv(OUT/'selection.csv'); selection.to_csv(str(PREFIX)+'_selection.csv',index=False)
    audit=dict(cpgs=len(f),fraction_improved_v1=float((f.delta_v1_mse<0).mean()),
               fraction_improved_v2=float((f.delta_v2_mse<0).mean()),
               delta_mse=float(np.average(f.delta_v1_mse,weights=f.n)),
               delta_bias2=float(np.average(f.delta_v1_bias2,weights=f.n)),
               delta_centered_mse=float(np.average(f.delta_v1_centered_mse,weights=f.n)))
    data=dict(frozen=frozen,dev=dev,dev_guardrails=guard,dev_guardrails_passed=all(guard.values()),audit=audit)
    Path(str(PREFIX)+'.json').write_text(json.dumps(data,indent=2)+'\n')
    (OUT/'dev_confirmation.json').write_text(json.dumps(data,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    colors={'v1_early':'#404040','v2_early':'#6286a7','joint':'#d76b45','freeze_mean':'#9670b5','freeze_cpg':'#29967c'}
    base=frozen['baseline']
    for row in frozen['candidates']:
        x=100*(row['known']['mse']/base['known']['mse']-1)
        y=100*(row['novel']['mse']/base['novel']['mse']-1)
        axes[0].scatter(x,y,color=colors[row['source']],s=40 if row['admissible'] else 20,
                        marker='o' if row['admissible'] else 'x',alpha=.65)
    axes[0].axhline(0,color='grey',lw=.8); axes[0].axvline(0,color='grey',lw=.8)
    axes[0].set(xlabel='MSE note: variazione vs v1 (%)',ylabel='MSE nuove: variazione vs v1 (%)',title='A · Selezione sui fold interni')
    labels=['v1','v2','candidate']; x=np.arange(3); width=.35
    for off,view,color in [(-width/2,'novel','#477ca8'),(width/2,'known','#dc9c45')]:
        axes[1].bar(x+off,[dev[n][view]['mse'] for n in labels],width,label=view,color=color)
    axes[1].set(xticks=x,xticklabels=['v1','v2','Candidato'],ylabel='MSE beta',title='B · Conferma dev (3 seed)'); axes[1].legend()
    axes[2].scatter(f.v1_mse,f.mse,s=5,alpha=.3,color='#29967c')
    lim=max(f.v1_mse.max(),f.mse.max())*1.02; axes[2].plot([0,lim],[0,lim],color='grey',lw=1)
    axes[2].set(xlabel='MSE CpG · v1',ylabel='MSE CpG · candidato',title='C · CpG nuove nel dev')
    fig.savefig(str(PREFIX)+'.png',dpi=180); fig.savefig(str(PREFIX)+'.pdf'); plt.close(fig)
    lines=['# AlphaGenome: prior genomico e intercette specifiche delle CpG note','',
           f"Guardrail interni: **{frozen['guardrails_passed']}**. Conferma dev: **{all(guard.values())}**.", '',
           f"Candidato congelato prima della conferma dev: `{c['source']}`, prior nuove CpG {c['novel_blend']:g}, calibrazione empirica CpG note {c['known_blend']:g}.", '',
           '## Ipotesi e protocollo', '',
           'Il prior genomico già testato nella v2 migliorava la MSE delle CpG nuove ma sacrificava quelle note. Qui si separano esplicitamente le due fonti di informazione: il prior è usato sui loci mai osservati; per quelli training si può apprendere un’intercetta specifica senza richiedere target durante l’inferenza.', '',
           '- Nessun ulteriore fit degli encoder: cinque sorgenti congelate (v1/v2 a 1.000 step e tre continuazioni v2), tre seed, tre fold interni.',
           '- Nuove CpG: media beta del modello su 512 pazienti training combinata con prior Ridge AlphaGenome (alpha 100), pesi 0 / 0,25 / 0,5; offset risolto monotonamente.',
           '- CpG note: media beta sui soli valori osservati nei pazienti training combinata con la media predetta sugli stessi valori, pesi 0 / 0,5 / 1; unico offset per locus, risposta RNA mantenuta.',
           '- Il caso CpG nota viene definito dall’appartenenza agli ID di fit, non dall’errore sui pazienti esclusi. Per le nuove CpG non si legge alcuna metilazione né statistica osservata.',
           '- Protocollo separato congelato prima delle valutazioni di questa verifica; ipotesi motivata dai risultati precedenti, quindi sviluppo adattivo. Nessuna valutazione dell’holdout ufficiale.',
           '- Stessi guardrail stretti della ricerca precedente: nuove MSE almeno −0,5%, note MSE/MAE entro +1%, nuove MAE non peggiore, PCC/WT entro −0,002, errori >0,2 entro +0,001.', '',
           '## Migliori candidati interni', '',
           '| Sorgente | Prior nuove | Adattamento note | MSE nuove | MSE note | Ammissibile |',
           '|---|---:|---:|---:|---:|---|']
    ranked=sorted(frozen['candidates'],key=lambda r:(not r['admissible'],r['novel']['mse'],r['known']['mse']))[:15]
    for row in ranked:
        lines.append(f"| {row['source']} | {row['novel_blend']:g} | {row['known_blend']:g} | {row['novel']['mse']:.6f} | {row['known']['mse']:.6f} | {row['admissible']} |")
    lines+=['', '## Conferma dev', '',
            '| Modello | MSE nuove | MSE note | MAE nuove | PCC nuove | WT nuove | Errori >0,2 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for name in labels:
        qq,kk=dev[name]['novel'],dev[name]['known']
        lines.append(f"| {name} | {qq['mse']:.6f} | {kk['mse']:.6f} | {qq['mae']:.5f} | {qq['mas_pcc']:.5f} | {qq['within_tissue']:.5f} | {100*qq['severe']:.2f}% |")
    lines += ['', 'Guardrail dev: '+', '.join(f'{name}={value}' for name,value in guard.items())+'.', '',
              f"Audit su {audit['cpgs']:,} CpG nuove, errori mediati sui seed: migliora il {100*audit['fraction_improved_v1']:.1f}% rispetto a v1 e il {100*audit['fraction_improved_v2']:.1f}% rispetto a v2. ΔMSE vs v1 {audit['delta_mse']:+.6f}, di cui bias² {audit['delta_bias2']:+.6f} ed errore centrato {audit['delta_centered_mse']:+.6f}.", '',
              '## Artefatti e limiti', '',
              '[Figura](experiments/alphagenome_hierarchical_2026-10-18.png) · [Metriche](experiments/alphagenome_hierarchical_2026-10-18.json) · [Selezione](experiments/alphagenome_hierarchical_2026-10-18_selection.csv) · [Errori CpG](experiments/alphagenome_hierarchical_2026-10-18_errors.csv.gz) · [Peggiori 100](experiments/alphagenome_hierarchical_2026-10-18_worst100.csv)', '',
              'Array chr1; fold e dev già utilizzati nello sviluppo. Non costituisce conferma esterna né dimostra ottimalità. I riferimenti v1 sono i controlli interni a 1.000 step; non rappresentano il massimo risultato ottenibile con qualsiasi durata di fit. La calibrazione di un’intercetta mantiene i fattori ma, per la non linearità della sigmoid, può cambiare PCC e within-tissue. Le statistiche delle CpG note sono parametri stimati sui target training; non sono ammesse per CpG nuove.', '',
              'Riproduzione: `python scripts/search_alphagenome_hierarchical.py`; rendering con matplotlib: `python scripts/report_alphagenome_hierarchical.py`.', '']
    (ROOT/'docs/ALPHAGENOME_HIERARCHICAL_2026-10-18.md').write_text('\n'.join(lines))
    print(json.dumps(dict(dev=dev,dev_guardrails=guard,audit=audit)))


if __name__=='__main__': main()
