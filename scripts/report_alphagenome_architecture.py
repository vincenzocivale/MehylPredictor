"""Render aggregate-only evidence for the two-stage AlphaGenome search."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
NAMES={'v1_early':'v1 · 1.000 step','v2_early':'v2 · 1.000 step','joint':'v2 · continuazione congiunta',
       'freeze_mean':'v2 · intercetta congelata','freeze_cpg':'v2 · ramo CpG congelato'}


def label(c):
    text=NAMES[c['mode']]
    if c['blend']: text+=f" + vicini {c['space']} {c['blend']:g}"
    return text


def summary(out,mode,key):
    docs=[json.loads((out/'dev'/mode/f'seed{s}'/'evaluation.json').read_text())[key] for s in [17,18,19]]
    return {view:{metric:float(np.mean([d[view][metric] for d in docs])) for metric in docs[0][view]}
            for view in docs[0]}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'artifacts/alphagenome_architecture_2026-10-18')
    args=ap.parse_args(); out=args.out
    complete=json.loads((out/'complete.json').read_text())
    frozen=json.loads((out/'frozen_recipe.json').read_text())
    c=frozen['selected'] or frozen['best_balanced']; key=f"{c['space']}_{c['blend']:g}"
    dev={'v1':summary(out,'v1_early','none_0'),'v2':summary(out,'v2_early','none_0'),
         'candidate':summary(out,c['mode'],key)}
    frames=[]
    for seed in [17,18,19]:
        path=out/'dev'/c['mode']/f'seed{seed}'/f'{key}_double_ood_per_cpg.npz'
        with np.load(path) as z: frame=pd.DataFrame({k:z[k] for k in z.files})
        for name,mode in [('v1','v1_early'),('v2','v2_early')]:
            with np.load(out/'dev'/mode/f'seed{seed}'/'none_0_double_ood_per_cpg.npz') as z:
                assert np.array_equal(z['cpg_idx'],frame.cpg_idx)
                for metric in ['mse','bias2','centered_mse','mae','corr','severe']:
                    frame[f'{name}_{metric}']=z[metric]
                    frame[f'delta_{name}_{metric}']=frame[metric]-z[metric]
        frame['seed']=seed
        frames.append(frame)
    all_errors=pd.concat(frames,ignore_index=True)
    grouped=all_errors.groupby('cpg_idx').mean(numeric_only=True).drop(columns='seed').reset_index()
    with np.load(out/'features.npz') as z:
        ids=z['cpg_idx']; ix=pd.Index(ids).get_indexer(grouped.cpg_idx); assert (ix>=0).all()
        grouped['pos_1based']=z['position'][ix]
        grouped['context']=np.array(['island','shore','shelf','open_sea'])[z['dense'][ix,:4].argmax(1)]
    grouped['rmse']=np.sqrt(grouped.mse)
    artifact=ROOT/'docs/experiments/alphagenome_architecture_2026-10-18'
    grouped.to_csv(str(artifact)+'_errors.csv.gz',index=False)
    grouped.sort_values('mse',ascending=False).head(100).to_csv(str(artifact)+'_worst100.csv',index=False)
    selection=pd.read_csv(out/'selection.csv')
    selection.to_csv(str(artifact)+'_selection.csv',index=False)
    retrieval=[]
    for mode in ['v2_early','joint','freeze_mean','freeze_cpg']:
        for space in ['raw','learned','random']:
            delta=[]
            for k in range(3):
                for seed in [17,18,19]:
                    d=json.loads((out/f'fold{k}'/mode/f'seed{seed}'/'evaluation.json').read_text())
                    delta.append(d[f'{space}_0.5']['double_ood']['mse']-d['none_0']['double_ood']['mse'])
            retrieval.append(dict(mode=mode,space=space,delta_mse=float(np.mean(delta))))
    audit=dict(cpgs=len(grouped),fraction_improved_vs_v1=float((grouped.delta_v1_mse<0).mean()),
               fraction_improved_vs_v2=float((grouped.delta_v2_mse<0).mean()),
               delta_mse_vs_v1=float(np.average(grouped.delta_v1_mse,weights=grouped.n)),
               delta_bias2_vs_v1=float(np.average(grouped.delta_v1_bias2,weights=grouped.n)),
               delta_centered_mse_vs_v1=float(np.average(grouped.delta_v1_centered_mse,weights=grouped.n)))
    data=dict(frozen=frozen,dev=dev,retrieval=retrieval,audit=audit,complete=complete)
    Path(str(artifact)+'.json').write_text(json.dumps(data,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    colors={'v1_early':'#303030','v2_early':'#6286a7','joint':'#d76b45','freeze_mean':'#9670b5','freeze_cpg':'#29967c'}
    b=frozen['baseline']; q=b['double_ood']; k=b['known_loci']
    for row in frozen['candidates']:
        x=100*(row['known_loci']['mse']/k['mse']-1); y=100*(row['double_ood']['mse']/q['mse']-1)
        axes[0].scatter(x,y,color=colors[row['mode']],marker='o' if row['blend']==0 else '^',
                        alpha=1 if row['blend']==0 else .45,s=65 if row['blend']==0 else 30)
        if row['blend']==0: axes[0].annotate(row['mode'],(x,y),xytext=(4,4),textcoords='offset points',fontsize=8)
    axes[0].axvline(0,color='grey',lw=.8); axes[0].axhline(0,color='grey',lw=.8)
    axes[0].set(xlabel='MSE CpG note: variazione vs v1 (%)',ylabel='MSE CpG nuove: variazione vs v1 (%)',
                title='A · Compromesso nei fold interni')
    labels=['v1','v2','candidate']; x=np.arange(3); width=.35
    for offset,view,color in [(-width/2,'double_ood','#477ca8'),(width/2,'known_loci','#dc9c45')]:
        axes[1].bar(x+offset,[dev[n][view]['mse'] for n in labels],width,label=view,color=color)
    axes[1].set(xticks=x,xticklabels=['v1','v2','Candidato'],ylabel='MSE beta',title='B · Conferma dev (3 seed)')
    axes[1].legend(fontsize=8)
    axes[2].scatter(grouped.v1_mse,grouped.mse,s=5,alpha=.3,color='#29967c')
    lim=max(grouped.v1_mse.max(),grouped.mse.max())*1.02
    axes[2].plot([0,lim],[0,lim],color='grey',lw=1)
    axes[2].set(xlabel='MSE per CpG · v1',ylabel='MSE per CpG · candidato',title='C · Errori dev mediati sui seed')
    fig.savefig(str(artifact)+'.png',dpi=180); fig.savefig(str(artifact)+'.pdf'); plt.close(fig)
    lines=['# AlphaGenome: ricerca architetturale in due fasi e trasferimento locale','',
           'Esperimenti interni del 18 ottobre 2026. Non è una dimostrazione di ottimalità architetturale.', '',
           f"Guardrail interni superati: **{frozen['guardrails_passed']}**. Candidato esaminato sul dev: **{label(c)}**.", '',
           '## Disegno', '',
           '- Tre fold genomici interni × tre seed; stesso split e normalizzazione dei precedenti esperimenti.',
           '- Tre continuazioni da ciascun checkpoint EMA v2 a 1.000 step: congiunta, intercetta congelata, intero ramo CpG congelato. Altri 1.000 step, AdamW riavviato in tutti i bracci. Non equivale a un fit ininterrotto da 2.000 step.',
           '- Congelare la media significa congelare tutta la funzione genomica che produce l’intercetta. La media beta sulla popolazione può ancora cambiare attraverso la risposta RNA e la sigmoid.',
           '- Trasferimento: media dei fattori di cinque vicini training, blend 0 / 0,25 / 0,5; geometria AlphaGenome originale o fattori appresi. Standardizzazione/PCA sui soli loci training; distanza ≥1 Mb; appaiamento contesto/regione/quartile TSS, fallback globale se necessario.',
           '- L’intercetta viene ricalibrata per preservare la media beta originale su 512 pazienti training di riferimento. Nessun target della CpG query è usato nel recupero o nella calibrazione.',
           '- Controllo negativo: cinque vicini casuali con gli stessi vincoli, blend 0,5; escluso dalla selezione.',
           '- Nuovi pazienti valutati su tutte le CpG nuove e su 4.096 CpG note prefissate per fold. Guardrail: MSE nuova almeno −0,5%; MSE/MAE note entro +1%; MAE nuova non peggiore; PCC/within-tissue entro −0,002; errori >0,2 entro +0,001 rispetto a v1.',
           '- Se nessun candidato passa, nessuna promozione: si riporta separatamente il minimo rapporto MSE medio fra le due viste come compromesso esplorativo.', '',
           '## Selezione interna', '',
           '| Modello | MSE nuove | MSE note | PCC nuove | WT nuove | Ammissibile |',
           '|---|---:|---:|---:|---:|---|']
    for row in frozen['candidates']:
        qq,kk=row['double_ood'],row['known_loci']
        lines.append(f"| {label(row)} | {qq['mse']:.6f} | {kk['mse']:.6f} | {qq['mas_pcc']:.5f} | {qq['within_tissue']:.5f} | {row['admissible']} |")
    lines += ['', '## Conferma dev a scelta congelata','',
              '| Modello | MSE nuove | MSE note | MAE nuove | PCC nuove | WT nuove | Errori >0,2 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name in labels:
        qq,kk=dev[name]['double_ood'],dev[name]['known_loci']
        lines.append(f"| {name} | {qq['mse']:.6f} | {kk['mse']:.6f} | {qq['mae']:.5f} | {qq['mas_pcc']:.5f} | {qq['within_tissue']:.5f} | {100*qq['severe']:.2f}% |")
    lines += ['', '## Controllo del trasferimento', '',
              'Differenza di MSE sulle nuove CpG rispetto allo stesso checkpoint senza trasferimento, media dei nove confronti interni. Un segno negativo indica un miglioramento.', '',
              '| Continuazione | Spazio dei vicini | ΔMSE con blend 0,5 |','|---|---|---:|']
    for r in retrieval: lines.append(f"| {r['mode']} | {r['space']} | {r['delta_mse']:+.6f} |")
    lines += ['', '## Errori per CpG', '',
              f"Audit dev su {audit['cpgs']:,} nuove CpG, mediato sui tre seed. Migliora rispetto alla v1 il {100*audit['fraction_improved_vs_v1']:.1f}% dei loci; rispetto alla v2 il {100*audit['fraction_improved_vs_v2']:.1f}%.",
              f"ΔMSE vs v1 = {audit['delta_mse_vs_v1']:+.6f}: bias² {audit['delta_bias2_vs_v1']:+.6f}, errore centrato {audit['delta_centered_mse_vs_v1']:+.6f}.", '',
              '## Artefatti e limiti', '',
              '[Figura](experiments/alphagenome_architecture_2026-10-18.png) · [Metriche](experiments/alphagenome_architecture_2026-10-18.json) · [Selezione CSV](experiments/alphagenome_architecture_2026-10-18_selection.csv) · [Errori CpG](experiments/alphagenome_architecture_2026-10-18_errors.csv.gz) · [Peggiori 100](experiments/alphagenome_architecture_2026-10-18_worst100.csv)', '',
              'Array chr1; split interni e dev già usati nello sviluppo precedente, quindi nessuna nuova conferma esterna. Nessun target dell’holdout ufficiale è stato selezionato o valutato. Embargo 300 kb: non garantisce contesti AlphaGenome completamente disgiunti. Le medie sui seed non sono un ensemble. Il recupero esplicito richiede un banco di fattori training e una ricerca dei vicini: è un costo aggiuntivo rispetto all’encoder CpG cacheabile.', '',
              'I checkpoint di continuazione sono portabili tramite AlphaGenomePredictorFromCheckpoint. Gli esperimenti di trasferimento sono implementati nella funzione transfer_response e nel valutatore; il checkpoint base da solo non incorpora il recupero dei vicini.', '',
              'Riproduzione: `python scripts/search_alphagenome_architecture.py`; rendering: `MPLCONFIGDIR=/tmp/methylpredictor-matplotlib python scripts/report_alphagenome_architecture.py` con matplotlib disponibile.', '']
    (ROOT/'docs/ALPHAGENOME_ARCHITECTURE_SEARCH_2026-10-18.md').write_text('\n'.join(lines))
    (out/'report_complete.json').write_text(json.dumps(dict(complete=True,audit=audit),indent=2)+'\n')
    print(json.dumps(dict(report_complete=True,dev=dev,audit=audit)))


if __name__=='__main__': main()
