"""Publication-facing report of the source-matched MethylProphet benchmark."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/alphagenome_mixed_2026-10-18'
STEM = ROOT/'docs/experiments/alphagenome_mixed_2026-10-18'
DOC = ROOT/'docs/ALPHAGENOME_MIXED_METHYLPROPHET_2026-10-18.md'


def table(f, columns):
    lines = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for _, row in f.iterrows():
        vals=[]
        for col in columns:
            x=row[col]
            vals.append('—' if pd.isna(x) else f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x))
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=OUT)
    out=ap.parse_args().out
    assert (out/'comparison/complete.json').exists()
    assert (out/'budget_matched/complete.json').exists()
    recipe=json.loads((out/'frozen_recipe.json').read_text())
    audit=json.loads((out/'reference/aligned/complete.json').read_text())
    frame=pd.read_csv(out/'comparison/metrics.csv')
    raw=pd.read_csv(out/'evaluation/metrics.csv')
    for ix,row in raw.iterrows():
        per=np.load(out/f"evaluation/{row['view']}_{row['model']}_seed{int(row['seed'])}_per_cpg.npz")
        raw.loc[ix,'mas_pcc']=float(np.nanmedian(per['corr']))
        per.close()
    raw_rows=raw[raw.model=='raw'].copy();raw_rows['model']='AlphaGenome_mixed_raw'
    frame=pd.concat([frame,raw_rows],ignore_index=True)
    means=frame.groupby(['view','model'],as_index=False)[['mse','mae','mas_pcc','within_tissue','severe']].mean()
    paper=frame[frame.model.isin(['MethylProphet_released','AlphaGenome_mixed_v3'])].groupby(['view','model'],as_index=False)[['cpg_pcc_mean','mas_pcc','sample_pcc_mean','sample_pcc_median']].mean()
    paper.to_csv(str(STEM)+'_paper_correlations.csv',index=False)
    contrasts=pd.read_csv(out/'comparison/paired_contrasts.csv')
    primary=contrasts[contrasts.seed=='mean_loss_seeds']
    strata=pd.read_csv(out/'comparison/training_exposure_strata.csv').groupby(['view','scope'],as_index=False)[['loci','our_mse','reference_mse','our_mae','reference_mae','our_pcc','reference_pcc']].mean()
    strata['delta_mse_percent']=100*(strata.our_mse/strata.reference_mse-1)
    means.to_csv(str(STEM)+'_metrics.csv',index=False)
    contrasts.to_csv(str(STEM)+'_paired_contrasts.csv',index=False)
    strata.to_csv(str(STEM)+'_training_exposure_strata.csv',index=False)
    raw.to_csv(str(STEM)+'_raw_and_calibrated.csv',index=False)
    selected=pd.DataFrame([dict(policy=r['policy'],step=r['step'],known_mse=r['known']['mse'],
        novel_array_mse=r['novel_array']['mse'],known_mae=r['known']['mae'],novel_array_mae=r['novel_array']['mae'],
        score=r['balanced_mse_ratio']) for r in recipe['candidates']])
    selected.to_csv(str(STEM)+'_selection.csv',index=False)
    errors=[]
    for p in (out/'comparison').glob('*_paired_cpg_errors.csv.gz'):
        target=Path(str(STEM)+'_'+p.name);shutil.copy2(p,target)
        view=p.name.removesuffix('_paired_cpg_errors.csv.gz')
        f=pd.read_csv(p);f['view']=view;errors.append(f)
    error=pd.concat(errors)
    manifest=ROOT/'artifacts/alphagenome_hypotheses_2026-10-18/external/HM450.hg38.manifest.tsv.gz'
    if manifest.exists():
        annotation=pd.read_csv(manifest,sep='\t',usecols=['probeID','CpG_chrm','CpG_beg','gene_HGNC'])
        annotation['pos']=annotation.CpG_beg+1
        annotation=annotation[(annotation.CpG_chrm=='chr1') & annotation.pos.isin(error.pos.unique())]
        def join(values):
            return ';'.join(sorted(set(values.dropna().astype(str))))
        annotation=annotation.groupby('pos',as_index=False).agg(illumina_probe_ids=('probeID',join),gene_HGNC=('gene_HGNC',join))
        error=error.merge(annotation,on='pos',how='left',validate='many_to_one')
    error['chr']='chr1'
    worst=pd.concat([f.nlargest(50,'our_mse') for _,f in error.groupby('view')])
    worst['our_rmse']=np.sqrt(worst.our_mse)
    worst.to_csv(str(STEM)+'_worst_cpgs.csv',index=False)
    budget_meta=json.loads((out/'budget_matched/frozen_budget.json').read_text())
    budget=pd.read_csv(out/'budget_matched/metrics.csv').groupby('view',as_index=False)[['mse','mae','mas_pcc']].mean()
    baseline=means[means.model=='MethylProphet_released'][['view','mse']].rename(columns={'mse':'reference_mse'})
    budget=budget.merge(baseline,on='view',validate='one_to_one')
    budget['delta_mse_percent']=100*(budget.mse/budget.reference_mse-1)
    budget.to_csv(str(STEM)+'_matched_exposure.csv',index=False)
    cpg_summary=error.groupby('view').agg(loci=('cpg_idx','size'),fraction_improved=('delta_mse',lambda x:float((x<0).mean())),
        median_delta_mse=('delta_mse','median')).reset_index()
    cpg_summary.to_csv(str(STEM)+'_cpg_improvement.csv',index=False)
    facts=dict(frozen_recipe=recipe,audit=audit,metrics=means.to_dict('records'),paired_contrasts=contrasts.to_dict('records'),strata=strata.to_dict('records'),
        paper_correlations=paper.to_dict('records'),budget_rule=budget_meta,budget_metrics=budget.to_dict('records'),cpg_improvement=cpg_summary.to_dict('records'))
    Path(str(STEM)+'.json').write_text(json.dumps(facts,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    views=['train_cpg-val_sample','val_cpg-train_sample','val_cpg-val_sample']
    x=np.arange(3)
    for ax,key,title in zip(axes[0],['mse','mas_pcc'],['MSE (lower is better)','Median within-CpG correlation (higher is better)']):
        for i,(model,label) in enumerate([('MethylProphet_released','MethylProphet released'),('AlphaGenome_mixed_v3','AlphaGenome mixed, mean of 3 seeds')]):
            y=means[means.model==model].set_index('view').loc[views,key]
            ax.bar(x+(i-.5)*.35,y,width=.35,label=label)
        ax.set_xticks(x,['Known CpGs\nnew patients','New Array CpGs\ntraining patients','New Array CpGs\nnew patients']);ax.set_title(title)
    axes[0,0].legend(fontsize=8)
    ax=axes[1,0]
    p=primary.set_index('view').loc[views]
    val=p.mse_delta_percent.to_numpy();low=p.ci95_percent_low.to_numpy();high=p.ci95_percent_high.to_numpy()
    ax.errorbar(x,val,yerr=np.stack([val-low,high-val]),fmt='o',capsize=5)
    ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(x,['Known/new','New/train','New/new'])
    ax.set_title('Paired patient / 5Mb block bootstrap');ax.set_ylabel('MSE change vs MethylProphet (%)')
    ax=axes[1,1]
    for policy in selected.policy.unique():
        rows=selected[selected.policy==policy].sort_values('step')
        ax.plot(rows.step,rows.score,marker='o',label=policy)
    ax.set_title('Selection using internal data only');ax.set_xlabel('Training steps');ax.set_ylabel('Balanced relative MSE');ax.legend()
    fig.savefig(str(STEM)+'.png',dpi=180);fig.savefig(str(STEM)+'.pdf');plt.close(fig)
    r=recipe['selected']
    text=f'''# AlphaGenome Array+EPIC+WGBS vs MethylProphet — 18 ottobre 2026

Confronto del nostro modello riaddestrato da zero sui tre dataset con le predizioni del modello pubblicato, appaiate per paziente, CpG e target. La ricetta è stata congelata sullo sviluppo interno prima di calcolare le metriche ufficiali o quelle di MethylProphet.

## Correzione dello split prima del confronto

L’audit sulle predizioni rilasciate ha rilevato che il precedente protocollo derivato `methylprophet_table5_tcga_chr1/table5_protocol` ricostruiva un diverso split di pazienti: 723 pazienti validation differenti, nonostante gli stessi conteggi 8.260/918. Gli assi canonici `protocols/tcga_mix_chr1` coincidono invece con il rilascio. Il primo fit esplorativo è stato interrotto dopo 500 aggiornamenti, archiviato in `rejected_pre_alignment` ed escluso dalla selezione. Anche prior, grafo e normalizzazione sono stati ricostruiti. I vecchi risultati del progetto conservano il loro significato sui propri split, ma non sono confronti appaiati con questo modello pubblicato.

## Dati e rappresentazione

Array: 8.260 pazienti × 33.885 CpG training; 918 pazienti e 6.742 CpG esclusi dall’Array training. Tutti i 1.706 profili EPIC e le 32 misure WGBS di 31 pazienti sono usati nel training, con 71.748 e 1.999.446 CpG rispettivamente. Il filtro nativo sulle finestre di sequenza esclude 102 loci WGBS contenenti N. Le misure replicate WGBS sono preservate nel training per rispettare il dataset; non diventano pazienti indipendenti negli intervalli di valutazione Array.

Training finito: **275.082.367 Array + 115.856.100 EPIC + 63.982.272 WGBS = 454.920.739 coppie**. Nessun paziente Array validation compare nelle fonti ausiliarie. Tutti i 25.017 geni canonici sono disponibili; il nostro modello usa RNA standardizzata sui soli pazienti training, MethylProphet usa la propria quantizzazione nativa. Il numero di coppie disponibili non equivale al numero di coppie campionate durante l’ottimizzazione.

Cache AlphaGenome uniforme: `../data/alphagenome_cpg_embeddings/chr1.h5`, `emb_1bp`, 1.536 dimensioni, contesto 1.048.576 bp. Copertura completa degli assi di training; ID e coordinate confrontati con i tre registri canonici, zero discrepanze. Nessuna fusione con i vecchi embedding a 524.288 bp. Il modello è riaddestrato da zero. Le informazioni pretrained di AlphaGenome e le annotazioni funzionali sono parte del metodo: stesso dataset di metilazione non significa stesso pretraining o stesso costo totale.

## Selezione e architettura

Testa della media superficiale, otto FFN CpG, rango 32, un encoder RNA e regolarizzazione locale di peso 0,01. Per scalare a due milioni di loci, i cinque vicini distanti almeno 1 Mb sono cercati in un banco fisso di al massimo 65.536 loci training, con tutte le CpG Array training e un campione delle altre, dopo PCA64 training-only e matching delle annotazioni. Il grafo è quindi un’approssimazione esplicita della precedente ricerca su tutti i loci; gli assi di supervisione restano completi.

Due politiche di campionamento delle fonti: uguale peso oppure proporzionale alle coppie finite disponibili. Il rapporto originale di costruzione offline degli shard non è stato ricostruito; non si dichiara identità di distribuzione dei minibatch. Seed17 per la selezione, tre refit indipendenti seed17/18/19. Pazienti interni generati soltanto entro il vero training pubblicato, con vecchi blocchi genomici interni; scelta per media dei rapporti MSE delle viste CpG note e nuove rispetto all’Array. **Scelta: {r['policy']}, {r['step']} step**.

Calibrazione v3: prior Ridge100 sulle medie delle sole CpG Array training, blend 0,25 per gli altri loci; offset empirici esclusivamente sulle CpG Array training usando i pazienti Array training. Nessun target Array escluso dal training entra nella calibrazione. I checkpoint finali includono gli offset e sono ricaricabili dall’adapter pubblico.

Checkpoint principale: `artifacts/alphagenome_mixed_2026-10-18/official/{r['policy']}/seed17/checkpoint.pt`; repliche nelle cartelle `seed18` e `seed19`. Richiede feature AlphaGenome native `emb_1bp` a 1 Mb e gli ID CpG globali, oltre alle 23 annotazioni e al contratto delle 4.165 tracce funzionali. La cache Array precedente a 512 kb non è intercambiabile.

{table(selected,['policy','step','known_mse','novel_array_mse','known_mae','novel_array_mae','score'])}

## Risultati appaiati sulle tre viste pubblicate

{table(means,['view','model','mse','mae','mas_pcc','within_tissue','severe'])}

`mas_pcc` è la mediana delle correlazioni entro CpG attraverso i pazienti; le medie sui tre seed sono medie delle metriche, non un ensemble. MSE e MAE usano gli stessi target finiti per entrambi i metodi. Le predizioni pubblicate sono valori di inferenza BF16; le nostre predizioni vengono calcolate FP32 per la valutazione. `output_BF16` è un controllo dell’arrotondamento finale dei nostri output: non equivale a rifare tutto il forward BF16. `mixed_raw` mostra il modello prima della calibrazione, per non attribuire alla rete un guadagno dovuto agli offset.

Le aggregazioni originali sono riportate anche su entrambi gli assi. Le sei metriche ricalcolate di MethylProphet (MSE, MAE, media/mediana PCC per CpG e per paziente) sono verificate contro i JSON degli autori entro tolleranza assoluta 2e-5.

{table(paper,['view','model','cpg_pcc_mean','mas_pcc','sample_pcc_mean','sample_pcc_median'])}

## Differenze e incertezza

Segno negativo: MSE del nostro modello inferiore a MethylProphet. Bootstrap appaiato con 2.000 ricampionamenti indipendenti di pazienti e blocchi genomici da 5 Mb. Gli intervalli sono condizionati ai modelli stimati e non comprendono tutta la variabilità di training. Le loss dei tre seed sono mediate senza trattarli come pazienti indipendenti.

{table(primary,['view','mse_delta_percent','ci95_percent_low','ci95_percent_high'])}

{table(cpg_summary,['view','loci','fraction_improved','median_delta_mse'])}

## Quali CpG sono davvero nuove?

Nel protocollo misto **6.373 delle 6.742 CpG escluse dall’Array sono osservate in EPIC/WGBS; solo 369 sono escluse da tutte le fonti training**. Questo è mantenuto identico per entrambi i metodi. Il pannello Array-held nel suo insieme non viene chiamato zero-shot genomico; il sottoinsieme di 369 loci viene riportato separatamente, con la cautela dovuta alle dimensioni ridotte.

{table(strata,['view','scope','loci','our_mse','reference_mse','delta_mse_percent','our_mae','reference_mae','our_pcc','reference_pcc'])}

## Controllo con esposizione alle coppie simile al modello pubblicato

La configurazione originale specifica 55.524 aggiornamenti, batch locale 512 e 16 GPU, cioè **454.852.608 esposizioni a coppie**. Prima della selezione primaria e di qualsiasi scoring pubblicato è stata fissata una regola secondaria: scegliere fra i checkpoint già previsti quello con numero medio effettivo di coppie finite campionate più vicino a questo valore. Nessuna metrica test sceglie questo checkpoint.

Controllo: **{budget_meta['selected']['step']} step, {budget_meta['selected']['mean_finite_pairs']:,.0f} coppie finite medie** sui tre seed. È un confronto di esposizione approssimata, non di FLOPs, tempo o costo del pretraining. I risultati primari restano quelli della ricetta selezionata internamente; il controllo verifica quanto dipendano da un numero diverso di coppie campionate.

{table(budget,['view','reference_mse','mse','delta_mse_percent','mae','mas_pcc'])}

## Riproduzione e artefatti

Le predizioni ufficiali provengono da [MethylProphet/eval-tcga_mix_chr1-bs_512-c2b2](https://huggingface.co/datasets/MethylProphet/eval-tcga_mix_chr1-bs_512-c2b2); la configurazione del modello da [MethylProphet/tcga_mix_chr1-bs_512-c2b2](https://huggingface.co/MethylProphet/tcga_mix_chr1-bs_512-c2b2). Revisione dataset: `{audit['released_source_revision']}`. Hash e file originali nel sottoalbero `reference/`. L’audit verifica tutti gli ID, tutti i target e le maschere finite, senza selezionare sulla performance pubblicata.

[Figura](experiments/alphagenome_mixed_2026-10-18.png) · [PDF](experiments/alphagenome_mixed_2026-10-18.pdf) · [Metriche](experiments/alphagenome_mixed_2026-10-18_metrics.csv) · [Contrasti appaiati](experiments/alphagenome_mixed_2026-10-18_paired_contrasts.csv) · [Esposizione delle CpG](experiments/alphagenome_mixed_2026-10-18_training_exposure_strata.csv) · [Errori maggiori](experiments/alphagenome_mixed_2026-10-18_worst_cpgs.csv).

Comandi: `python scripts/fetch_methylprophet_reference.py` (con accesso Hugging Face già configurato per i repository), `python scripts/train_alphagenome_mixed.py --stage all`, `python scripts/compare_alphagenome_methylprophet.py --stage score`, `python scripts/evaluate_alphagenome_mixed_budget.py`, `python scripts/report_alphagenome_mixed.py` (matplotlib disponibile per il report). Protocolli e snapshot in `artifacts/alphagenome_mixed_2026-10-18/`. Il confronto pubblicato qui riguarda il rilascio chr1; non dimostra superiorità su altri cromosomi o su EPIC/WGBS indipendenti, ora usati nel training.

La famiglia architetturale è stata sviluppata in precedenza su altri split di questo stesso dataset Array. Il refit e la selezione di questa campagna escludono i pazienti test corretti, ma il dataset non è una coorte mai esaminata durante la ricerca. Per una conferma indipendente servirà una nuova coorte.
'''
    DOC.write_text(text)
    (out/'report_complete.json').write_text(json.dumps(dict(complete=True,report=str(DOC),primary=primary.to_dict('records')),indent=2)+'\n')
    print(primary.to_string(index=False))


if __name__=='__main__':
    main()
