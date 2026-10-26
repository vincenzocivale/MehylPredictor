"""Render the complete hypothesis-validation campaign, including negative results."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import argparse
import h5py
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/alphagenome_hypotheses_2026-10-18'
TAG='alphagenome_hypotheses_2026-10-18'


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def save(path,value):path.write_text(json.dumps(clean(value),indent=2,ensure_ascii=False)+'\n')


def table(frame,columns,formats=None):
    formats=formats or {}
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,row in frame.iterrows():
        values=[]
        for c in columns:
            v=row[c]
            if pd.isna(v):v='—'
            elif c in formats:v=formats[c](v)
            elif c in {'patients','cpgs','support','parameters','train_patients','large_effect_cpgs'}:v=str(int(v))
            elif c=='delta_mse_percent':v=f'{v:+.2f}'
            elif isinstance(v,(float,np.floating)):v=f'{v:.6f}'
            values.append(str(v))
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def block_difference(frame,other,seed=20261018):
    merged=frame[['cpg_idx','chr','pos','mse','n']].merge(other[['cpg_idx','mse']],on='cpg_idx',suffixes=('','_base'),validate='one_to_one')
    groups=merged.chr.astype(str)+':'+(merged.pos//5_000_000).astype(str)
    sums=pd.DataFrame(dict(group=groups,delta=(merged.mse-merged.mse_base)*merged.n,n=merged.n)).groupby('group').sum()
    rng=np.random.default_rng(seed);samples=rng.integers(len(sums),size=(2000,len(sums)))
    delta=sums.delta.to_numpy()[samples].sum(1)/sums.n.to_numpy()[samples].sum(1)
    return dict(delta_mse=float(np.average(merged.mse-merged.mse_base,weights=merged.n)),
                ci95_genomic_blocks=np.quantile(delta,[.025,.975]).tolist(),blocks=len(sums),
                limitation='Conditional on fixed evaluation patients; not joint patient/locus uncertainty')


def cohort_metadata():
    root=ROOT.parent/'data/MethylPredictorData/methylation';result={};sample_sets={}
    for source,file in [('array','tcga_array_official_full.h5'),('epic','epic_full.h5'),('wgbs','wgbs_full.h5')]:
        with h5py.File(root/file) as f:
            names=f['raw_sample_name'].asstr()[:]
            result[source]=dict(measurements=len(names),unique_patient_ids=len(np.unique(f['sample_idx'][:])),
                                raw_name_prefix_counts=pd.Series([n.split('-')[0] for n in names]).value_counts().to_dict(),
                                provenance=json.loads(f['provenance_json'][()].decode()))
            sample_sets[source]=set(names)
    for source in ['epic','wgbs']:
        result[source]['raw_name_overlap_array']=len(sample_sets['array']&sample_sets[source])
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=OUT)
    args=ap.parse_args();out=args.out
    required=['sequence_complete.json','tissue_complete.json','transfer_complete.json','reliability_complete.json','named_rna/complete.json']
    for name in required:assert (out/name).exists(),f'Incomplete campaign: {name}'
    files=ROOT/'docs/experiments';files.mkdir(exist_ok=True)
    rna=pd.read_csv(out/'rna_signal.csv');shots=pd.read_csv(out/'few_shot.csv')
    named=pd.read_csv(out/'named_rna/metrics.csv');seq=pd.read_csv(out/'sequence_metrics.csv')
    tissue=pd.read_csv(out/'tissue_metrics.csv');transfer=pd.read_csv(out/'transfer_metrics.csv')
    risk=pd.read_csv(out/'risk_controls.csv');external_risk=pd.read_csv(out/'external_risk_coverage.csv')
    external_adaptive=pd.read_csv(out/'external_adaptive_metrics.csv')
    adaptive=json.loads((out/'adaptive.json').read_text());qc=json.loads((out/'hard_cpgs/summary.json').read_text())
    tau=json.loads((out/'few_shot_recipe.json').read_text())['tau']
    named['arm']=np.where(named.shuffle==-1,'RNA corretta','RNA permutata entro tessuto')
    named_table=named.groupby(['view','arm'],as_index=False)[['mse','mae','mas_pcc','within_tissue','patients']].mean()
    dev=rna[rna.split=='dev']
    controls=dev.groupby(['view','source','arm'],as_index=False)[['mse','mae','mas_pcc','within_tissue']].mean()
    few=shots[(shots.split=='dev')&((shots.tau==tau)|(shots.support==0))].groupby('support',as_index=False)[['mse','mae','mas_pcc']].mean()
    zero=float(few.loc[few.support==0,'mse'].iloc[0]);few['delta_mse_percent']=100*(few.mse/zero-1)
    seqtable=seq[seq.split=='dev'].groupby('arm',as_index=False)[['mse','mae','mas_pcc','within_tissue','parameters']].mean()
    base=controls[(controls.view=='novel')&(controls.source=='v2_early')&(controls.arm=='prior0.25')].iloc[0]
    seqtable['delta_mse_percent']=100*(seqtable.mse/base.mse-1)
    seqtable['novel_guards']=(seqtable.mse<=base.mse*.995)&(seqtable.mae<=base.mae)&(seqtable.mas_pcc>=base.mas_pcc-.002)&(seqtable.within_tissue>=base.within_tissue-.002)
    tt=tissue[(tissue.population=='held_tissue')&(tissue.arm=='hierarchical')].groupby(['group','view','mode'],as_index=False)[['mse','mae','mas_pcc','patients','train_patients']].mean()
    primary=transfer[transfer.seed==17].copy();primary['delta_mse_percent']=0.
    ci={}
    for view in primary.view.unique():
        b=primary[(primary.view==view)&(primary.model=='v1')].mse.iloc[0]
        primary.loc[primary.view==view,'delta_mse_percent']=100*(primary.loc[primary.view==view,'mse']/b-1)
        frame=pd.read_csv(out/'transfer'/view/'v3_seed17_per_cpg.csv.gz')
        other=pd.read_csv(out/'transfer'/view/'v1_seed17_per_cpg.csv.gz')
        ci[view]=block_difference(frame,other)
    pri=primary[primary.model.isin(['v1','v3'])][['view','model','patients','cpgs','mse','mae','mas_pcc','delta_mse_percent']]
    contrasts=json.loads((out/'dev/contrasts.json').read_text())
    metadata=cohort_metadata();save(out/'cohort_metadata.json',metadata)
    summary=dict(rna_named=named_table.to_dict('records'),controls_dev=controls.to_dict('records'),
                 few_shot_tau=tau,few_shot_dev=few.to_dict('records'),adaptive=adaptive,
                 sequence_dev=seqtable.to_dict('records'),tissue_held_out=tt.to_dict('records'),
                 transfer_primary=pri.to_dict('records'),transfer_block_ci=ci,
                 risk_dev=risk[risk.view=='dev'].to_dict('records'),external_risk=external_risk.to_dict('records'),
                 external_adaptive=external_adaptive.to_dict('records'),hard_cpgs=qc,contrasts=contrasts,
                 cohort_metadata=metadata,new_fits=len(seq)+12,
                 limitations=['Internal axes previously used in development; fresh chromosome/assay confirmation is broader but restricted to array-overlapping loci.',
                              'AlphaGenome reference-genome pretraining is not held out by our methylation chromosome split.',
                              'EPIC/WGBS canonical tissue labels are unknown; within-tissue endpoints unavailable there.',
                              'Few-shot support consumes methylation labels on query loci; it is a separate task from RNA-only zero-shot.',
                              'Locus risk is not a calibrated patient-level interval, and low-risk selection favors low-variance loci.'])
    save(files/f'{TAG}.json',summary);save(out/'summary.json',summary)
    for label,df in [('rna_named',named_table),('few_shot',few),('sequence',seqtable),('tissue',tt),('transfer',pri),('risk',external_risk)]:
        df.to_csv(files/f'{TAG}_{label}.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
    fig,axes=plt.subplots(2,3,figsize=(17,10),layout='constrained')
    colors=['#2166ac','#b2182b','#4d9221','#762a83']
    ax=axes[0,0]
    for k,view in enumerate(['known','novel']):
        part=named_table[named_table.view==view]
        for j,arm in enumerate(['RNA corretta','RNA permutata entro tessuto']):
            ax.bar(k+(j-.5)*.32,part.loc[part.arm==arm,'mse'].iloc[0],width=.3,color=colors[j],label=arm if k==0 else None)
    ax.set(xticks=[0,1],xticklabels=['CpG note','CpG nuove'],ylabel='MSE',title='A · Segnale individuale RNA (tessuti noti)');ax.legend(fontsize=8)
    ax=axes[0,1]
    for value,label,color in [(0.,'Senza shrinkage',colors[1]),(tau,f'Shrinkage τ={tau:g}',colors[0])]:
        part=shots[(shots.split=='dev')&((shots.tau==value)|(shots.support==0))]
        grouped=part.groupby('support').mse.mean()
        ax.plot(grouped.index,grouped.values,'o-',label=label,color=color)
    ax.set(xscale='symlog',xlabel='Pazienti di supporto',ylabel='MSE',title='B · Adattamento a nuove CpG');ax.set_xticks([0,1,5,10,50],[0,1,5,10,50]);ax.legend(fontsize=8)
    ax=axes[0,2]
    for k,label in enumerate(['learned','seed_disagreement','predicted_sd','prior_midpoint']):
        part=risk[(risk.view=='dev')&(risk.estimator==label)]
        ax.plot(part.coverage,part.mse,'o-',label=label,color=colors[k])
    ax.set(xlabel='Frazione di CpG mantenute',ylabel='MSE',title='C · Affidabilità sul dev');ax.legend(fontsize=8)
    ax=axes[1,0];v3=primary[primary.model=='v3']
    ax.barh(v3.view,v3.delta_mse_percent,color=colors[0]);ax.axvline(0,color='black',lw=.7)
    ax.set(xlabel='Δ MSE % rispetto a v1 (seed17)',title='D · Cromosomi e piattaforme di conferma')
    ax=axes[1,1];new=tt[tt.view=='novel'];groups=new.group.unique()
    for j,mode in enumerate(['matched_random','leave_tissue_out']):
        values=[new[(new.group==g)&(new['mode']==mode)].mse.iloc[0] for g in groups]
        ax.bar(np.arange(len(groups))+(j-.5)*.32,values,.3,label=mode,color=colors[j])
    ax.set(xticks=np.arange(len(groups)),xticklabels=groups,ylabel='MSE nuove CpG',title='E · Tessuto escluso dal training');ax.legend(fontsize=8)
    ax=axes[1,2]
    ax.barh(seqtable.arm,seqtable.delta_mse_percent,color=colors[0]);ax.axvline(0,color='black',lw=.7)
    ax.set(xlabel='Δ MSE % rispetto a v3 sul dev',title='F · Correzioni genomiche addestrate')
    fig.savefig(files/f'{TAG}.png',dpi=180);fig.savefig(files/f'{TAG}.pdf');plt.close(fig)
    riskdev=risk[(risk.view=='dev')&(risk.coverage==.5)]
    riskext=external_risk[external_risk.coverage==.5]
    adaptive_dev=next(r for r in adaptive if r['split']=='dev')
    text=f'''# Validazione delle ipotesi AlphaGenome — 18 ottobre 2026

Campagna completata: tre fold genomici interni, tre seed e conferma dev; **{len(seq)+12} nuovi fit** (48 correzioni genomiche e 12 esclusioni di tessuto), controlli RNA, calibrazione con pochi campioni, affidabilità e conferme fuori da chr1. I risultati negativi sono inclusi. Nessun modello è promosso automaticamente.

## 1. Quanto segnale individuale contiene l’RNA?

Conferma limitata ai pazienti con tessuto esplicito: lo shuffle avviene esclusivamente entro il medesimo tessuto. I campioni NA sono esclusi da questa tabella; nel controllo originario erano un gruppo convenzionale e la loro permutazione poteva mescolare tessuti diversi. Media dei seed; per lo shuffle anche delle tre permutazioni.

{table(named_table,['view','arm','patients','mse','mae','mas_pcc','within_tissue'])}

Controlli sul dev completo; statistiche empiriche e prior Ridge stimati soltanto sui pazienti/loci training. Il baseline tessuto+genoma usa il tessuto osservato e non deve essere interpretato come RNA-only.

{table(controls[controls.arm.isin(['raw','calibrated','empirical_mean','empirical_tissue','genomic_mean','genomic_tissue','prior0.25'])],['view','source','arm','mse','mae','mas_pcc'])}

La calibrazione della v1 è competitiva con quella della v2/v3 sulle CpG note: il miglioramento rispetto a v1 non calibrata non identifica un vantaggio specifico della nuova rappresentazione. Lo shuffle e i controlli di tessuto verificano invece l’informazione individuale RNA. Una variazione della correlazione dopo la calibrazione è possibile per la sigmoid, anche a fattori fissi.

## 2. Da zero osservazioni a pochi campioni di supporto

Il modello resta congelato. Per CpG mai usate nel fit, 1/5/10/50 pazienti training forniscono metilazione di supporto; gli altri pazienti, esclusi dal fit e dal supporto, misurano l’errore. Cinque insiemi di supporto annidati, tre seed. Il peso empirico è n/(n+τ), n numero di osservazioni finite per locus. **τ={tau:g}** selezionato sui tre fold interni, senza ottimizzazione sul dev. Con nessuna osservazione il prior è mantenuto; livelli desiderati limitati a [0,001;0,999].

{table(few,['support','mse','mae','mas_pcc','delta_mse_percent'])}

L’adattamento senza shrinkage con un solo campione può peggiorare l’errore. Questo esperimento cambia il problema applicativo: richiede metilazione dei nuovi loci nei campioni di supporto. Non è un miglioramento zero-shot senza metilazione in ingresso. Il supporto qui proviene dalla popolazione Array; non dimostra ancora adattamento few-shot a EPIC/WGBS.

## 3. CpG persistentemente difficili

Audit descrittivo delle 6.742 CpG ufficiali nuove, di cui **{qc['hard_loci']}** difficili in almeno due seed. Il manifest pubblico HM450 mappa esattamente {qc['manifest_mapped']} loci alle coordinate hg38; tutte le 6.742 posizioni corrispondono a CG nel riferimento. **{qc['hard_qc_unmasked']} delle 660 CpG difficili** non sono segnalate da MASK_general; mapping, non unicità della sequenza e SNP comuni nei 5bp terminali non identificano un sottogruppo problematico nel pannello già selezionato. Questo non esclude altri artefatti sperimentali.

Controlli abbinati per contesto, regione, media e SD osservate sono esclusivamente descrittivi. Un grande scarto di media (>0,3 beta) rispetto al vicino AlphaGenome training distante almeno 1Mb riguarda il {100*qc['nearest_mean_gap_gt03_fraction']:.1f}% dei loci e il {100*qc['hard_nearest_mean_gap_gt03_fraction']:.1f}% di quelli difficili. La similarità coseno mediana del vicino è {qc['median_nearest_ag_cosine']:.3f}: non si tratta di embedding indistinguibili, né il confronto dimostra causalità. Il risultato indica limiti del trasferimento del livello medio.

Fonti: [manifest HM450]({qc['manifest']['url']}), [documentazione delle maschere]({qc['manifest']['documentation']}). SHA256 del manifest: `{qc['manifest']['sha256']}`. Annotazioni complete, controlli abbinati e vicini sono negli artefatti `hard_cpgs/`.

## 4. Regolarizzazione adattiva e affidabilità

Random forest con iperparametri prefissati, addestrata sugli errori delle altre regioni genomiche. Nessuna CpG valutata è presente nei target di fit della regola. Le feature sono annotazioni e statistiche delle predizioni: nessuna metilazione del locus interrogato. I fold interni condividono pazienti di valutazione, mentre i pazienti dev sono distinti; il dev è quindi il controllo più informativo sul trasferimento.

Sul dev la selezione per locus fra pesi 0/0,25/0,5 passa da MSE {adaptive_dev['fixed025_mse']:.6f} a **{adaptive_dev['adaptive_mse']:.6f}** ({100*(adaptive_dev['adaptive_mse']/adaptive_dev['fixed025_mse']-1):+.2f}%). MAE e correlazioni vanno considerate insieme: minimizzare la MSE non garantisce una dominanza sulle altre metriche. La regola adattiva resta sperimentale.

Affidabilità sul dev, mantenendo il 50% delle CpG ritenute più sicure:

{table(riskdev,['estimator','mse','error_prediction_spearman','hard_decile_auc'])}

Conferma della stessa regola, senza rifit su etichette esterne:

{table(riskext,['view','estimator','mse','hard_decile_auc'])}

Il baseline distanza della media da 0,5 è importante: selezionare CpG poco variabili è già relativamente facile. Questi risultati descrivono rischio medio per locus nella popolazione RNA interrogata; non sono intervalli predittivi calibrati per il singolo paziente. Si riportano tutti i controlli, compresi quelli che eguagliano o superano la regola appresa sotto shift.

## 5. Sequenza locale, nuovi cromosomi, tessuti e piattaforme

Correzioni addestrate da zero sopra v2 congelata: MLP AlphaGenome, CNN DNA di 1.002bp centrata sul dinucleotide, combinazione, e sequenze permutate come controllo negativo. La CNN media i due orientamenti reverse complement. Si correggono intercetta e fattori RNA con 1.000 step prefissati; il peso del prior sulle nuove CpG resta 0,25. I rami hanno capacità diverse, esplicitate nella tabella; non è un confronto a uguale numero di parametri. Tutti gli encoder RNA originali restano congelati.

{table(seqtable,['arm','parameters','mse','mae','mas_pcc','within_tissue','delta_mse_percent','novel_guards'])}

Le sequenze permutate permettono di distinguere il contributo della sequenza da una semplice correzione aggiuntiva. La CNN allineata non supera il controllo con sequenze permutate; nessuno dei rami supera i criteri sulle nuove CpG. Il risultato è negativo per questa implementazione. Questo test riguarda una piccola CNN locale su backbone congelato; non esaurisce finestre più ampie o riaddestramento congiunto. `novel_guards` richiede almeno −0,5% MSE senza peggiorare MAE, e PCC/within-tissue entro −0,002; non comprende i controlli sulle CpG note, non valutate per questi rami.

Conferma su **tutti** i loci Array di chr2 e chr22, mai usati dal nostro predittore chr1, e su pazienti EPIC/WGBS senza sovrapposizione di ID con Array. Confronto primario seed17; v2/v3 disponibili anche per seed18/19. Le fonti Array/EPIC/WGBS provengono dal pacchetto canonico MethylProphet; EPIC include campioni con prefissi C3L/C3N, HCM, HTMCP e TARGET, WGBS campioni TCGA. Il solo ID disgiunto non certifica assenza di ogni possibile dipendenza di studio. EPIC/WGBS non hanno tessuti canonici annotati, quindi la correlazione entro tessuto non è stimabile.

{table(pri,['view','model','patients','cpgs','mse','mae','mas_pcc','delta_mse_percent'])}

Loci noti e nuovi sono definiti rispetto al training del modello chr1. La valutazione EPIC/WGBS è limitata alle CpG comuni al pannello Array chr1: non dimostra copertura dei milioni di loci esclusivi WGBS. AlphaGenome può aver visto queste sequenze nel pretraining di altri task: i cromosomi sono nuovi per il nostro fit della metilazione, non necessariamente per AlphaGenome. I risultati di conferma non hanno modificato le ricette.

WGBS contiene 32 misure e 31 pazienti distinti. Due misure dello stesso paziente sono mediate per CpG, ignorando i valori mancanti, prima delle metriche; il paziente mantiene peso uno. La correzione del lettore è tracciata in `transfer_protocol_amendment.json`, dopo il primo accesso a WGBS; le valutazioni chr2/chr22/EPIC già completate non sono cambiate.

Esclusione completa di BRCA o LGG dal training, contro rimozione casuale dello stesso numero di pazienti; stessi loci e stessi pazienti di test. Tre seed, dodici fit nuovi. Si riporta la popolazione del tessuto escluso, con calibrazione gerarchica.

{table(tt,['group','view','mode','patients','train_patients','mse','mae','mas_pcc'])}

Differenze BRCA–LGG sul dev: contrasto definito dai due gruppi più numerosi nel training, quindi indipendente dai target di valutazione. Si misurano effetti per CpG, non regioni DMR né effetti causali.

{table(pd.DataFrame(contrasts),['arm','effect_mse','effect_corr','large_effect_cpgs','direction_accuracy'])}

## Interpretazione e limiti

I controlli permettono di distinguere informazione individuale RNA, calibrazione del livello medio e trasferimento del contesto genomico. Il segnale RNA entro tessuto può ancora includere purezza tumorale, composizione cellulare e batch: non dimostra una risposta intrinseca delle singole cellule. L’adattamento con pochi campioni e la stima del rischio sono risultati da sviluppare; l’aumento indiscriminato della capacità o la sola selezione della MSE non risolvono tutti i compromessi. I confronti calibrati sono necessari per attribuire correttamente i guadagni architetturali.

La v3 trasferisce il vantaggio di MSE a chr2 e chr22, ma non domina sulle altre metriche; sulle CpG nuove WGBS non supera v1. L’esclusione di un tessuto peggiora la risposta anche con lo stesso numero di pazienti training: calibrare il livello medio non risolve il trasferimento dello stato biologico. La regola adattiva del prior migliora lievemente MSE su Array, ma non risolve MAE e non dà un vantaggio affidabile su EPIC/WGBS. Nessuna di queste varianti è una sostituzione universale della ricetta esistente.

Fold e dev sono già stati usati nello sviluppo. L’embargo interno di 300kb non rende completamente disgiunte tutte le finestre AlphaGenome da 524kb; la conferma su cromosomi diversi riduce questa ambiguità per il fit del nostro decoder. Gli intervalli bootstrap in JSON ricampionano blocchi genomici di 5Mb e sono condizionati ai pazienti osservati; non sostituiscono una replica con pazienti/studi indipendenti. Tre seed non rappresentano tre coorti. Le maschere di qualità consultate non coprono tutti i fattori sperimentali. Nessuna interpretazione biologica causale è dimostrata.

## Riproducibilità

Script: `validate_alphagenome_hypotheses.py`, `validate_alphagenome_named_rna.py`, `audit_alphagenome_hard_cpgs.py`, `validate_alphagenome_sequence.py`, `validate_alphagenome_tissue.py`, `validate_alphagenome_transfer.py`, `validate_alphagenome_reliability.py`; rendering `report_alphagenome_hypotheses.py`. Tutti in `scripts/`. Protocolli, assi, checkpoint, per-CpG e hash in `artifacts/alphagenome_hypotheses_2026-10-18/`.

[Figura](experiments/{TAG}.png) · [PDF](experiments/{TAG}.pdf) · [Metriche complete](experiments/{TAG}.json). La suite di test e i controlli finali sono tracciati in `verification.json` negli artefatti.
'''
    report=ROOT/'docs/ALPHAGENOME_HYPOTHESES_VALIDATION_2026-10-18.md';report.write_text(text)
    save(out/'report_complete.json',dict(complete=True,report=str(report),summary_sha256=hashlib.sha256((out/'summary.json').read_bytes()).hexdigest()))
    print(json.dumps(dict(report=str(report),new_fits=summary['new_fits'])))


if __name__=='__main__':main()
