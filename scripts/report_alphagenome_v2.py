"""Export v2 CpG error tables, biological strata, figures and a readable report.

Consumes frozen predictions' aggregate errors only. No model selection or fit.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/methylpredictor-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'artifacts/alphagenome_v2_2026-10-18')
    args=ap.parse_args(); out=args.out.resolve()
    complete=read(out/'complete.json'); freeze=read(out/'frozen_recipe.json')
    official=out/'official'
    baseline=read(official/'baseline_v1/metrics.json')
    seeds={s:read(official/f'seed{s}/metrics.json') for s in [17,18,19]}
    frame=pd.read_csv(official/'seed17/double_ood_vs_v1.csv.gz')
    # Human-readable cg identifiers are annotations joined by exact hg38
    # coordinate; global CpG registry IDs and coordinates remain primary keys.
    mapping=ROOT.parent/'CpGRepresentationBenchmark/data/processed/ComputAgeBench/illumina_probe_grch38.parquet'
    mapped=0
    if mapping.exists():
        probes=pd.read_parquet(mapping)
        probes=probes[probes['chr'].isin(['chr1','1'])].copy()
        probes=probes.groupby('pos').probe_id.agg(lambda x:';'.join(sorted(set(x)))).rename('illumina_probe_ids')
        frame=frame.merge(probes,left_on='pos_1based',right_index=True,how='left',validate='many_to_one')
        mapped=int(frame.illumina_probe_ids.notna().sum())
    else: frame['illumina_probe_ids']=None
    source=np.load(ROOT.parent/'chr1_alphagenome_v1_2026-10-17/cache/array_chr1.npz',allow_pickle=True)
    fit=set(np.load(out/'official/mean_prior.npz')['fit_cpg_idx'].tolist())
    trainpos=np.sort(source['cpg_pos'][np.isin(source['cpg_idx'],list(fit))])
    loc=np.searchsorted(trainpos,frame.pos_1based.to_numpy())
    left=trainpos[np.maximum(loc-1,0)]; right=trainpos[np.minimum(loc,len(trainpos)-1)]
    frame['nearest_training_cpg_bp']=np.minimum(abs(frame.pos_1based-left),abs(frame.pos_1based-right))
    frame['bias_fraction']=frame.bias2/frame.mse.clip(lower=1e-12)
    frame['mse_over_oracle_mean_mse']=frame.mse/(frame['std']**2).clip(lower=1e-12)
    frame['mean_dominated_error']=frame.bias_fraction>.5
    predictions=[]; correlations=[]; worst=[]
    for seed in seeds:
        f=pd.read_csv(official/f'seed{seed}/double_ood_per_cpg.csv.gz')
        assert np.array_equal(f.cpg_idx,frame.cpg_idx)
        predictions.append(f.mse.to_numpy()); correlations.append(f['corr'].to_numpy())
        worst.append(f.mse.to_numpy()>=f.mse.quantile(.9))
    predictions=np.array(predictions); worst=np.array(worst)
    frame['v2_mse_mean_seeds']=predictions.mean(0)
    frame['v2_mse_min_seeds']=predictions.min(0); frame['v2_mse_max_seeds']=predictions.max(0)
    frame['v2_pcc_mean_seeds']=np.nanmean(correlations,0)
    frame['worst_decile_seed_count']=worst.sum(0)
    frame['v2_better_v1_seed_count']=(predictions<frame.v1_mse.to_numpy()[None]).sum(0)
    frame['v1_pred_mean']=frame['mean']+frame.v1_bias
    frame['delta_bias2']=frame.bias2-frame.v1_bias**2
    frame['delta_centered_mse']=frame.centered_mse-(frame.v1_mse-frame.v1_bias**2)
    frame.sort_values('mse',ascending=False,inplace=True)
    frame.to_csv(official/'double_ood_annotated_errors.csv.gz',index=False)
    frame.head(100).to_csv(official/'double_ood_annotated_worst100.csv',index=False)
    names=['cpg_idx','illumina_probe_ids','coordinate','context','region','n','mse','rmse','mae','bias',
           'mean','pred_mean','std','pred_std','corr','within_corr','severe','abs_error_p90','abs_error_p95',
           'max_abs_error','error_gt_0p1_fraction','v1_mse','v1_mae','v1_bias','v1_corr','v1_severe',
           'v1_pred_mean','delta_mse','delta_mae','delta_bias','delta_corr','delta_severe','bias_fraction',
           'delta_bias2','delta_centered_mse',
           'bias2','centered_mse','std_ratio','mse_over_oracle_mean_mse','mean_dominated_error',
           'worst_decile_seed_count','v2_mse_mean_seeds','v2_mse_min_seeds','v2_mse_max_seeds',
           'v2_pcc_mean_seeds','v2_better_v1_seed_count','nearest_training_cpg_bp']
    target=ROOT/'docs/experiments/alphagenome_v2_2026-10-18'
    frame[names].to_csv(target.with_name(target.name+'_errors.csv.gz'),index=False)
    frame[names].head(100).to_csv(target.with_name(target.name+'_worst100.csv'),index=False)
    frame.sort_values('delta_mse',ascending=False)[names].head(100).to_csv(target.with_name(target.name+'_regressions100.csv'),index=False)
    group_records=[]
    for (context,sd),part in frame.groupby(['context','observed_sd_bin'],observed=True):
        group_records.append(dict(context=str(context),observed_sd_bin=str(sd),cpgs=len(part),
                                   mse=float(np.average(part.mse,weights=part.n)),
                                   v1_mse=float(np.average(part.v1_mse,weights=part.n)),
                                   fraction_improved=float(np.mean(part.delta_mse<0))))
    summary=dict(freeze=freeze,baseline_v1=baseline,official_v2={str(s):r for s,r in seeds.items()},
                 cg_mapping=dict(source=str(mapping),mapped=mapped,total=len(frame),join='hg38 chromosome + 1-based cytosine'),
                 stable_worst_decile_cpgs=int(np.sum(worst.sum(0)>=2)),
                 joint_context_sd_diagnostics=group_records,
                 worst20=frame[names].head(20).replace({np.nan:None}).to_dict(orient='records'))
    selected=freeze['selected']; dev={}
    for label,arm,alpha in [('control','baseline',0.),('v2',selected['arm'],selected['mean_blend'])]:
        docs=[read(out/'dev'/arm/f'seed{s}/validation.json')[str(alpha)] for s in seeds]
        dev[label]={view:{k:float(np.mean([d[view][k] for d in docs])) for k in docs[0][view]} for view in docs[0]}
    summary['dev_confirmation']=dev
    distances=[]
    bins=pd.cut(frame.nearest_training_cpg_bp,[-1,100,1000,10000,np.inf],labels=['<=100bp','100bp-1kb','1-10kb','>10kb'])
    for label,part in frame.groupby(bins,observed=True):
        distances.append(dict(distance_bin=str(label),cpgs=len(part),
                              mse=float(np.average(part.mse,weights=part.n)),
                              v1_mse=float(np.average(part.v1_mse,weights=part.n)),
                              fraction_improved=float(np.mean(part.delta_mse<0))))
    summary['distance_diagnostics']=distances
    summary['error_change_decomposition']=dict(
        delta_bias2=float(np.average(frame.delta_bias2,weights=frame.n)),
        delta_centered_mse=float(np.average(frame.delta_centered_mse,weights=frame.n)),
        fraction_bias2_improved=float(np.mean(frame.delta_bias2<0)),
        fraction_centered_mse_improved=float(np.mean(frame.delta_centered_mse<0)))
    target.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
    figure(frame,target)
    report(out,target,frame,summary)
    print(json.dumps({'report':str(ROOT/'docs/TCGA_CHR1_ALPHAGENOME_V2_2026-10-18.md'),
                      'mapped_cpgs':mapped,'cpgs':len(frame)}),flush=True)


def figure(f,target):
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    colors={'island':'#4477aa','shore':'#228833','shelf':'#ccbb44','open_sea':'#aa3377'}
    ax=axes[0,0]
    for context,color in colors.items():
        part=f[f.context==context]
        ax.scatter(part.v1_mse,part.mse,s=7,alpha=.45,color=color,label=context,rasterized=True)
    lim=max(f.v1_mse.max(),f.mse.max())*1.03
    ax.plot([0,lim],[0,lim],ls='--',color='#666666',lw=1)
    ax.set(xlim=(0,lim),ylim=(0,lim),xlabel='v1 per-CpG MSE',ylabel='v2 per-CpG MSE',title='A. Same held-out patients and CpGs')
    ax.legend(fontsize=8)
    ax=axes[0,1]
    ax.scatter(f['std'],f.rmse,s=8,alpha=.4,color='#136f9b',rasterized=True)
    ax.set(xlabel='Observed methylation SD',ylabel='RMSE (beta units)',title='B. Error and observed variability')
    ax=axes[1,0]; labels=list(colors)
    bias=[]; centered=[]; old=[]
    for name in labels:
        part=f[f.context==name]; w=part.n
        bias.append(np.average(part.bias2,weights=w)); centered.append(np.average(part.centered_mse,weights=w))
        old.append(np.average(part.v1_mse,weights=w))
    ax.bar(labels,bias,color='#c97a3a',label='v2 squared mean bias')
    ax.bar(labels,centered,bottom=bias,color='#136f9b',label='v2 centered error')
    ax.scatter(labels,old,color='#333333',marker='_',s=160,label='v1 total MSE',zorder=3)
    ax.set(ylabel='MSE',title='C. Error decomposition by CpG context'); ax.legend(fontsize=8)
    ax=axes[1,1]
    bound=np.quantile(abs(f.bias),.99)
    points=ax.scatter(f.pos_1based/1e6,f.rmse,c=f.bias,cmap='coolwarm',vmin=-bound,vmax=bound,
                      s=9,alpha=.65,rasterized=True)
    fig.colorbar(points,ax=ax,label='Mean signed error (predicted - observed)')
    ax.set(xlabel='chr1 position (Mb, hg38)',ylabel='RMSE (beta units)',title='D. Location and direction of errors')
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.13); ax.set_axisbelow(True)
    fig.suptitle('AlphaGenome v2: official double-OOD, primary seed 17',fontsize=14)
    fig.savefig(target.with_suffix('.png'),dpi=180)
    fig.savefig(target.with_suffix('.pdf'))


def report(out,target,f,summary):
    display_out=out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    freeze=summary['freeze']; sel=freeze['selected']; seeds=summary['official_v2']
    primary=seeds['17']; baseline=summary['baseline_v1']
    lines=['# AlphaGenome chr1 v2: training e audit degli errori CpG','',
           'Versione creata e addestrata il 18 ottobre 2026. Selezione sui soli fold interni, '
           'refit su tutti gli assi training ufficiali, poi valutazione delle tre viste ufficiali.','',
           '## Ricetta selezionata','',
           f"Braccio **{sel['arm']}**, mean blend **{sel['mean_blend']}**, 1.000 aggiornamenti, EMA 0,99; seed 17 primario e repliche 18/19. "
           f"Guardrail interni superati: **{freeze['guardrails_passed']}**.",'',
           (f"Configurazione attiva: intercetta shallow={sel['arm']!='baseline'}; "
            f"penalità locale={dict(baseline=0.,mean=0.,local01=.01,local10=.1)[sel['arm']]}; "
            f"blend del prior={sel['mean_blend']}."),'',
           ('La calibrazione verso il prior non è attiva nella ricetta selezionata.'
            if sel['mean_blend']==0 else 'La calibrazione della media è attiva nel checkpoint selezionato.'),'',
           ('Il candidato supera i guardrail interni prefissati.' if freeze['guardrails_passed'] else
            '**Nessun candidato nuovo supera tutti i guardrail interni. La v2 viene consegnata come candidata sperimentale, con il compromesso esplicitato; non è una superiorità dimostrata sulla v1.**'),'',
           'Il decoder resta fattorizzato a rango 32. La famiglia candidata confronta intercetta shallow/deep, '
           'regolarizzazione locale delle risposte RNA (peso 0 / 0,01 / 0,1) e calibrazione della media (blend 0 / 0,5 / 1). '
           'La calibrazione risolve un offset affinché la media prevista su 512 fattori RNA training corrisponda '
           'alla combinazione fra media del modello e prior Ridge AlphaGenome. Il prior è fittato soltanto sulle '
           'medie dei pazienti e loci training; nessuna metilazione della CpG nuova è un input.','',
           'I vicini del grafo sono cinque CpG training, distanti almeno 1 Mb, scelte nello spazio AlphaGenome '
           'standardizzato/PCA64 e appaiate per contesto, regione e quartile TSS; fallback globale se necessario. '
           'La penalità confronta risposte RNA centrate, non intercette o coefficienti dipendenti dalla base. '
           'Il grafo serve solo nel training.','',
           '## Selezione interna e generalizzazione','',
           'Tre seed × tre fold genomici da 5 Mb con embargo 300 kb; pazienti esclusi separatamente. '
           'Budget, lambda e blend prefissati prima dei risultati. Si sceglie la minima MSE double-OOD '
           'con PCC e within-tissue non inferiori di oltre 0,005 al controllo; sui loci noti PCC non inferiore '
           'di oltre 0,01 e MSE entro +5%. Se nessun candidato passa, si addestra e riporta il miglior '
           'candidato nuovo dichiarandolo non confermato. Lo split dev successivo non cambia la scelta.','',
           '| Braccio | Blend | MSE interna | PCC interna | MSE loci noti | Ammissibile |',
           '|---|---:|---:|---:|---:|---|']
    for c in freeze['candidates']:
        lines.append(f"| {c['arm']} | {c['mean_blend']} | {c['double_ood']['mse']:.6f} | {c['double_ood']['mas_pcc']:.5f} | {c['known_loci_new_patients']['mse']:.6f} | {c['admissible']} |")
    dev=summary['dev_confirmation']
    lines+=['',f"Conferma dev a ricetta congelata, media dei tre seed: MSE double-OOD "
            f"{dev['control']['double_ood']['mse']:.6f}→{dev['v2']['double_ood']['mse']:.6f}, "
            f"PCC {dev['control']['double_ood']['mas_pcc']:.5f}→{dev['v2']['double_ood']['mas_pcc']:.5f}. "
            'Questi risultati non modificano la scelta. Il dev a blocchi non è direttamente confrontabile '
            'con lo split ufficiale intercalato delle CpG.','']
    lines+=['','## Confronto ufficiale','',
            'Stessi ID, target canonici float32, preprocessing salvato separatamente per ciascun checkpoint e forward FP32. '
            'Il controllo è il checkpoint v1 seed17 già congelato. Il confronto primario seed17 è appaiato; '
            'gli altri seed v2 misurano variabilità del candidato e non sono tre baseline v1 appaiati.','',
            '| Vista | Modello | MSE | MAE | MAS-PCC | Within-tissue | Errori >0,2 beta |',
            '|---|---|---:|---:|---:|---:|---:|']
    labels={'new_patients':'Nuovi pazienti / CpG note','new_cpgs':'Nuove CpG / pazienti noti','double_ood':'Nuovi pazienti e nuove CpG'}
    for view,label in labels.items():
        for name,result in [('v1',baseline),('v2 seed17',primary)]:
            m=result['views'][view]['metrics']
            lines.append(f"| {label} | {name} | {m['mse']:.6f} | {m['mae']:.5f} | {m['mas_pcc']:.5f} | {m['within_tissue']:.5f} | {100*m['severe']:.2f}% |")
    lines+=['','| Seed v2 | MSE double-OOD | PCC double-OOD |','|---:|---:|---:|']
    for seed,result in seeds.items():
        m=result['views']['double_ood']['metrics']; lines.append(f"| {seed} | {m['mse']:.6f} | {m['mas_pcc']:.5f} |")
    comp=primary['views']['double_ood']['vs_v1']; audit=primary['views']['double_ood']['audit']
    lines+=['',f"Sul confronto primario migliora in MSE il **{100*comp['fraction_cpg_mse_improved']:.1f}% delle CpG**. "
            f"Differenza media MSE v2−v1: {comp['mse_difference']:+.6f}; intervallo bootstrap per blocchi "
            f"{comp['block_uncertainty']['ci95_blocks']}. Intervalli descrittivi, senza bootstrap pazienti o correzione multipla.",'',
            f"La differenza MSE si scompone in {summary['error_change_decomposition']['delta_bias2']:+.6f} "
            f"di bias² e {summary['error_change_decomposition']['delta_centered_mse']:+.6f} di errore centrato.",'',
            '## In quali CpG sbaglia e di quanto','',
            f"Audit di **{len(f):,} CpG nuove** sui pazienti esclusi. Il 10% peggiore dei loci concentra "
            f"il **{100*audit['worst_decile_sse_fraction']:.1f}% dell’errore quadratico**. "
            f"Il bias medio del locus rappresenta il **{100*audit['all']['bias_fraction']:.1f}% dell’SSE**. "
            f"{summary['stable_worst_decile_cpgs']} CpG sono nel decile peggiore in almeno due seed su tre.",'',
            '| Contesto | CpG | RMSE beta | MAE beta | PCC mediana | Errori >0,2 beta |',
            '|---|---:|---:|---:|---:|---:|']
    for context,values in audit['strata']['context'].items():
        lines.append(f"| {context} | {values['cpgs']} | {values['rmse']:.4f} | {values['mae']:.4f} | {values['median_pcc']:.4f} | {100*values['error_gt_0p2_fraction']:.1f}% |")
    lines+=['','Queste differenze sono descrittive: contesto e variabilità dei target sono correlati. '
            'Lo snapshot riporta anche gli incroci contesto×SD; la SD osservata non viene usata per predire. '
            'Sono disponibili anche diagnostiche per distanza dalla CpG training più vicina. '
            'Una bassa PCC su CpG quasi costanti può accompagnarsi a un errore assoluto piccolo. '
            'La SD predetta inferiore alla SD osservata può riflettere variabilità non prevedibile dall’RNA.','',
            'Le peggiori 15 CpG per MSE nel seed primario:','',
            '| CpG Illumina | Coordinate hg38, C 1-based | Contesto | RMSE beta | MAE beta | Bias | PCC | Errori >0,2 |',
            '|---|---|---|---:|---:|---:|---:|---:|']
    for row in f.head(15).itertuples():
        probe=row.illumina_probe_ids if pd.notna(row.illumina_probe_ids) else f'ID {row.cpg_idx}'
        lines.append(f"| {probe} | {row.coordinate} | {row.context} | {row.rmse:.3f} | {row.mae:.3f} | {row.bias:+.3f} | {row.corr:.3f} | {100*row.severe:.1f}% |")
    lines+=['','Bias positivo significa sovrastima della metilazione. Il CSV completo include media osservata/predetta, '
            'SD, RMSE, MAE, bias, PCC, PCC sui residui within-tissue, p90/p95 e massimo errore assoluto, '
            'frazione errori >0,1 e >0,2, distanza dalla CpG training più vicina e confronto locus per locus con v1. '
            'Il rapporto MSE/varianza osservata è solo un confronto diagnostico con una media-oracolo non disponibile in deployment.','',
            f"Identificatori Illumina disponibili per {summary['cg_mapping']['mapped']}/{len(f)} CpG. Join esatto per coordinate "
            'hg38, mantenendo tutti gli eventuali probe associati alla stessa coordinata; gli ID del registro restano univoci.','',
            '## Artefatti e utilizzo','',
            '[Figura PNG](experiments/alphagenome_v2_2026-10-18.png) · '
            '[PDF](experiments/alphagenome_v2_2026-10-18.pdf) · '
            '[Metriche JSON](experiments/alphagenome_v2_2026-10-18.json) · '
            '[Tutte le CpG held-out, CSV.gz](experiments/alphagenome_v2_2026-10-18_errors.csv.gz) · '
            '[Peggiori 100, CSV](experiments/alphagenome_v2_2026-10-18_worst100.csv) · '
            '[Maggiori peggioramenti rispetto alla v1, CSV](experiments/alphagenome_v2_2026-10-18_regressions100.csv)','',
            f"Checkpoint primario: `{display_out}/official/seed17/checkpoint.pt`. "
            'La directory contiene anche repliche, freeze, normalizzazione, grafo, prior e audit completi delle tre viste.','',
            '```python',
            'from methylation_predictor.modeling.alphagenome import AlphaGenomePredictorFromCheckpoint',
            f"predictor = AlphaGenomePredictorFromCheckpoint('{display_out}/official/seed17/checkpoint.pt', device='cuda')",
            'beta = predictor(raw_rna, gene_ids, track_indices, offsets, functional_dense, alphagenome)',
            '```','',
            'RNA nelle unità canoniche di X, ordine dei geni esplicito ed esatto; annotazioni 23D nella rappresentazione '
            'dello store congelato prima della standardizzazione TSS aggiuntiva. Annotazioni e AlphaGenome devono '
            'condividere lo stesso ordine CpG, con lookup per ID. Nessun accesso ai target durante inferenza. '
            'Il checkpoint è ricaricato e confrontato con input reali nel controllo di inferenza.','',
            'Verifiche: suite disponibile eseguita con esito positivo; otto test specifici su calibrazione, '
            'invarianza al batching, separazione dei gradienti, regolarizzazione delle risposte, ricaricamento '
            'del checkpoint e parità del controllo con la v1. Gli assi di fit e l’ordine dei 25.017 geni della '
            'v1 sono stati verificati prima del confronto. I conteggi delle coppie osservate sono confrontati '
            'con il protocollo canonico per tutte le viste.','',
            'Limiti: Array chr1, tre seed; holdout ufficiale già usato per la v1, quindi confronto successivo al precedente '
            'sviluppo e non nuova conferma esterna. Nessun tuning dopo il freeze in risposta alle metriche ufficiali. '
            'Queste misure non stabiliscono causalità biologica dei singoli errori. La nuova factory ha un contratto '
            'AlphaGenome esplicito; il trainer funzionale storico non viene usato per leggere automaticamente questi input.','']
    (ROOT/'docs/TCGA_CHR1_ALPHAGENOME_V2_2026-10-18.md').write_text('\n'.join(lines))


if __name__=='__main__': main()
