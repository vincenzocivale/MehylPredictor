"""Append the frozen hierarchical refit audit; never selects a new recipe."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/alphagenome_hierarchical_2026-10-18'
PREFIX=ROOT/'docs/experiments/alphagenome_hierarchical_2026-10-18_official'


def main():
    done=json.loads((OUT/'refit_complete.json').read_text())
    seeds={s:json.loads((OUT/'official'/f'seed{s}'/'metrics.json').read_text()) for s in [17,18,19]}
    old=ROOT/'artifacts/alphagenome_v2_2026-10-18/official'
    v1=json.loads((old/'baseline_v1/metrics.json').read_text())
    v2=json.loads((old/'seed17/metrics.json').read_text())
    frame=pd.read_csv(OUT/'official/seed17/double_ood_vs_v1.csv.gz')
    comparison=pd.read_csv(OUT/'official/seed17/double_ood_vs_v2.csv.gz')
    assert np.array_equal(frame.cpg_idx,comparison.cpg_idx)
    for col in comparison.columns:
        if col.startswith('v2_'): frame[col]=comparison[col]
    frame['delta_mse_vs_v2']=frame.mse-frame.v2_mse
    mapping=ROOT.parent/'CpGRepresentationBenchmark/data/processed/ComputAgeBench/illumina_probe_grch38.parquet'
    probes=pd.read_parquet(mapping)
    probes=probes[probes['chr'].isin(['chr1','1'])].groupby('pos').probe_id.agg(lambda x:';'.join(sorted(set(x))))
    frame=frame.merge(probes.rename('illumina_probe_ids'),left_on='pos_1based',right_index=True,how='left',validate='many_to_one')
    arrays=[]; worst=[]
    for seed in [17,18,19]:
        f=pd.read_csv(OUT/'official'/f'seed{seed}'/'double_ood_per_cpg.csv.gz')
        assert np.array_equal(frame.cpg_idx,f.cpg_idx)
        arrays.append(f.mse.to_numpy()); worst.append(f.mse>=f.mse.quantile(.9))
    frame['mse_mean_seeds']=np.mean(arrays,axis=0)
    frame['mse_min_seeds']=np.min(arrays,axis=0); frame['mse_max_seeds']=np.max(arrays,axis=0)
    frame['worst_decile_seed_count']=np.sum(worst,axis=0)
    frame['bias_fraction']=frame.bias2/frame.mse.clip(lower=1e-12)
    assert len(frame)==6742 and frame.cpg_idx.is_unique
    assert np.allclose(frame.delta_mse,frame.delta_bias2+frame.delta_centered_mse,atol=1e-10)
    frame.sort_values('mse',ascending=False).to_csv(str(PREFIX)+'_errors.csv.gz',index=False)
    frame.sort_values('mse',ascending=False).head(100).to_csv(str(PREFIX)+'_worst100.csv',index=False)
    frame.sort_values('delta_mse',ascending=False).head(100).to_csv(str(PREFIX)+'_regressions100.csv',index=False)
    data=dict(recipe=done['recipe'],v1=v1,v2=v2,seeds=seeds,mapped_probes=int(frame.illumina_probe_ids.notna().sum()),
              persistent_worst_decile=int((frame.worst_decile_seed_count>=2).sum()))
    Path(str(PREFIX)+'.json').write_text(json.dumps(data,indent=2)+'\n')
    lines=['## Architettura selezionata','',
           'Il corpo è la v2 local01 a 1.000 step: encoder RNA condiviso, rappresentazione CpG AlphaGenome+annotazioni, intercetta shallow e risposta profonda a rango 32. Il recupero dei vicini e il congelamento aggiuntivo non sono attivi. La penalità locale della v2 resta una componente del training.', '',
           '```mermaid',
           'flowchart LR',
           '  RNA[RNA paziente] --> U[Encoder RNA e 32 fattori]',
           '  Features[AlphaGenome e annotazioni CpG] --> C[Encoder CpG: intercetta e 32 fattori]',
           '  Features --> P[Prior genomico della media]',
           '  C --> B[Calibrazione intercetta]',
           '  P --> B',
           '  K[Offset appresi sulle CpG training] --> B',
           '  ID[ID globale CpG: nota o nuova] --> B',
           '  U --> Y[Prodotto fattori + intercetta e sigmoid]',
           '  C --> Y',
           '  B --> Y',
           '```', '',
           'Per una CpG nuova, la media sulla popolazione RNA di riferimento è combinata al 75% con quella del modello e al 25% con il prior Ridge. Per una CpG nota, un offset stimato sui soli valori osservati nei pazienti training calibra il livello medio mantenendo i fattori. Gli ID servono a recuperare parametri appresi: non sono una scelta basata sugli errori di valutazione. La calibrazione si può precalcolare con i fattori CpG; il costo ricorrente mantiene un solo encoder RNA.', '',
           '## Refit e confronto ufficiale a ricetta congelata','',
           'Tre refit su tutti gli assi training ufficiali. Il confronto primario usa seed17 e le stesse osservazioni di v1/v2. I seed18/19 misurano variabilità del candidato; il riferimento ufficiale v1 ha un solo seed.', '',
           '| Vista | Modello | MSE | MAE | PCC | Within-tissue | Errori >0,2 |',
           '|---|---|---:|---:|---:|---:|---:|']
    for view,name in [('new_patients','CpG note / nuovi pazienti'),('new_cpgs','CpG nuove / pazienti noti'),('double_ood','CpG nuove / nuovi pazienti')]:
        for model,doc in [('v1',v1),('v2',v2),('gerarchico',seeds[17])]:
            m=doc['views'][view]['metrics']
            lines.append(f"| {name} | {model} | {m['mse']:.6f} | {m['mae']:.5f} | {m['mas_pcc']:.5f} | {m['within_tissue']:.5f} | {100*m['severe']:.2f}% |")
    q=seeds[17]['views']['double_ood']['metrics']; b=v1['views']['double_ood']['metrics']
    k=seeds[17]['views']['new_patients']['metrics']; kb=v1['views']['new_patients']['metrics']
    lines += ['',f"Nel confronto primario la MSE migliora del {100*(1-q['mse']/b['mse']):.2f}% sulle nuove CpG e del {100*(1-k['mse']/kb['mse']):.2f}% sulle CpG note. La MAE double-OOD cambia da {b['mae']:.5f} a {q['mae']:.5f}, la within-tissue da {b['within_tissue']:.5f} a {q['within_tissue']:.5f} e gli errori >0,2 da {100*b['severe']:.2f}% a {100*q['severe']:.2f}%: **il candidato non domina v1 su tutte le metriche ufficiali**. I criteri superati internamente e sul dev non si trasferiscono integralmente al confronto ufficiale. Nessun peso è stato ritoccato per correggere questo risultato."]
    lines+=['', '| Seed gerarchico | MSE double-OOD | MSE CpG note | PCC double-OOD |', '|---|---:|---:|---:|']
    for seed,doc in seeds.items():
        q=doc['views']['double_ood']['metrics']; k=doc['views']['new_patients']['metrics']
        lines.append(f"| {seed} | {q['mse']:.6f} | {k['mse']:.6f} | {q['mas_pcc']:.5f} |")
    lines+=['', f"Migliora in MSE il {100*(frame.delta_mse<0).mean():.1f}% delle CpG rispetto a v1 e il {100*(frame.delta_mse_vs_v2<0).mean():.1f}% rispetto a v2. Persistono {data['persistent_worst_decile']} CpG nel decile peggiore in almeno due seed su tre.", '',
            '| CpG | Coordinate hg38 | RMSE beta | Bias | PCC |','|---|---|---:|---:|---:|']
    for _,r in frame.sort_values('mse',ascending=False).head(10).iterrows():
        lines.append(f"| {r.illumina_probe_ids} | {r.coordinate} | {r.rmse:.3f} | {r.bias:+.3f} | {r['corr']:.3f} |")
    lines+=['', '[Errori ufficiali completi](experiments/alphagenome_hierarchical_2026-10-18_official_errors.csv.gz) · [Peggiori 100](experiments/alphagenome_hierarchical_2026-10-18_official_worst100.csv) · [Maggiori regressioni](experiments/alphagenome_hierarchical_2026-10-18_official_regressions100.csv) · [Metriche](experiments/alphagenome_hierarchical_2026-10-18_official.json)', '',
            'Checkpoint primario: `artifacts/alphagenome_hierarchical_2026-10-18/official/seed17/checkpoint.pt`. Inferenza con `AlphaGenomePredictorFromCheckpoint`, passando anche `cpg_ids=` con gli ID globali del registro, allineati alle feature. Gli ID non sono le posizioni delle righe né le stringhe Illumina cg.', '',
            'Il livello medio delle CpG note è appreso esclusivamente dai pazienti training. Le CpG nuove usano solo il prior genomico e il predittore. Fattori e intercette si possono precalcolare; il decoder mantiene un solo encoder RNA e rango 32. Nessuna scelta è stata cambiata dopo i risultati ufficiali. L’holdout era già stato esaminato nello sviluppo v1/v2: non è una nuova conferma esterna.', '']
    report=ROOT/'docs/ALPHAGENOME_HIERARCHICAL_2026-10-18.md'
    text=report.read_text().split('## Architettura selezionata')[0].split('## Refit e confronto ufficiale a ricetta congelata')[0]
    report.write_text(text+'\n'+'\n'.join(lines))
    print(json.dumps(dict(complete=True,views=seeds[17]['views'],persistent_worst=data['persistent_worst_decile'])))


if __name__=='__main__': main()
