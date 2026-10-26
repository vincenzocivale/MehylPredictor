"""Report matched-support validation without selecting on external results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_cpg_errors import clean_json

OUT = ROOT / 'artifacts/alphagenome_matched_support_2026-10-18'
STEM = ROOT / 'docs/experiments/alphagenome_matched_support_2026-10-18'
DOC = ROOT / 'docs/ALPHAGENOME_MATCHED_SUPPORT_2026-10-18.md'


def dump(path, value):
    path.write_text(json.dumps(clean_json(value), indent=2) + '\n')


def table(frame, columns):
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for _, row in frame.iterrows():
        vals = []
        for key in columns:
            value = row[key]
            vals.append('—' if pd.isna(value) else f'{value:.6f}' if isinstance(value, (float, np.floating)) else str(value))
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


def average_per(out, view, method, shot):
    paths = sorted((out / view).glob(f'{method}_seed*_rep*_k{shot}.npz'))
    assert len(paths) == 15, (view, method, shot, len(paths))
    stats = []
    for path in paths:
        with np.load(path) as z:
            stats.append({key: z[key] for key in ['cpg_idx', 'n', 'mse', 'mae', 'bias2', 'centered_mse']})
    ids = stats[0]['cpg_idx']
    assert all(np.array_equal(ids, x['cpg_idx']) for x in stats)
    n = np.sum([x['n'] for x in stats], axis=0)
    result = dict(cpg_idx=ids, n=n)
    for key in ['mse', 'mae', 'bias2', 'centered_mse']:
        result[key] = np.sum([x[key] * x['n'] for x in stats], axis=0) / np.maximum(n, 1)
    return result


def contrasts(out, summary):
    # Genomic blocks, conditional on the observed patients/support sets. These
    # are not joint patient-genome confidence intervals or independent fits.
    canon = ROOT.parent / 'data/mhc_experiments_root/datasets/methylprophet_repro_v1'
    registry = pd.read_parquet(canon / 'cpg/registries/array_cpg_map.parquet').set_index('cpg_idx')
    comparisons, error_frames = [], []
    for view in ['dev', 'epic', 'wgbs']:
        shots = [0, 1, 5, 10] + ([50] if view != 'wgbs' else [])
        for k in shots:
            candidate = average_per(out, view, 'v3_logit', k)
            blocks = registry.loc[candidate['cpg_idx'], 'pos'].to_numpy() // 5_000_000
            for method in ['v1_prior_logit', 'v1_prior_residual', 'v3_residual', 'support_mean', 'support_shrunk_mean', 'v3_population']:
                base = average_per(out, view, method, k)
                np.testing.assert_array_equal(candidate['n'], base['n'])
                n = base['n']; delta = candidate['mse'] - base['mse']
                block_ids = np.unique(blocks)
                numer = np.array([np.sum(delta[blocks == b] * n[blocks == b]) for b in block_ids])
                denom = np.array([n[blocks == b].sum() for b in block_ids])
                rng = np.random.default_rng(20261018)
                draws = rng.integers(0, len(block_ids), (2000, len(block_ids)))
                values = numer[draws].sum(1) / denom[draws].sum(1)
                mse = np.average(candidate['mse'], weights=n); bmse = np.average(base['mse'], weights=n)
                comparisons.append(dict(view=view, support=k, candidate='v3_logit', baseline=method,
                    mse=mse, baseline_mse=bmse, mse_delta=mse-bmse, mse_delta_percent=100*(mse/bmse-1),
                    mae_delta=np.average(candidate['mae']-base['mae'], weights=n),
                    bias2_delta=np.average(candidate['bias2']-base['bias2'], weights=n),
                    centered_mse_delta=np.average(candidate['centered_mse']-base['centered_mse'], weights=n),
                    fraction_cpg_improved=float(np.mean(delta < 0)),
                    ci95_genomic_blocks_low=np.quantile(values, .025), ci95_genomic_blocks_high=np.quantile(values, .975),
                    uncertainty='5Mb genomic block bootstrap conditional on patients and repeated support sets'))
                if k in [5, 10] and method == 'v1_prior_logit':
                    error_frames.append(pd.DataFrame(dict(view=view, support=k, cpg_idx=candidate['cpg_idx'],
                        v3_mse=candidate['mse'], v1_mse=base['mse'], delta_mse=delta,
                        v3_bias2=candidate['bias2'], v3_centered_mse=candidate['centered_mse'],
                        v3_mae=candidate['mae'], effective_pair_repetitions=n)))
    frame = pd.DataFrame(comparisons)
    frame.to_csv(str(STEM) + '_contrasts.csv', index=False)
    errors = pd.concat(error_frames)
    errors = errors.merge(registry.reset_index(), on='cpg_idx', validate='many_to_one')
    errors['v3_rmse'] = np.sqrt(errors.v3_mse)
    manifest = ROOT / 'artifacts/alphagenome_hypotheses_2026-10-18/external/HM450.hg38.manifest.tsv.gz'
    if manifest.exists():
        anno = pd.read_csv(manifest, sep='\t', usecols=['probeID', 'CpG_chrm', 'CpG_beg', 'gene_HGNC'])
        anno = anno.rename(columns={'probeID': 'illumina_probe_ids', 'CpG_chrm': 'chr'})
        anno['pos'] = anno.CpG_beg + 1
        anno = anno[anno.chr.isin(errors.chr.unique()) & anno.pos.isin(errors.pos.unique())]
        def joined(values):
            return ';'.join(sorted(set(values.dropna().astype(str))))
        anno = anno.groupby(['chr', 'pos'], as_index=False).agg(illumina_probe_ids=('illumina_probe_ids', joined), gene_HGNC=('gene_HGNC', joined))
        errors = errors.merge(anno, on=['chr', 'pos'], how='left', validate='many_to_one')
    previous = ROOT / 'artifacts/alphagenome_hypotheses_2026-10-18/hard_cpgs/annotated_errors.csv.gz'
    if previous.exists():
        hard = pd.read_csv(previous, usecols=['cpg_idx', 'hard']).rename(columns={'hard': 'previously_hard_array'})
        errors = errors.merge(hard, on='cpg_idx', how='left', validate='many_to_one')
    errors.to_csv(str(STEM) + '_cpg_errors.csv.gz', index=False)
    worst = pd.concat([x.nlargest(50, 'v3_mse') for _, x in errors.groupby(['view', 'support'])])
    worst.to_csv(str(STEM) + '_worst_cpgs.csv', index=False)
    return frame, worst


def figure(summary, feature, head):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    labels = dict(v3_logit='v3 intercept adaptation', v1_prior_logit='v1 + prior + intercept',
                  v3_residual='v3 additive residual', support_mean='Support mean',
                  support_shrunk_mean='Shrunk support mean', v3_population='v3 population, no query RNA')
    for ax, view in zip(axes[0], ['dev', 'epic', 'wgbs']):
        for method, label in labels.items():
            rows = summary[(summary.view == view) & (summary.method == method)].sort_values('support')
            ax.plot(rows.support, rows.mse, marker='o', markersize=3, label=label)
        ax.set_title(view.upper()); ax.set_xlabel('Support patients per CpG'); ax.set_ylabel('MSE')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    ax = axes[1, 0]
    for arm in feature.arm.unique():
        rows = feature[(feature.arm == arm) & (feature.calibration == 'raw')].sort_values('support')
        ax.plot(rows.support, rows.mse, marker='o', markersize=3, label=arm)
    ax.set_title('Matched feature pilot (one seed)'); ax.set_xlabel('Support patients'); ax.set_ylabel('MSE')
    ax.legend(fontsize=7); ax.grid(alpha=.2)
    ax = axes[1, 1]
    rows = head[(head.view == 'novel') & (head.blend == 0)].set_index('arm').loc[['baseline', 'mean', 'local01', 'local10']]
    ax.bar(rows.index, rows.mse); ax.set_title('Mean head and graph (internal)'); ax.set_ylabel('MSE')
    ax = axes[1, 2]
    for view in ['dev', 'epic', 'wgbs']:
        rows = summary[(summary.view == view) & (summary.method == 'v3_logit')].sort_values('support')
        base = summary[(summary.view == view) & (summary.method == 'v1_prior_logit')].sort_values('support')
        ax.plot(rows.support, 100*(rows.mse.to_numpy()/base.mse.to_numpy()-1), marker='o', label=view)
    ax.axhline(0, color='black', linewidth=.7); ax.set_xlabel('Support patients'); ax.set_ylabel('v3 vs calibrated v1: MSE change (%)')
    ax.set_title('Architecture gain at equal support'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(str(STEM) + '.png', dpi=180); fig.savefig(str(STEM) + '.pdf'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args(); out = args.out
    views = ['fold0', 'fold1', 'fold2', 'dev', 'epic', 'wgbs']
    assert all((out / view / 'complete.json').exists() for view in views)
    assert (out / 'feature_attribution/complete.json').exists() and (out / 'head_graph_attribution/complete.json').exists()
    frame = pd.concat([pd.read_csv(out / view / 'metrics.csv') for view in views])
    frame.to_csv(str(STEM) + '_all_metrics.csv', index=False)
    metric_cols = ['mse', 'mae', 'mas_pcc', 'within_tissue', 'severe', 'bias_fraction']
    summary = frame[frame.view.isin(['dev', 'epic', 'wgbs'])].groupby(['view', 'support', 'method'], as_index=False)[metric_cols].mean()
    summary.to_csv(str(STEM) + '_summary.csv', index=False)
    feature = pd.read_csv(out / 'feature_attribution/metrics.csv').groupby(['arm', 'calibration', 'support'], as_index=False)[metric_cols].mean()
    head = pd.read_csv(out / 'head_graph_attribution/metrics.csv').groupby(['arm', 'view', 'blend'], as_index=False)[metric_cols].mean()
    feature.to_csv(str(STEM) + '_features.csv', index=False); head.to_csv(str(STEM) + '_head_graph.csv', index=False)
    contrast, worst = contrasts(out, summary)
    crossed = pd.read_csv(out / 'crossed_uncertainty.csv')
    crossed.to_csv(str(STEM) + '_crossed_uncertainty.csv', index=False)
    selection = json.loads((out / 'frozen_selection.json').read_text())
    dump(Path(str(STEM) + '.json'), dict(selection=selection, summary=summary.to_dict('records'),
        contrasts=contrast.to_dict('records'), crossed_uncertainty=crossed.to_dict('records'),
        features=feature.to_dict('records'), head_graph=head.to_dict('records')))
    figure(summary, feature, head)
    shown = summary[summary.method.isin(['v3_logit', 'v1_prior_logit', 'v1_prior_residual', 'v3_residual',
                                        'support_mean', 'support_shrunk_mean', 'v3_population']) & summary.support.isin([0, 5, 10, 50])]
    text = '''# Controlli a parità di supporto e attribuzione del metodo — 18 ottobre 2026

Il beneficio dell’adattamento con pochi campioni si conferma rispetto alle medie senza RNA. La superiorità della nuova architettura rispetto a una v1 ugualmente calibrata non si conferma sulle due coorti esterne. Il confronto con il metodo pubblicato a parità di training resta aperto; questi risultati sostengono un lavoro sull’adattamento e sui limiti di trasferimento, ma non bastano per dichiarare v3 metodologicamente superiore.

## Protocollo

Encoder congelati. Ogni metodo riceve gli stessi 0/1/5/10/50 pazienti di supporto e gli stessi target finiti; per WGBS il massimo è 10. Tre seed per i modelli interni e v2/v3 ufficiali; cinque insiemi di supporto annidati. L’unica v1 ufficiale è seed17: la sua ripetizione nei confronti appaiati non equivale a tre fit indipendenti.

La forza di shrinkage di ciascun metodo è selezionata esclusivamente sui tre fold genomici interni, fra 0, 5 e 20, con lo stesso budget di ricerca. Scelte congelate prima di valutare dev ed esterni. Questi dataset erano già stati esaminati in precedenza: è una conferma di follow-up, non un test intatto.

Nel dev, supporto dai pazienti training e test sugli 825 pazienti dev. In EPIC, 50 pazienti riservati al supporto e 1.656 al test per ripetizione. In WGBS, 10 al supporto e 21 al test, dopo aggregazione delle repliche tecniche per paziente. Nessun paziente esterno compare nel training Array. Gli stessi pazienti test sono mantenuti per tutti i valori di k nella stessa ripetizione; le ripetizioni possono sovrapporsi.

`support_mean`: media delle beta osservate, senza RNA; senza osservazioni finite usa il prior genomico. `support_shrunk_mean`: media regolarizzata verso il prior genomico. `support_tissue_mean` usa anche il tessuto osservato; gruppi ignoti ricadono sulla media globale. `v3_population` aggiorna la media predetta sui riferimenti RNA training, senza RNA del paziente query. `residual` corregge il residuo medio nello spazio beta e limita a [0,1]; `logit` adatta l’intercetta prima della sigmoid. v1 e v3 ricevono lo stesso prior genomico training-only e gli stessi pazienti di riferimento, codificati attraverso il proprio encoder.

## Scelte interne congelate

'''
    text += table(pd.DataFrame([dict(method=k, tau=v) for k, v in selection['selected'].items()]), ['method', 'tau'])
    text += '\n\n## Risultati a parità di dati\n\n' + table(shown, ['view', 'support', 'method', 'mse', 'mae', 'mas_pcc'])
    text += '\n\n## Quanto aggiunge v3 alla v1 già calibrata?\n\n'
    text += table(contrast[(contrast.baseline == 'v1_prior_logit') & contrast.support.isin([0, 5, 10, 50])],
                  ['view', 'support', 'mse_delta_percent', 'mae_delta', 'fraction_cpg_improved', 'ci95_genomic_blocks_low', 'ci95_genomic_blocks_high'])
    text += '\n\nGli intervalli ricampionano blocchi genomici da 5 Mb, condizionatamente ai pazienti e agli insiemi di supporto osservati. Non sono intervalli congiunti pazienti-genoma. In particolare, 21 pazienti WGBS per test restano una limitazione sostanziale. Seed e ripetizioni di supporto non sono trattati come pazienti indipendenti.\n'
    text += '\n## Incertezza congiunta sui pazienti esterni e sui blocchi genomici\n\nRicampionamento incrociato con 2.000 estrazioni: ogni paziente riceve lo stesso peso in tutte le ripetizioni in cui compare come query, evitando di contarle come osservazioni indipendenti. Le loss sono mediate sui seed, senza costruire un ensemble. Il supporto e gli adattamenti già stimati restano fissi: questi intervalli non includono tutta l’incertezza derivante da un nuovo campione di supporto. Le loss sono state ricomputate indipendentemente e verificate contro i risultati per CpG.\n\n'
    text += table(crossed[crossed.baseline == 'v1_prior_logit'],
                  ['view', 'support', 'mse_delta_percent', 'ci95_percent_low', 'ci95_percent_high'])
    text += '\n## Attribuzione degli input\n\nConfronto originale appaiato, stesso training, budget 1.000 step, seed17. Le prime tre condizioni condividono le loss; `hybrid_wt` aggiunge la loss entro tessuto. Nessuna conclusione multi-seed sugli input. Aggiungere `AG_prior025` introduce informazioni AlphaGenome anche nel modello funzionale.\n\n'
    text += table(feature[(feature.support.isin([0, 5, 50])) & (feature.calibration == 'raw')], ['arm', 'support', 'mse', 'mae', 'mas_pcc'])
    text += '\n\n## Attribuzione della testa e della regolarizzazione\n\nTre fold × tre seed, assi di training verificati uguali nei checkpoint. `mean − baseline` isola la testa della media superficiale; `local01 − mean` aggiunge la penalità sui vicini di peso 0,01; `local10` ne aumenta il peso. Le misure sono descrittive su fold già usati nello sviluppo.\n\n'
    text += table(head[head.view == 'novel'], ['arm', 'blend', 'mse', 'mae', 'mas_pcc', 'within_tissue'])
    text += '''

## Confronto con MethylProphet

L’inventario locale trova il checkpoint originale ENCODE (24.337 input RNA), mentre il nostro asse canonico ne contiene 25.017. Il modello TCGA elencato nel MODEL_ZOO è allenato su Array+EPIC+WGBS: usarlo come baseline Array-only o come test esterno indipendente su queste coorti sarebbe scorretto. Non è stata prodotta una falsa comparazione numerica. Per una dichiarazione di superiorità rispetto al metodo pubblicato serve ancora un refit MethylProphet Array-only, con mapping RNA/tokenizzazione nativi verificati e identici assi di training, oltre alla stessa calibrazione sui campioni di supporto. Questa campagna non chiude quel confronto.

## Errori CpG e artefatti

Il CSV `*_cpg_errors.csv.gz` contiene per ogni locus, a k=5 e k=10, MSE/MAE v3, MSE v1, differenza, bias² ed errore centrato, mediati sulle predizioni dei seed e sulle ripetizioni (non un ensemble). `*_worst_cpgs.csv` riporta i 50 loci con maggiore MSE per vista e supporto. Le coordinate e gli identificatori provengono dal registro canonico.

[Figura](experiments/alphagenome_matched_support_2026-10-18.png) · [PDF](experiments/alphagenome_matched_support_2026-10-18.pdf) · [Riepilogo](experiments/alphagenome_matched_support_2026-10-18_summary.csv) · [Contrasti](experiments/alphagenome_matched_support_2026-10-18_contrasts.csv) · [Errori CpG](experiments/alphagenome_matched_support_2026-10-18_cpg_errors.csv.gz) · [CpG peggiori](experiments/alphagenome_matched_support_2026-10-18_worst_cpgs.csv).

Riproduzione: `python scripts/validate_alphagenome_matched_support.py`, `python scripts/validate_alphagenome_attribution.py`, `python scripts/validate_alphagenome_support_uncertainty.py`, quindi `python scripts/report_alphagenome_matched_support.py` con matplotlib disponibile. Protocolli, selezione interna, assi supporto/test, hash dei checkpoint e risultati per CpG in `artifacts/alphagenome_matched_support_2026-10-18/`.
'''
    text += '\n\nTre esempi degli errori maggiori per vista con 10 pazienti di supporto; RMSE e MAE sono in unità beta. I nomi Illumina e i geni sono annotazioni descrittive del manifest, senza attribuzione causale.\n\n'
    examples = pd.concat([x.nlargest(3, 'v3_mse') for _, x in worst[worst.support == 10].groupby('view')])
    columns = [c for c in ['view', 'illumina_probe_ids', 'chr', 'pos', 'gene_HGNC', 'v3_rmse', 'v3_mae', 'delta_mse'] if c in examples]
    text += table(examples, columns) + '\n'
    DOC.write_text(text)
    dump(out / 'report_complete.json', dict(complete=True, report=str(DOC), evaluation_records=len(frame),
        matched_external_published_method_comparison_complete=False))
    print(json.dumps(dict(report=str(DOC), records=len(frame), summary=summary[summary.support == 5].to_dict('records'))))


if __name__ == '__main__':
    main()
