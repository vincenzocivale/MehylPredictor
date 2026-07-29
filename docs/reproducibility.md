# Riproducibilità e migrazione

L'upstream è il submodule `third_party/MethylProphet` al commit `b24f5af3c7b4d6aa2689950e2ea4e3b2bcc8ddfd`; il suo worktree deve essere pulito. Il workaround storico al progress bar non modifica più upstream: ogni compatibilità passa da `diagnostics/methylprophet/upstream.py`.

| Percorso precedente | Destinazione | Decisione |
|---|---|---|
| `audit/locus_dominance.py` e analisi gerarchiche | `src/.../diagnostics/methylprophet/` | mantenere |
| `audit/empirical_prior_hybrid.py` | `.../empirical_hybrid.py` | mantenere |
| `audit/run_gene_encoder_intervention.py` e swap | `.../gene_intervention.py` e helper | mantenere |
| static prior, neighborhood, NTv3 e variabilità | `src/.../genomic_encoder/` | mantenere |
| `outputs/locus_dominance`, `outputs/empirical_prior_hybrid` | `artifacts/diagnostics/methylprophet/` | canonico |
| static/NTv3/neighborhood/variability output | `artifacts/genomic_encoder/` | canonico |
| matrici, checkpoint, shard, FASTA, embedding | `artifacts/cache/` | cache ignorata |
| smoke, retry, log e runner superseded | `artifacts/archive/` o rimosso | non operativo |

I risultati del paper, log autori e predizioni rilasciate restano fonti separate. I test unitari usano fixture sintetiche; i test completi devono essere marcati `gpu`, `data` o `regression`.
