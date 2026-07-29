# Diagnosi di MethylProphet

La diagnosi usa `third_party/MethylProphet` come dipendenza read-only e le righe di predizione rilasciate come evidenza degli split effettivamente valutati. Le PCC delegano a `src.eval.compute_pcc_by_group` dell'upstream.

Le analisi coprono integrità delle predizioni, MSE/MAE per split, predominanza del locus, decomposizione locus/cancer/paziente, DynamicSkill e TotalSkill, interventi sul gene encoder, ibrido prior empirico+dynamica e baseline RNA medio. Le conclusioni riportate nei risultati esistenti sono: prior locus-specific forte, dinamica utile ma sovra-amplificata.

Per TCGA, le metriche ricalcolate dalle predizioni rilasciate coincidono con i `log_dict` degli autori: MSE/MAE rispettivamente 0.020902/0.089509, 0.027571/0.101290 e 0.028263/0.102559 per i tre split pubblicati. Non riproducono la tabella ICLR; questa discrepanza è mantenuta esplicitamente come differenza fra paper e artefatti rilasciati.

L'unico output pubblico per il ramo genomico è il contratto in `artifacts/diagnostics/methylprophet/export/`; il ramo genomico non legge shard, MDS, checkpoint o quantizzazione RNA upstream.
