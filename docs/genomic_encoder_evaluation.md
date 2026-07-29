# Valutazione del genomic encoder

Il ramo genomico valuta feature statiche, reference neighborhood e probe NTv3 sul prior e sulle componenti di variabilità. Gli embedding grandi restano cache non versionata.

La decisione finale è NTv3-650M-post frozen, hg38, 32.768 bp, orientamento forward e readout media dell'embedding finale alle basi C/G centrali. Il pooling locale non ha migliorato il readout; reference neighborhood è opzionale e non core; fine-tuning su chr1 non è giustificato. I limiti includono la sola esposizione chr1 e l'esposizione Borzoi post-training.

La configurazione selezionata ottiene MSE beta-space 0.008688 in validation e 0.009155 in test per il prior. Le sonde sulla variabilità mantengono target train-only e riportano separatamente varianza totale, between-cancer e within-cancer; le metriche di selezione e test restano nei manifest canonici.
