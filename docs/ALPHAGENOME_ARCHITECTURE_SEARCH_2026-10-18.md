# AlphaGenome: ricerca architetturale in due fasi e trasferimento locale

Esperimenti interni del 18 ottobre 2026. Non è una dimostrazione di ottimalità architetturale.

Guardrail interni superati: **False**. Candidato esaminato sul dev: **v2 · continuazione congiunta**.

## Disegno

- Tre fold genomici interni × tre seed; stesso split e normalizzazione dei precedenti esperimenti.
- Tre continuazioni da ciascun checkpoint EMA v2 a 1.000 step: congiunta, intercetta congelata, intero ramo CpG congelato. Altri 1.000 step, AdamW riavviato in tutti i bracci. Non equivale a un fit ininterrotto da 2.000 step.
- Congelare la media significa congelare tutta la funzione genomica che produce l’intercetta. La media beta sulla popolazione può ancora cambiare attraverso la risposta RNA e la sigmoid.
- Trasferimento: media dei fattori di cinque vicini training, blend 0 / 0,25 / 0,5; geometria AlphaGenome originale o fattori appresi. Standardizzazione/PCA sui soli loci training; distanza ≥1 Mb; appaiamento contesto/regione/quartile TSS, fallback globale se necessario.
- L’intercetta viene ricalibrata per preservare la media beta originale su 512 pazienti training di riferimento. Nessun target della CpG query è usato nel recupero o nella calibrazione.
- Controllo negativo: cinque vicini casuali con gli stessi vincoli, blend 0,5; escluso dalla selezione.
- Nuovi pazienti valutati su tutte le CpG nuove e su 4.096 CpG note prefissate per fold. Guardrail: MSE nuova almeno −0,5%; MSE/MAE note entro +1%; MAE nuova non peggiore; PCC/within-tissue entro −0,002; errori >0,2 entro +0,001 rispetto a v1.
- Se nessun candidato passa, nessuna promozione: si riporta separatamente il minimo rapporto MSE medio fra le due viste come compromesso esplorativo.

## Selezione interna

| Modello | MSE nuove | MSE note | PCC nuove | WT nuove | Ammissibile |
|---|---:|---:|---:|---:|---|
| v1 · 1.000 step | 0.027531 | 0.015462 | 0.50380 | 0.33817 | False |
| v2 · 1.000 step | 0.026583 | 0.015910 | 0.50436 | 0.33851 | False |
| v2 · 1.000 step + vicini raw 0.25 | 0.026605 | 0.016464 | 0.50458 | 0.33898 | False |
| v2 · 1.000 step + vicini raw 0.5 | 0.027046 | 0.017521 | 0.48882 | 0.33098 | False |
| v2 · 1.000 step + vicini learned 0.25 | 0.026585 | 0.016177 | 0.50436 | 0.33910 | False |
| v2 · 1.000 step + vicini learned 0.5 | 0.026762 | 0.016653 | 0.49809 | 0.33620 | False |
| v2 · continuazione congiunta | 0.027488 | 0.012990 | 0.49649 | 0.33917 | False |
| v2 · continuazione congiunta + vicini raw 0.25 | 0.027367 | 0.013697 | 0.50021 | 0.34153 | False |
| v2 · continuazione congiunta + vicini raw 0.5 | 0.027753 | 0.015077 | 0.48693 | 0.33435 | False |
| v2 · continuazione congiunta + vicini learned 0.25 | 0.027415 | 0.013325 | 0.49831 | 0.34111 | False |
| v2 · continuazione congiunta + vicini learned 0.5 | 0.027557 | 0.013938 | 0.49268 | 0.33879 | False |
| v2 · intercetta congelata | 0.027556 | 0.013043 | 0.49748 | 0.33985 | False |
| v2 · intercetta congelata + vicini raw 0.25 | 0.027437 | 0.013749 | 0.50115 | 0.34206 | False |
| v2 · intercetta congelata + vicini raw 0.5 | 0.027827 | 0.015133 | 0.48746 | 0.33488 | False |
| v2 · intercetta congelata + vicini learned 0.25 | 0.027484 | 0.013379 | 0.49906 | 0.34184 | False |
| v2 · intercetta congelata + vicini learned 0.5 | 0.027629 | 0.013998 | 0.49314 | 0.33945 | False |
| v2 · ramo CpG congelato | 0.026928 | 0.015715 | 0.50386 | 0.34124 | False |
| v2 · ramo CpG congelato + vicini raw 0.25 | 0.026897 | 0.016242 | 0.50520 | 0.34257 | False |
| v2 · ramo CpG congelato + vicini raw 0.5 | 0.027317 | 0.017314 | 0.49058 | 0.33490 | False |
| v2 · ramo CpG congelato + vicini learned 0.25 | 0.026892 | 0.015960 | 0.50479 | 0.34255 | False |
| v2 · ramo CpG congelato + vicini learned 0.5 | 0.027050 | 0.016437 | 0.49929 | 0.33986 | False |

## Conferma dev a scelta congelata

| Modello | MSE nuove | MSE note | MAE nuove | PCC nuove | WT nuove | Errori >0,2 |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 0.026422 | 0.017172 | 0.10630 | 0.52686 | 0.36252 | 17.47% |
| v2 | 0.025633 | 0.017861 | 0.10651 | 0.52724 | 0.36245 | 17.45% |
| candidate | 0.026668 | 0.014344 | 0.10823 | 0.52300 | 0.36297 | 17.93% |

## Controllo del trasferimento

Differenza di MSE sulle nuove CpG rispetto allo stesso checkpoint senza trasferimento, media dei nove confronti interni. Un segno negativo indica un miglioramento.

| Continuazione | Spazio dei vicini | ΔMSE con blend 0,5 |
|---|---|---:|
| v2_early | raw | +0.000463 |
| v2_early | learned | +0.000180 |
| v2_early | random | +0.000938 |
| joint | raw | +0.000265 |
| joint | learned | +0.000068 |
| joint | random | +0.000727 |
| freeze_mean | raw | +0.000270 |
| freeze_mean | learned | +0.000073 |
| freeze_mean | random | +0.000732 |
| freeze_cpg | raw | +0.000389 |
| freeze_cpg | learned | +0.000121 |
| freeze_cpg | random | +0.000866 |

## Errori per CpG

Audit dev su 3,972 nuove CpG, mediato sui tre seed. Migliora rispetto alla v1 il 48.0% dei loci; rispetto alla v2 il 40.6%.
ΔMSE vs v1 = +0.000246: bias² +0.000365, errore centrato -0.000119.

## Artefatti e limiti

[Figura](experiments/alphagenome_architecture_2026-10-18.png) · [Metriche](experiments/alphagenome_architecture_2026-10-18.json) · [Selezione CSV](experiments/alphagenome_architecture_2026-10-18_selection.csv) · [Errori CpG](experiments/alphagenome_architecture_2026-10-18_errors.csv.gz) · [Peggiori 100](experiments/alphagenome_architecture_2026-10-18_worst100.csv)

Array chr1; split interni e dev già usati nello sviluppo precedente, quindi nessuna nuova conferma esterna. Nessun target dell’holdout ufficiale è stato selezionato o valutato. Embargo 300 kb: non garantisce contesti AlphaGenome completamente disgiunti. Le medie sui seed non sono un ensemble. Il recupero esplicito richiede un banco di fattori training e una ricerca dei vicini: è un costo aggiuntivo rispetto all’encoder CpG cacheabile.

I checkpoint di continuazione sono portabili tramite AlphaGenomePredictorFromCheckpoint. Gli esperimenti di trasferimento sono implementati nella funzione transfer_response e nel valutatore; il checkpoint base da solo non incorpora il recupero dei vicini.

Riproduzione: `python scripts/search_alphagenome_architecture.py`; rendering: `MPLCONFIGDIR=/tmp/methylpredictor-matplotlib python scripts/report_alphagenome_architecture.py` con matplotlib disponibile.
