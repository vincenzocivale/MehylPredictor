# RNA branch — risultati consolidati

NTv3 fornisce un prior genomico migliore del prior statico del checkpoint
MethylProphet sui CpG locus-OOD. Il prior empirico same-locus train-only è una
baseline supervisionata diversa e non un vero prior locus-OOD.

L'RNA reale contiene segnale patient-specific oltre il cancer type. Il modello
vincente usa RNA completo standardizzato solo sul train, proiezione lineare
21.792→64, interazione bilineare e gate informato dalla variabilità. PCA,
random projection, MLP, bottleneck alternativi, interaction MLP, mixture
bilineari e loss/sampling per terzili non migliorano il double-OOD.

Il miglioramento più importante è il protocollo `lr=2e-5`, 20 epoche e
selezione per validation beta-MSE. L'ensemble C0 ottiene MSE 0,024090850,
contro 0,027192729 del checkpoint MethylProphet originale.

Il bootstrap gerarchico della differenza C0−MP include zero: non esiste un
claim statisticamente conclusivo di superiorità su chr1. Con prior NTv3
identico, la dinamica MP calibrata ottiene MSE 0,023357464 contro 0,024061835
della dinamica C0 calibrata. Il vantaggio del C0 completo deriva soprattutto
dal prior genomico; il ramo RNA MethylProphet resta attualmente più efficace.

Il prossimo obiettivo scientifico è migliorare il ramo RNA senza riaprire
indiscriminatamente la ricerca sull'encoder genomico.
