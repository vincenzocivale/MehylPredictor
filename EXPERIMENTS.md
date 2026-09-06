# Risultati principali
- nostro modello (cross attention RNA) addestrato su TCHA chr1 e chr123 e ENCODE con split ufficiali per confrontarsi direttamente con methylProphet
- performance methylProphet stessi split per riprendere dati tabella
- performance Luyg... tabella paper MethylProphet
- baseline semplici
- performance patient training x cpg val con FM addestrati con masking (da paper MethylProphet)

# Contributo mean

il nostro modello porta come contributo metodologico è fare proxy di training con predizione media. Esperimenti che dimostrino perchè funziona (biologicamente) e mostrino differenze tra usarlo che training classico

Fare analisi su quale tipologia di CpG la media aiuta a performare meglio (CpG meno variaibli?)

# contrbuto cross attention

secondo contrbiuto è fare cross attention su dati RNA. Quindi far esperimenti per trovare configurazione ottimale (attualmente 64 token da embedding 256). Altra variante utilizzare come query embedding NTv3 ottimizzato per predizione della media invece che quello raw

In parallelo lanciare esperimenti di confronto con altri metodi usato per encoding RNA per confrontare con cross attention nostra (chr1 tcga e forse tcga chr123):
- MLP bottleneck
- BulkRNABert per encoding RNA
- gene-pathway encoder(usato da MethylProphet non so cosa signfichi)
- altro?

oltr esperimenti per performance compute classifici, esperimenti per cercare motivazione cross attentio funziona meglio

# ablation generalizzazione 

- training chr1 tcga test tcga chr 123 e viceversa (nostro modello cs MethylProphet)
- training TCGA (chr1 e chr123) e test ENCODE e vicercsa(per confronto con MethylProphet riusare loro checkpoint preaddestrati)
- training TCGA array test TCGA array
- training TCGA array + W  test TCGA array
- training TCGA array + E  test TCGA array
- training TCGA array + E + W  test TCGA array