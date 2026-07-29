# Genomic encoder selection

- Backbone: NTv3-650M-post
- State: frozen
- Reference: hg38
- Context: 32,768 bp
- Orientation: forward
- Readout: mean final per-base embedding at central C/G
- Local pooling: rejected
- Reference neighborhood: optional, not core encoder
- Fine-tuning on chr1: rejected
- Outputs: locus embedding, mean methylation prior, total variability, between-cancer variability, within-cancer variability, locus projection for future RNA interaction
