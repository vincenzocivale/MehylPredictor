## Shared genomic feature store

Frozen NTv3 locus embeddings used by the RNA-to-DNAm experiments are
precomputed once and stored independently of individual training runs.

Default location:

/raid/DATASETS/MethylPredictionData/genomic_features/
  ntv3_650m_post/
    hg38_L32768_forward_cpg_center/

Embedding protocol:

- model: InstaDeepAI/NTv3_650M_post
- genome: hg38
- sequence length: 32,768 bp
- orientation: forward
- locus representation: mean of the central C/G bins
- inference dtype: bfloat16
- storage dtype: float16
- extraction: sharded, resumable and optionally multi-GPU

The production extractor uses a character-level A/C/G/T/N lookup table
instead of invoking the Hugging Face tokenizer for every batch. The lookup
table is verified against the loaded tokenizer before extraction. A
validation benchmark showed bit-identical central-CpG embeddings
(max absolute difference 0) and approximately 1.33x higher throughput.

The NTv3 embedding store is independent of RNA model architecture,
training seed and loss configuration and is therefore reused by E2, E3,
E4 and subsequent experiments.

Predicted methylation prior and variability features are stored separately
under:

/raid/DATASETS/MethylPredictionData/genomic_features/methylation_prior/

because these features additionally depend on the methylation training
protocol used to fit the genomic probes.

Experiment-specific checkpoints, metrics and logs are stored under:

/raid/DATASETS/MethylPredictionData/experiments/
