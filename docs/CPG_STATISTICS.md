# CpG statistics predictor

`CpGStatisticsPredictor` is the only trainable static-locus auxiliary model.
It jointly exposes mean and scale while keeping their neural parameter sets
independent.

The target builder uses all requested technologies and stores source-specific
observation counts.  This prevents a weighting rule from becoming an implicit
implementation detail.  The default `sample_weighted` target treats every
finite methylation observation equally; `technology_balanced` gives each
technology with data at a locus equal mixture weight.

`mu` is a beta-space mean. `sigma` is a standard deviation in clipped
`logit(beta)` space, matching the residual scale consumed by
`RNAMethylationPredictor`.

The model is selected only on a genomic-block inner split of official train
CpGs. Official held-out CpGs are final evaluation labels only.  For the RNA
feature cache, empirical training-locus statistics may be inserted after model
training while held-out loci always receive predictor outputs.
