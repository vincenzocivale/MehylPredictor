# MethylPredictor model

The final paper-facing architecture is `EfficientSingleAttentionPredictor` with
8 retrieval FFN blocks, 8 functional FFN blocks, `deep_query=false`, and 2
post-fusion residual FFN blocks in the prediction head.

No NTv3/genomic-FM embedding is a model input. The inputs are normalized bulk
RNA plus patient-independent functional locus annotations.

## Functional branch

4,165 sparse regulatory-track indicators are encoded with `EmbeddingBag`; 23
dense functional annotations are independently projected to width 256. Their
sum is normalized to `h_c`, then refined by 8 residual FFN blocks
(256 -> 1024 -> 256), producing `h_c_deep`.

## RNA branch and retrieval

RNA is compressed to width 256 and mapped to 64 learned program tokens. A
single 4-head locus-to-RNA cross-attention uses the SHALLOW `h_c` as query.
The attention output is added residually, then the patient/locus state is
refined by 8 residual FFN blocks. Cross-attention is therefore evaluated only
once.

## Final prediction head

`LayerNorm(h_c_deep)` and `LayerNorm(state)` are concatenated to width 512.
The final head is:

```text
512
 -> Linear(512,256) + GELU + Dropout(0.15)
 -> FeedForwardResidual(256 -> 1024 -> 256, dropout=0.15)
 -> FeedForwardResidual(256 -> 1024 -> 256, dropout=0.15)
 -> Linear(256,128) + GELU + Dropout(0.15)
 -> Linear(128,1)
 -> sigmoid
```

The two added head FFNs contribute exactly 1,052,160 trainable parameters
(526,080 each), bringing the model from about 20.6M to about 21.65M parameters.

## Mean proxy

The training-only mean head reads `h_c_deep` and does not feed the final beta
predictor.

## Objective

`L = L_beta_MSE + 0.15 L_locus_PCC + 0.15 L_mean`.

## Final selector

```text
efficient_single_attn_8ffn_residual_functional8_head2
```

The historical J9b selector remains unchanged so its existing checkpoints stay
loadable during the transition.
