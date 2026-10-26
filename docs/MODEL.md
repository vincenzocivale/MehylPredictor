# MethylPredictor model

The latest AlphaGenome mixed-source research model is documented in the
[research dossier](ALPHAGENOME_RESEARCH_STATE_2026-10-18.md) and
[inference runbook](ALPHAGENOME_REPRODUCIBILITY_2026-10-18.md). It uses native
1Mb AlphaGenome features and a hierarchical low-rank predictor. The functional
attention architecture described below is a separate workflow.

The historical functional paper-facing architecture is `EfficientSingleAttentionPredictor` with
8 retrieval FFN blocks, 8 functional FFN blocks, `deep_query=false`, and 2
post-fusion residual FFN blocks in the prediction head.

The final architecture takes no genomic-FM embedding: inputs are normalized bulk
RNA plus patient-independent functional locus annotations.

## Perceiver variants (`modeling/perceiver.py`, `modeling/cpg_query.py`)

`PerceiverCpGPredictor` replaces the RNA encoder with a Perceiver (gene patches
-> learned latents) and keeps the CpG-query cross-attention head. Selectors:

- `perceiver_cpg_query`: inherited functional query (4,165 tracks + 23 dense).
- `perceiver_functional_generic`: `GenericFunctionalQuery`, optionally masked to
  a track contract (default histone+DNase, 2,492 tracks).
- `perceiver_functional_latent`: `LatentQuery` over a precomputed, frozen,
  reference-sequence functional-model latent (default Enformer trunk, 3072-d)
  read via `--genomic-fm-cache`. The store must cover every CpG of all training
  axes (array, EPIC, WGBS) in the scope.

- `perceiver_deep_decoder` (`modeling/deep_decoder.py`, `configs/models/perceiver_deep.yaml`):
  same inherited functional query, but the part after the CpG query is deeper. The
  RNA tokens are re-read in `n_rounds` cross-attention rounds (each followed by
  `ffn_per_round` FFNs) at decoder `width` (512), followed by a wider head. The CpG
  branch is unchanged, so the comparison with `perceiver_cpg_query` isolates the
  decoder (19.9M -> 38.9M parameters at 25,017 genes). Guards against memorising
  training CpGs (untuned defaults, set under `model.decoder`): `cpg_dropout` on
  `h_c`, `drop_path` ramped over rounds, `zero_init` of rounds 2.. (training starts
  from the shallow decoder), head dropout = `final_regressor_dropout`. Judge it on
  the validation-CpG and double-OOD views, not on the train-CpG view.

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
