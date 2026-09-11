# Repo-v2 refactoring contract

The repository is being changed from a research-history codebase into the
paper-facing implementation.  Refactoring is allowed to remove dead parameters,
obsolete architecture families and historical dispatch machinery, but it must
not silently change the scientific experiment represented by the current J0/J1
candidates.

## Frozen candidate semantics

### J0: single retrieval

The current reference candidate is:

```text
functional peaks + dense locus annotations
        -> locus representation h_c
        -> training-only mean proxy

RNA expression
        -> 64 program tokens

h_c --query--> RNA program tokens
        -> pure RNA context r_pc

LN(h_c) || LN(r_pc)
        -> 512 -> 256 -> 128 -> 1
        -> beta_hat
```

The cross-attention output has no implicit `+ h_c` residual.  `mu_hat` never
feeds `beta_hat`.  The legacy genomic-embedding argument is ignored.

### J1: iterative retrieval

J1 shares functional encoder, RNA encoder, mean proxy, final regressor, loss,
optimizer, batching, seed and training budget with J0.  Its intended
architectural difference is the four-block iterative retrieval stack.

Each block is:

```text
state <- state + CrossAttention(LN(state), RNA)
state <- state + FFN(LN(state))
```

and the final regression still concatenates the original stable `h_c` with the
normalized final retrieval state.

## Training objective

For the current final candidates:

```text
L = beta_MSE + 0.15 * locus_PCC_loss + 0.15 * mean_proxy_MSE
```

Historical centered-MSE, sample-PCC, residual-Huber, shrinkage and beta-NLL
experiments are not part of the candidate objective.

## Refactor invariants

The refactor must preserve:

1. official Array/EPIC/WGBS pool membership for `mode=final`;
2. the three official evaluation views;
3. functional-only input semantics for J0/J1;
4. 64 x 256 RNA program-token semantics;
5. mean-proxy gradient isolation;
6. direct beta regression (the mean proxy is not an inference component);
7. J0's pure non-residual locus-to-RNA retrieval;
8. J1's four residual retrieval+FFN blocks;
9. current loss weights and batching;
10. checkpoint/result provenance sufficient to reproduce paper tables.

Tests in `tests/test_refactor_candidate_contract.py` freeze these contracts
before implementation code is moved.

## Known cleanup target: duplicated RNA attention machinery

J0/J1 currently reuse `LocusConditionedRNAEncoder` to generate RNA program
tokens, but that historical class also owns its own locus-attention
`query/key/value/out` projections.  J0/J1 do not use those projections: they
perform retrieval in `SimpleCrossAttention` / `BatchedCrossAttention` instead.

Phase 2b should therefore split:

```text
RNA expression -> ProgramTokenEncoder
locus + tokens -> Retrieval
```

into separate modules.  Removing the unused internal attention parameters is
considered a behavior-preserving cleanup, although old checkpoints may contain
those now-unused keys.  Checkpoint loading must handle that migration
explicitly rather than silently changing model behavior.

## Migration strategy

Phase 2a only establishes the public namespace and regression contracts.

Phase 2b will move implementations behind:

```python
methylation_predictor.modeling.SingleRetrievalPredictor
methylation_predictor.modeling.IterativeRetrievalPredictor
```

while keeping historical imports as temporary compatibility shims.  Only after
old and new implementations are shown equivalent will obsolete classes and
dispatch branches be deleted from `models.py` and `locus_cls_trainer.py`.

## Phase 2b status

J0 and J1 production dispatch now uses the paper-facing implementations in
`methylation_predictor.modeling`.

The historical classes remain temporarily in `models.py` so exact equivalence
and old-import compatibility can still be tested.

`tests/test_refactor_candidate_contract.py` verifies exact equality of the
initialized state dictionaries and deterministic forward outputs between the
historical and extracted J0/J1 implementations.

No RNA-token cleanup is included yet. The extracted candidates deliberately
still use the historical `build_rna_encoder`, preserving checkpoint keys and
numerical behavior. Removing the unused internal attention projections from
that encoder is phase 2c.
