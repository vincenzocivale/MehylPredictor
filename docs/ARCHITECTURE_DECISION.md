# Final architecture decision

**Decision date:** 2026-09-12  
**Status:** locked for fresh final training

The final architecture is J9b functional depth plus two residual FFN blocks in
the final prediction head.

```text
class                       EfficientSingleAttentionPredictor
retrieval FFN blocks        8
functional FFN blocks       8
deep_query                  false
head FFN blocks             2
selector                    efficient_single_attn_8ffn_residual_functional8_head2
```

The historical J9b recipe/selector is not modified. Existing J9b checkpoints
remain audit/resume compatible, but they are not checkpoints of the Head2 final
model. Final Head2 paper results must come from fresh training.
