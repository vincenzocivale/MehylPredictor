# A.4 — MethylProphet Table 9: RNA encoder (gene-pathway) ablation

**Source: MethylProphet paper (published), not our training.** MethylProphet's own comparison of
its default RNA encoder against a gene-pathway variant, on their architecture.

| RNA representation | Train CpG × Val Sample | Val CpG × Train Sample | Val CpG × Val Sample |
|---|---:|---:|---:|
| MethylProphet gene-pathway | 0.5371 | 0.4194 | 0.3959 |
| MethylProphet default | 0.5455 | 0.4194 | 0.3904 |

**Interpretation for our story**: MethylProphet's own gene-pathway variant barely moves their
number. This motivates our claim that a biologically-structured RNA encoder alone isn't the lever
— what matters in our model is making the RNA representation **locus-conditioned** (cross-attention),
not just biologically structured. **Do not blend these numbers with our own gene-pathway encoder's
results as if they were the same implementation** — this table is a citation from MethylProphet's
paper, on MethylProphet's architecture, nothing more.

**Ours**: see [`../ours/02_rna_encoder_comparison.yaml`](../ours/02_rna_encoder_comparison.yaml)
for the current-architecture RNA-encoder comparison table (cross-attention vs. bottleneck-MLP vs.
BulkRNABert vs. our own gene-pathway encoder vs. BulkFormer-147M) — a different, current-repo
experiment, not a reproduction of this table.
