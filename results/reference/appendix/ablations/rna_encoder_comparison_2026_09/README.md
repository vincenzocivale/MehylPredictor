# RNA encoder comparison (2026-09)

New comparators:
- `gene_pathway`: trainable SurvPath-style biologically sparse gene→pathway encoder.
- `bulkformer_147m`: frozen BulkFormer-147M sample embedding + trainable LayerNorm/Linear adapter.

Development selection uses fixed split seed 17. After architecture freeze, use the runner with `--mode final` for all three official TCGA chr1 views.
