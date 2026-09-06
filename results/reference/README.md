# results/reference — layout (rebuilt 2026-09-06)

Organized around the paper plan in `docs/PAPER_ROADMAP.md`, not around when/how each number was
produced. Three top-level folders:

- **`paper/`** — numbers published by MethylProphet, zero compute of our own. One file per table
  (A.1–A.5 in the plan): `table5_main_comparison.md`, `table7_source_ablation.md`,
  `table8_chromosome_generalization.md`, `table9_rna_encoder_ablation.md`,
  `predecessor_comparators.md`.
- **`ours/`** — our own experiments, one YAML per item, numbered to match the plan's execution
  order (B.1–B.6): `01_final_chr1_model.yaml`, `02_rna_encoder_comparison.yaml`,
  `03_mean_contribution.yaml`, `04_baselines.yaml`, `05_chromosome_generalization.yaml`,
  `06_source_ablation.yaml`. **Every entry states its `status`** — `confirmed_official` (real
  official-split number, checkpoint verified present), `candidate_found_needs_confirmation` (a
  real run exists but wasn't reviewed/promoted — read its `open_question`/`note` before citing),
  `dev_only_needs_official_run`, `found_but_suspicious_needs_review`, or `not_started`. Never cite
  a number from here as final without checking its `status` field first.
- **`appendix/`** — everything that was here before this reorganization (2026-09-06), moved
  wholesale, nothing deleted: `ablations.yaml` + `ablations/` (the architecture-selection ladder
  history — rungs A–F, architecture_novelty_2026_09, query_representation_2026_09,
  rna_encoder_comparison_2026_09 development numbers), `baselines/` and `methylprophet_comparison/`
  (the retired two-stage engine's frozen historical numbers), `mean_contribution_2026_09/`
  (protocol pointer), `rna_methylation/chr123.yaml` (the retired-engine chr123 number, kept as
  `legacy_two_stage`), `PROVENANCE.md` (checkpoint sha256 audit trail — paths inside it predate
  this reorganization, treat as historical, not current file locations).

**Rule**: `paper/*.md` files never carry a number of our own inline — they point to the matching
`ours/*.yaml` file instead, so a number is never duplicated in two places and can't drift out of
sync. Update `ours/*.yaml` when a run finishes; `paper/*.md` only changes when a MethylProphet
transcription needs correcting.
