# Experiment launch log

Human-readable ledger of every paper experiment run launched, across all
machines. This is a process/coordination aid (multiple machines and people
can launch runs against the same shared `METHYL_DATA_ROOT`) — it is not a
substitute for the per-run `metadata.json` (which already records `host`,
`git_commit`, `created_at_utc` and the full resolved config) or for
`configs/paper_studies.yaml` (the frozen arm/seed matrix). See
[`PAPER_EXPERIMENTS.md`](PAPER_EXPERIMENTS.md) for the experiment catalog
this log's `study`/`arm` columns refer to.

**Add one row here whenever you launch a run against the shared paper study
surface** (`configs/paper_studies.yaml`), before or immediately after
starting it — see the logging rule in `CLAUDE.md`.

| Date (UTC) | Study | Arm | Seed | Run ID | Machine | Status | Notes |
|---|---|---|---|---|---|---|---|
| 2026-09-12 | main | main | 17 | main-seed17 | kingkong | superseded | crashed before first checkpoint; superseded by main-seed17-r3 |
| 2026-09-12 | main | main | 42 | main-seed42 | kingkong | superseded | crashed before first checkpoint; superseded by main-seed42-r3 |
| 2026-09-12 | main | main | 123 | main-seed123 | kingkong | failed, not relaunched (by decision) | crashed before first checkpoint. 2026-09-13: `configs/paper_studies.yaml` and `configs/data/storage_v2.yaml` updated to seeds [17, 42, 123] (was [17, 29, 43]) to match reality instead of relaunching; seed 123 still needs an actual successful run before the `main` study has 3 complete seeds |
| 2026-09-13 | main | main | 17 | main-seed17-r3 | kingkong | running | 53/80 epochs as of 2026-09-13, loss decreasing normally |
| 2026-09-13 | main | main | 42 | main-seed42-r3 | kingkong | running | 53/80 epochs as of 2026-09-13, loss decreasing normally |
| 2026-09-13 | functional_representation | basic_context | 17 | functional_representation__basic_context__seed17 | ekko | running | launched via `scripts/paper_worker.py --loop --study functional_representation` (first job in a new shared campaign); worker will proceed through the remaining basic_context/full_functional/minimal x seed17/29/43 jobs automatically |
| 2026-09-13 | main (chr123, not in paper_studies.yaml matrix) | main | 17 | chr123-main-seed17-iotuned | dune.micc | stopped, superseded | killed after ~2min to free GPU1 for a single urgent higher-priority run (chr123-main-seed17-bigbatch below); no checkpoint reached |
| 2026-09-13 | main (chr123, not in paper_studies.yaml matrix) | main | 42 | chr123-main-seed42-iotuned | dune.micc | stopped, superseded | killed alongside seed17-iotuned above, same reason; no checkpoint reached |
| 2026-09-13 | main (chr123, not in paper_studies.yaml matrix) | main | 17 | chr123-main-seed17-bigbatch | dune.micc | running | urgent single-seed run for paper writing (ENCODE experiment), requested ahead of the 3-seed matrix. Recipe `configs/models/chr123/main_chr123_bigbatch_solo.yaml` (hdf5_cache_mb 256->2048, array.cpg_size 640->2048 chunk-aligned, larger sample_size per pool since running solo on GPU1 with no second process sharing memory); same rna_cache/cpg-targets-dir/locus-store as the iotuned recipes above. Running at ~39.7/41GB GPU1 memory, 100% util -- little headroom, watch for OOM. Seeds 42/123 (and the full paper matrix) deferred until after this result. |
