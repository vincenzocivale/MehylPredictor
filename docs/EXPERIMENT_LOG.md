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
