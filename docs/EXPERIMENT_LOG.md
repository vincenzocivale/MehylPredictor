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
| 2026-09-13 | functional_representation | basic_context | 17 | functional_representation__basic_context__seed17 | ekko | running | restarted via `scripts/run_functional_representation_local.sh` (bypasses `paper_worker.py`'s shared campaign; single-seed only per user direction 2026-09-13 -- secondary/supplementary studies get seed 17 for now, not the full 3-seed matrix). Reads chr1_runtime inputs from a local /mnt/hdd mirror instead of the sshfs dune mount, after a transient concurrent-write failure there. Along the way fixed two repo-wide bugs found blocking this: `classify_recipe()` not resolving `extends:` (976f9f8), and `MatchedChr1Protocol` missing `primary_source`/`train_cpg_idx`/`val_cpg_idx` after 695043e's generalization, which broke all chr1 training (488af12) |
| 2026-09-13 | functional_representation | full_functional | 17 | functional_representation__full_functional__seed17 | ekko | queued | next in `run_functional_representation_local.sh`'s sequential queue after basic_context/seed17 |
| 2026-09-13 | functional_representation | minimal | 17 | functional_representation__minimal__seed17 | ekko | queued | last in `run_functional_representation_local.sh`'s sequential queue |
| 2026-09-13 | main (chr123, not in paper_studies.yaml matrix) | main | 17 | chr123-main-seed17-iotuned | dune.micc | stopped, superseded | killed after ~2min to free GPU1 for a single urgent higher-priority run (chr123-main-seed17-bigbatch below); no checkpoint reached |
| 2026-09-13 | main (chr123, not in paper_studies.yaml matrix) | main | 42 | chr123-main-seed42-iotuned | dune.micc | stopped, superseded | killed alongside seed17-iotuned above, same reason; no checkpoint reached |
| 2026-09-13 | main (chr123, not in paper_studies.yaml matrix) | main | 17 | chr123-main-seed17-bigbatch | dune.micc | running | urgent single-seed run for paper writing (ENCODE experiment), requested ahead of the 3-seed matrix. Recipe `configs/models/chr123/main_chr123_bigbatch_solo.yaml` (hdf5_cache_mb 256->2048, array.cpg_size 640->2048 chunk-aligned, larger sample_size per pool since running solo on GPU1 with no second process sharing memory); same rna_cache/cpg-targets-dir/locus-store as the iotuned recipes above. Running at ~39.7/41GB GPU1 memory, 100% util -- little headroom, watch for OOM. Seeds 42/123 (and the full paper matrix) deferred until after this result. |
| 2026-09-13/14 | encode_wgbs (not in paper_studies.yaml matrix; single-source ENCODE WGBS study, see `configs/models/encode_wgbs_runtime.yaml` + `--dataset-source encode_wgbs`, added in 695043e) | main | 17 | encode-wgbs-seed17-b128-c65536 | hal | resumed, running | First attempt reached 18/80 epochs (loss decreasing normally, 0.0986->0.0537) then died silently with no Python traceback -- consistent with an OOM-kill, not a code bug (GPU peak only 10.2/40GB; `hal` is a heavily shared multi-user JupyterHub host, host swap was 100% full at diagnosis time from other users' processes). Resumed via `scripts/train.py --resume` from `checkpoints/last.pt` (epoch 18) on 2026-09-14; epoch 19 completed successfully (loss 0.0535) past the previous failure point. Data lives in `local_methyl_data/` (fast local mirror of `dune_data`, synced via `scripts/sync_local_encode_cache.sh` / direct `rsync -e ssh dune:...` -- the sshfs/rclone mount of `dune_data` was too slow for bulk transfer and is unmounted). Earlier attempts `encode-wgbs-seed17`, `encode-wgbs-seed17-b128`, `encode-wgbs-seed29`, `encode-wgbs-seed43` never logged a first epoch (pre-date the 695043e/488af12 fixes) and are abandoned in favor of this run; seeds 29/43 not yet relaunched. |
