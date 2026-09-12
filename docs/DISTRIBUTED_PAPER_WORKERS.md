# Distributed paper experiment orchestration

The final paper jobs can be launched from multiple machines that see the same
`METHYL_DATA_ROOT` on dune. No fixed sharding is required.

## Shared registry

Operational state lives at:

```text
$METHYL_DATA_ROOT/experiments/coordination/paper_chr1_v1/
├── campaign.json
├── experiments.json
├── jobs/
├── claims/
├── logs/
└── locks/
```

`experiments.json` contains every paper job and its state:

```text
pending
running
completed
failed
blocked
```

For running jobs it also records the host, PID, GPU, attempt, timestamps and
shared log path. Completed jobs retain their paper record and Git publication
status.

The scientific completion marker remains the run's:

```text
paper/record.json
```

## Atomic scheduling

Each worker scans the same deterministic queue and atomically creates a claim
directory for the first missing runnable job. Only one machine can claim a job.

The worker writes a heartbeat every 60 seconds. A claim with no heartbeat for
15 minutes is stale and may be reclaimed. If `checkpoints/last.pt` exists, the
new worker automatically asks the paper API to resume the interrupted run.

## Frozen code provenance

The first worker freezes a `base_git_commit` in `campaign.json`.

Later Git commits are allowed only under `results/paper/**`. Every scientific
run records the frozen base code commit through
`METHYLPREDICTOR_CODE_COMMIT`, so result-only Git pushes do not change the
reported code provenance of later experiments.

If code/config/recipes change after the campaign has started, workers refuse to
continue. Start a new campaign name instead.

## Telegram

Create a bot using BotFather, send at least one message to the bot, and export
the credentials on every worker:

```bash
export METHYLPREDICTOR_TELEGRAM_BOT_TOKEN='123456:ABC...'
export METHYLPREDICTOR_TELEGRAM_CHAT_ID='123456789'
```

Test it with:

```bash
python scripts/test_telegram_notification.py
```

Notifications are sent on:

```text
STARTED
COMPLETED
ERROR
PUSH_ERROR
RECOVERED
```

Telegram failures are warnings and never terminate training.

## Launch a worker

Every machine should have a clean checkout of:

```text
refactor/repo-v2-2026-09
```

and its own machine-specific mount path:

```bash
export METHYL_DATA_ROOT=/.../MethylPredictionData
```

Then:

```bash
python scripts/preflight_paper_training.py \
  --allow-blocked-bulkrnabert

python scripts/paper_worker.py \
  --gpu 0 \
  --loop
```

Run the same command independently on kingkong, hal, ekko, dune, etc. All
workers use the shared claim/registry directory and therefore dynamically take
the next missing experiment.

Use filters when needed:

```bash
python scripts/paper_worker.py --gpu 0 --loop --study main
python scripts/paper_worker.py --gpu 0 --loop --study rna_encoder
python scripts/paper_worker.py --gpu 0 --loop --arm bottleneck_mlp
python scripts/paper_worker.py --gpu 0 --loop --seed 17
```

Failed jobs are not automatically repeated in the same campaign unless:

```bash
python scripts/paper_worker.py --gpu 0 --loop --retry-failed
```

## Watch progress

```bash
python scripts/paper_experiment_status.py
```

or:

```bash
python scripts/paper_experiment_status.py --watch 30
```

The raw shared JSON is:

```text
$METHYL_DATA_ROOT/experiments/coordination/paper_chr1_v1/experiments.json
```

## Per-experiment JSON and Git push

At completion, `paper_experiment.py` creates the full shared record on dune.

The worker then writes exactly one portable Git-curated JSON:

```text
results/paper/<study>/<arm>/seed<seed>.json
```

A shared dune lock serializes `git fetch`, commit and push, so workers on
different machines do not race.

Aggregate files (`runs.csv`, `runs.jsonl`, `registry.json`) are deliberately
not rewritten after every run because they would become multi-writer conflict
hotspots. Regenerate them deterministically after the campaign:

```bash
python scripts/collect_paper_runs.py
```

If the experiment completes but Git push fails, the shared scientific result
remains complete, Telegram emits `PUSH_ERROR`, and the next worker startup
retries publication without retraining.
