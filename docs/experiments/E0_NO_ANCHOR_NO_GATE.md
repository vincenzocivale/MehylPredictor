# E0 architecture ablation: no mean-RNA anchoring, no variability gate

## Question

Does the current RNA-to-DNAm model benefit from the two explicit residual
constraints, or is a simpler direct residual model sufficient/better?

The candidate removes **both**:

- mean-RNA anchoring;
- the locus-specific variability gate.

Everything else remains unchanged relative to the current exact E0
Array-chr1 benchmark.

## Baseline

Current model:

```text
raw(s,i) = interaction(RNA_s, CpG_i)
delta(s,i) = gate_i * [raw(s,i) - raw(mean_RNA,i)]
beta_hat(s,i) = sigmoid(logit(prior_i) + delta(s,i))
```

## Candidate

```text
delta(s,i) = interaction(RNA_s, CpG_i)
beta_hat(s,i) = sigmoid(logit(prior_i) + delta(s,i))
```

The NTv3 locus embedding, frozen prior, RNA encoder, concat/product
interaction head, zero-initialized residual head, objective, optimizer,
batching, full-coverage schedule, nested development split, final refit and
three official Array-chr1 evaluation views are unchanged.

`gate.kind: none` is implemented as a parameter-free identity gate and is
available only when explicitly requested by configuration. The canonical
default remains `gate.kind: variability` and `anchor_to_mean_rna: true`.

## Why this is an isolated experiment

The experiment is not promoted into `configs/train.yaml`. The launcher
creates the exact derived config inside its own run directory:

```text
artifacts/architecture_ablation/e0_no_anchor_no_gate_seed17/
  ablation_base_config.yaml
```

Therefore a negative result requires no rollback of the canonical model.
If the experiment is later accepted, promotion should be a separate commit
that changes the defaults and documentation deliberately.

## Run

```bash
nohup bash scripts/run_e0_ablation_no_anchor_no_gate.sh \
  > e0_no_anchor_no_gate.launch.log 2>&1 &
```

The launcher reuses `scripts/run_overnight_current_model_vs_mp.sh`, so the
data/split/training protocol is the same as the current E0 benchmark.

Released MethylProphet predictions are not required. Automatic download is
disabled by default for this ablation because the release dataset may be
gated.

## Outputs

Candidate:

```text
artifacts/architecture_ablation/e0_no_anchor_no_gate_seed17/
```

If the current E0 baseline headline exists at the default path, the launcher
also writes:

```text
evaluation/compare_to_current_e0.json
```

The comparison is descriptive and intentionally does **not** decide whether
the candidate should become permanent.
