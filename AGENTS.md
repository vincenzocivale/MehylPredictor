# Workspace conventions

## Repository layout

- `third_party/MethylProphet/` is the pinned upstream MethylProphet submodule. Treat it as read-only and preserve a clean worktree.
- `docs/` is the documentation area for this audit project. Put reports, decisions, and links to external artifacts here.
- Project code belongs under `src/methylation_predictor/diagnostics/methylprophet/` or `src/methylation_predictor/genomic_encoder/`; shared code belongs in `common/`.
- Large downloaded artifacts and generated outputs remain untracked under `artifacts/`; diagnostics may expose only the canonical export contract to the genomic-encoder domain.

## MethylProphet reproducibility audit

- Prefer the upstream evaluation functions and preprocessing scripts over reimplementing their scientific operations.
- For PCC, use `MethylProphet/src/eval.py:compute_pcc_by_group` whenever possible. Any out-of-core logic may only partition input data; it must not redefine the metric.
- Treat released prediction rows (`group_idx`, `cpg_idx`, `sample_idx`, `pred_methyl`, `gt_methyl`) as the authoritative evidence of the split actually evaluated.
- Keep paper values, released prediction results, and author-provided log files as separate sources. Do not silently substitute one for another.
- Record Hugging Face repository name, revision when available, file list/size, commands, hashes of extracted ID manifests, and numerical comparison tolerances in `doc/`.

## Validation and reporting

- Run lightweight synthetic tests before processing the complete releases.
- For full runs, report the exact command, source artifact paths, number of processed rows, duplicate handling, NaN handling, and full-precision metrics.
- State explicitly whether a result reproduces the paper table, the released author logs, both, or neither.
