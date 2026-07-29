"""Recompute the ICLR 2026 paper metrics from released MethylProphet predictions.

This script deliberately delegates Pearson correlation to ``src.eval.compute_pcc_by_group``.
The only new logic is an on-disk partitioning layer, needed because the released ENCODE
predictions do not fit in the memory required by the original ``eval_pcc.py`` script.

Example:
    python -m methylation_predictor.diagnostics.methylprophet.reproduce_paper_metrics \
      --input_result_df artifacts/cache/methylprophet/upstream_outputs/eval/eval-encode/eval_results-test.parquet \
      --input_group_idx_name_mapping_json artifacts/cache/methylprophet/upstream_outputs/eval/eval-encode/group_idx_name_mapping-test.json \
      --output_dir artifacts/diagnostics/methylprophet/reproducibility/encode
"""

import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from absl import app, flags, logging

from methylation_predictor.diagnostics.methylprophet.upstream import import_upstream

import_upstream()
from src.eval import compute_pcc_by_group  # type: ignore[import-not-found]


FLAGS = flags.FLAGS

flags.DEFINE_string("input_result_df", None, "Parquet file or directory of Parquet prediction shards.")
flags.mark_flag_as_required("input_result_df")
flags.DEFINE_string("input_group_idx_name_mapping_json", None, "Original group_idx mapping JSON.")
flags.mark_flag_as_required("input_group_idx_name_mapping_json")
flags.DEFINE_string("output_dir", None, "Directory for metrics, manifests, and the temporary work directory.")
flags.mark_flag_as_required("output_dir")
flags.DEFINE_integer("num_partitions", 128, "Hash partitions used for out-of-core grouping.")
flags.DEFINE_boolean("overwrite", False, "Replace an existing output directory.")
flags.DEFINE_boolean("keep_workdir", False, "Keep partitioned intermediate Parquet files.")
flags.DEFINE_string("input_sample_mapping_csv", None, "Optional sample_idx mapping CSV for annotated split manifests.")
flags.DEFINE_string("input_cpg_mapping_parquet", None, "Optional cpg_idx mapping Parquet for annotated split manifests.")
flags.DEFINE_string("artifact_revision", None, "Optional Hugging Face commit/revision for the downloaded artifacts.")
flags.DEFINE_enum(
    "stage",
    "all",
    ["all", "partition_cpg", "partition_sample", "pcc_cpg", "pcc_sample", "losses", "finalize"],
    "Run all stages or one resumable audit stage.",
)
flags.DEFINE_integer("input_shard_index", None, "Optional zero-based input shard index for the partition_cpg stage.")
flags.DEFINE_integer("group_idx", None, "Required group index for partition_sample, pcc_*, and losses stages.")
flags.DEFINE_integer("pcc_subpartitions", 1, "Additional on-disk partitions per PCC bucket; does not change metric logic.")


REQUIRED_COLUMNS = ["group_idx", "cpg_idx", "sample_idx", "pred_methyl", "gt_methyl"]
PAPER_METRICS = {
    "encode": {
        "train_cpg-val_sample": (0.3436, 0.9398, 0.0079, 0.0608),
        "val_cpg-train_sample": (0.7165, 0.9297, 0.0108, 0.0679),
        "val_cpg-val_sample": (0.3411, 0.9330, 0.0086, 0.0634),
    },
    "tcga": {
        "train_cpg-val_sample": (0.5455, 0.9320, 0.0199, 0.0882),
        "val_cpg-train_sample": (0.4194, 0.9065, 0.0266, 0.1000),
        "val_cpg-val_sample": (0.3904, 0.9059, 0.0271, 0.1011),
    },
}


def parquet_files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise ValueError(f"No Parquet files found in {path}")
    return files


def prepare_output_dir(output_dir, overwrite):
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise ValueError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    return output_dir


def load_group_mapping(path):
    with open(path) as handle:
        raw_mapping = json.load(handle)
    mapping = {int(group_idx): Path(group_name).name.replace(".parquet", "") for group_idx, group_name in raw_mapping.items()}
    expected = {"train_cpg-val_sample", "val_cpg-train_sample", "val_cpg-val_sample"}
    if set(mapping.values()) != expected:
        raise ValueError(f"Unexpected validation groups: {mapping}. Expected exactly {sorted(expected)}")
    return mapping


def read_prediction_shard(path):
    df = pd.read_parquet(path, columns=REQUIRED_COLUMNS)
    missing = set(REQUIRED_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def write_partitioned_predictions(files, work_dir, num_partitions):
    """Project and hash-partition predictions while recording raw-row statistics."""
    if num_partitions < 1:
        raise ValueError("num_partitions must be positive")
    counters = defaultdict(int)
    summary = {"input_files": len(files), "input_rows": 0, "rows_with_nan": 0, "rows_after_dropna": 0}
    for file_idx, file_path in enumerate(files):
        logging.info("Reading shard %d/%d: %s", file_idx + 1, len(files), file_path)
        df = read_prediction_shard(file_path)
        summary["input_rows"] += len(df)
        valid_df = df.dropna()
        summary["rows_with_nan"] += len(df) - len(valid_df)
        summary["rows_after_dropna"] += len(valid_df)
        for group_idx, group_df in valid_df.groupby("group_idx", sort=False):
            group_idx = int(group_idx)
            # The cpg partition is also used for global pointwise losses and deduplication.
            bucket = pd.util.hash_pandas_object(group_df["cpg_idx"], index=False).to_numpy() % num_partitions
            group_df = group_df.assign(_bucket=bucket)
            for bucket_idx, bucket_df in group_df.groupby("_bucket", sort=False):
                key = (group_idx, int(bucket_idx))
                output_path = (
                    work_dir
                    / "by_cpg"
                    / f"group={group_idx}"
                    / f"bucket={int(bucket_idx):04d}"
                    / f"part-{file_path.stem}-{counters[key]:06d}.parquet"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                bucket_df.drop(columns="_bucket").to_parquet(output_path, index=False)
                counters[key] += 1
    return summary


def partition_by_sample(work_dir, num_partitions, only_group_idx=None):
    counters = defaultdict(int)
    for cpg_file in sorted((work_dir / "by_cpg").rglob("*.parquet")):
        group_idx = int(cpg_file.parent.parent.name.split("=", 1)[1])
        if only_group_idx is not None and group_idx != only_group_idx:
            continue
        df = pd.read_parquet(cpg_file)
        bucket = pd.util.hash_pandas_object(df["sample_idx"], index=False).to_numpy() % num_partitions
        df = df.assign(_bucket=bucket)
        for bucket_idx, bucket_df in df.groupby("_bucket", sort=False):
            key = (group_idx, int(bucket_idx))
            output_path = work_dir / "by_sample" / f"group={group_idx}" / f"bucket={int(bucket_idx):04d}" / f"part-{counters[key]:06d}.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            bucket_df.drop(columns="_bucket").to_parquet(output_path, index=False)
            counters[key] += 1


def deduplicate_partition(df):
    key_columns = ["group_idx", "cpg_idx", "sample_idx"]
    duplicate_rows = int(df.duplicated(subset=key_columns, keep=False).sum())
    conflicting_rows = 0
    if duplicate_rows:
        duplicated_df = df.loc[df.duplicated(subset=key_columns, keep=False)]
        spread = duplicated_df.groupby(key_columns, sort=False)[["pred_methyl", "gt_methyl"]].agg(["min", "max"])
        conflicts = (spread[("pred_methyl", "min")] != spread[("pred_methyl", "max")]) | (
            spread[("gt_methyl", "min")] != spread[("gt_methyl", "max")]
        )
        conflicting_rows = int(conflicts.sum())
        if conflicting_rows:
            raise ValueError(
                f"Found {conflicting_rows} duplicate prediction keys with conflicting values; cannot reproduce the author's keep-first semantics safely."
            )
        df = df.drop_duplicates(subset=key_columns, keep="first")
    return df, duplicate_rows, conflicting_rows


def compute_partitioned_pcc(work_dir, partition_kind, group_mapping, subpartitions=1):
    group_key = "cpg_idx" if partition_kind == "by_cpg" else "sample_idx"
    output = defaultdict(list)
    audit = defaultdict(lambda: {"duplicate_rows": 0, "conflicting_duplicate_keys": 0})
    for group_idx, scenario in group_mapping.items():
        group_dir = work_dir / partition_kind / f"group={group_idx}"
        for bucket_dir in sorted(group_dir.glob("bucket=*")):
            files = sorted(bucket_dir.glob("*.parquet"))
            if not files:
                continue
            work_buckets = [files]
            if subpartitions > 1:
                sub_root = work_dir / "pcc_subpartitions" / partition_kind / f"group={group_idx}" / bucket_dir.name
                for file in files:
                    df = pd.read_parquet(file)
                    sub_idx = pd.util.hash_pandas_object(df[group_key], index=False).to_numpy() % subpartitions
                    for value, sub_df in df.assign(_sub_idx=sub_idx).groupby("_sub_idx", sort=False):
                        target = sub_root / f"sub={int(value):03d}" / file.name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        sub_df.drop(columns="_sub_idx").to_parquet(target, index=False)
                work_buckets = [sorted(path.glob("*.parquet")) for path in sorted(sub_root.glob("sub=*"))]
            for partition_files in work_buckets:
                df = pd.concat([pd.read_parquet(file) for file in partition_files], ignore_index=True)
                df, duplicate_rows, conflicting_rows = deduplicate_partition(df)
                audit[scenario]["duplicate_rows"] += duplicate_rows
                audit[scenario]["conflicting_duplicate_keys"] += conflicting_rows
                output[scenario].append(compute_pcc_by_group(df, group_key, backend="pandas"))
    return {scenario: pd.concat(series_list).sort_index() for scenario, series_list in output.items()}, dict(audit)


def compute_pointwise_losses_and_split_ids(work_dir, group_mapping):
    losses = defaultdict(lambda: {"n": 0, "squared_error_sum": 0.0, "absolute_error_sum": 0.0, "duplicate_rows": 0})
    ids = defaultdict(lambda: {"cpg_idx": [], "sample_idx": []})
    for group_idx, scenario in group_mapping.items():
        for bucket_dir in sorted((work_dir / "by_cpg" / f"group={group_idx}").glob("bucket=*")):
            files = sorted(bucket_dir.glob("*.parquet"))
            if not files:
                continue
            df = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
            df, duplicate_rows, _ = deduplicate_partition(df)
            error = df["pred_methyl"] - df["gt_methyl"]
            losses[scenario]["n"] += len(df)
            losses[scenario]["squared_error_sum"] += float((error**2).sum())
            losses[scenario]["absolute_error_sum"] += float(error.abs().sum())
            losses[scenario]["duplicate_rows"] += duplicate_rows
            ids[scenario]["cpg_idx"].append(pd.Series(df["cpg_idx"].unique()))
            ids[scenario]["sample_idx"].append(pd.Series(df["sample_idx"].unique()))
    for scenario, values in losses.items():
        values["mse"] = values["squared_error_sum"] / values["n"]
        values["mae"] = values["absolute_error_sum"] / values["n"]
    compact_ids = {
        scenario: {
            key: pd.concat(parts, ignore_index=True).drop_duplicates().sort_values(ignore_index=True)
            for key, parts in value.items()
        }
        for scenario, value in ids.items()
    }
    return dict(losses), compact_ids


def save_split_manifests(split_ids, output_dir, sample_mapping_path=None, cpg_mapping_path=None):
    roles = {
        "train_cpg": ("train_cpg-val_sample", "cpg_idx"),
        "val_cpg": ("val_cpg-train_sample", "cpg_idx"),
        "train_sample": ("val_cpg-train_sample", "sample_idx"),
        "val_sample": ("train_cpg-val_sample", "sample_idx"),
    }
    split_dir = output_dir / "splits"
    split_dir.mkdir()
    sample_mapping = pd.read_csv(sample_mapping_path) if sample_mapping_path else None
    cpg_mapping = pd.read_parquet(cpg_mapping_path) if cpg_mapping_path else None
    summary = {}
    for role, (scenario, id_column) in roles.items():
        df = pd.DataFrame({id_column: split_ids[scenario][id_column]})
        if id_column == "sample_idx" and sample_mapping is not None and "sample_idx" in sample_mapping:
            df = df.merge(sample_mapping, on="sample_idx", how="left")
        if id_column == "cpg_idx" and cpg_mapping is not None and "cpg_idx" in cpg_mapping:
            df = df.merge(cpg_mapping, on="cpg_idx", how="left")
        df.to_parquet(split_dir / f"{role}.parquet", index=False)
        summary[role] = {"scenario_source": scenario, "n": len(df), "sha256": hash_series(df[id_column])}
    # The third validation group must agree with both validation component sets.
    if not split_ids["val_cpg-val_sample"]["cpg_idx"].equals(split_ids["val_cpg-train_sample"]["cpg_idx"]):
        raise ValueError("Validation CpG IDs differ between validation scenarios.")
    if not split_ids["val_cpg-val_sample"]["sample_idx"].equals(split_ids["train_cpg-val_sample"]["sample_idx"]):
        raise ValueError("Validation sample IDs differ between validation scenarios.")
    return summary


def hash_series(series):
    hasher = hashlib.sha256()
    for value in series.to_numpy():
        hasher.update(f"{value}\n".encode())
    return hasher.hexdigest()


def infer_dataset_name(output_dir):
    candidate = str(output_dir).lower()
    if "encode" in candidate:
        return "encode"
    if "tcga" in candidate:
        return "tcga"
    return None


def write_artifact_manifest(files, group_mapping_path, output_dir, artifact_revision):
    manifest = {
        "artifact_revision": artifact_revision,
        "group_mapping_file": str(Path(group_mapping_path).resolve()),
        "prediction_files": [
            {"path": str(file.resolve()), "size_bytes": file.stat().st_size} for file in files
        ],
    }
    with open(output_dir / "artifact_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)


def write_audit_report(output_dir, metrics, split_summary, comparison):
    lines = ["# MethylProphet ICLR 2026 reproducibility audit", "", "## Recomputed metrics", ""]
    lines.append(comparison.to_markdown(index=False))
    lines.extend(["", "## Extracted split components", ""])
    split_table = pd.DataFrame(
        [{"component": name, **values} for name, values in sorted(split_summary.items())]
    )
    lines.append(split_table.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Method",
            "",
            "PCC values were computed with `src.eval.compute_pcc_by_group` from the original repository. "
            "Prediction shards were partitioned only to make that unmodified function fit in memory. "
            "MSE and MAE follow the pointwise expressions in `MethylEval` after `dropna()`.",
            "",
            "A metric passes when its value rounds to the four decimal places reported in the ICLR 2026 paper.",
        ]
    )
    (output_dir / "AUDIT_REPORT.md").write_text("\n".join(lines) + "\n")


def update_json_object(path, values):
    existing = {}
    if path.exists():
        with open(path) as handle:
            existing = json.load(handle)
    existing.update(values)
    with open(path, "w") as handle:
        json.dump(existing, handle, indent=2)


def compare_with_paper(metrics, dataset_name):
    reference = PAPER_METRICS.get(dataset_name, {})
    rows = []
    for scenario, values in metrics.items():
        row = {"scenario": scenario, **values}
        if scenario in reference:
            for metric_name, expected in zip(["mas_pcc", "mac_pcc", "mse", "mae"], reference[scenario]):
                row[f"paper_{metric_name}"] = expected
                row[f"delta_{metric_name}"] = values[metric_name] - expected
                row[f"pass_{metric_name}"] = round(values[metric_name], 4) == round(expected, 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("scenario")


def main(_):
    stage = FLAGS.stage
    output_dir = Path(FLAGS.output_dir)
    if stage == "all":
        output_dir = prepare_output_dir(output_dir, FLAGS.overwrite)
    elif stage == "partition_cpg":
        if output_dir.exists() and FLAGS.overwrite:
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    elif not output_dir.exists():
        raise ValueError(f"Output directory does not exist for resumable stage {stage}: {output_dir}")
    group_mapping = load_group_mapping(FLAGS.input_group_idx_name_mapping_json)
    files = parquet_files(FLAGS.input_result_df)
    work_dir = output_dir / "work"
    work_dir.mkdir(exist_ok=True)
    result_dir = work_dir / "results"
    result_dir.mkdir(exist_ok=True)

    if stage in ("all", "partition_cpg"):
        selected_files = files
        if FLAGS.input_shard_index is not None:
            if FLAGS.input_shard_index < 0 or FLAGS.input_shard_index >= len(files):
                raise ValueError(f"input_shard_index must be in [0, {len(files) - 1}]")
            selected_files = [files[FLAGS.input_shard_index]]
        write_artifact_manifest(files, FLAGS.input_group_idx_name_mapping_json, output_dir, FLAGS.artifact_revision)
        input_summary = write_partitioned_predictions(selected_files, work_dir, FLAGS.num_partitions)
        if stage == "partition_cpg":
            return
    if stage in ("all", "partition_sample"):
        if stage == "partition_sample" and FLAGS.group_idx is None:
            raise ValueError("--group_idx is required for the resumable partition_sample stage")
        partition_by_sample(work_dir, FLAGS.num_partitions, FLAGS.group_idx)
        if stage == "partition_sample":
            return
    if stage in ("all", "pcc_cpg", "pcc_sample"):
        if stage != "all" and FLAGS.group_idx is None:
            raise ValueError(f"--group_idx is required for the resumable {stage} stage")
        selected_mapping = group_mapping
        if stage != "all":
            if FLAGS.group_idx not in group_mapping:
                raise ValueError(f"Unknown group_idx: {FLAGS.group_idx}")
            selected_mapping = {FLAGS.group_idx: group_mapping[FLAGS.group_idx]}
        if stage in ("all", "pcc_cpg"):
            pcc_by_cpg, cpg_audit = compute_partitioned_pcc(
                work_dir, "by_cpg", selected_mapping, FLAGS.pcc_subpartitions
            )
            for scenario, pcc in pcc_by_cpg.items():
                pcc.rename("pcc").to_frame().to_parquet(result_dir / f"pcc_by_cpg_id-{scenario}.parquet")
            update_json_object(result_dir / "cpg_audit.json", cpg_audit)
            if stage == "pcc_cpg":
                return
        if stage in ("all", "pcc_sample"):
            pcc_by_sample, sample_audit = compute_partitioned_pcc(
                work_dir, "by_sample", selected_mapping, FLAGS.pcc_subpartitions
            )
            for scenario, pcc in pcc_by_sample.items():
                pcc.rename("pcc").to_frame().to_parquet(result_dir / f"pcc_by_sample_id-{scenario}.parquet")
            update_json_object(result_dir / "sample_audit.json", sample_audit)
            if stage == "pcc_sample":
                return
    if stage in ("all", "losses"):
        if stage == "losses" and FLAGS.group_idx is None:
            raise ValueError("--group_idx is required for the resumable losses stage")
        selected_mapping = group_mapping if stage == "all" else {FLAGS.group_idx: group_mapping[FLAGS.group_idx]}
        losses, split_ids = compute_pointwise_losses_and_split_ids(work_dir, selected_mapping)
        for scenario, values in losses.items():
            with open(result_dir / f"losses-{scenario}.json", "w") as handle:
                json.dump(values, handle, indent=2)
            for key, series in split_ids[scenario].items():
                pd.DataFrame({key: series}).to_parquet(result_dir / f"ids-{scenario}-{key}.parquet", index=False)
        if stage == "losses":
            return
    if stage not in ("all", "finalize"):
        raise ValueError(f"Unsupported stage: {stage}")

    pcc_by_cpg = {}
    pcc_by_sample = {}
    losses = {}
    split_ids = {}
    for scenario in group_mapping.values():
        pcc_by_cpg[scenario] = pd.read_parquet(result_dir / f"pcc_by_cpg_id-{scenario}.parquet")["pcc"]
        pcc_by_sample[scenario] = pd.read_parquet(result_dir / f"pcc_by_sample_id-{scenario}.parquet")["pcc"]
        with open(result_dir / f"losses-{scenario}.json") as handle:
            losses[scenario] = json.load(handle)
        split_ids[scenario] = {
            key: pd.read_parquet(result_dir / f"ids-{scenario}-{key}.parquet")[key]
            for key in ("cpg_idx", "sample_idx")
        }
    split_summary = save_split_manifests(
        split_ids, output_dir, FLAGS.input_sample_mapping_csv, FLAGS.input_cpg_mapping_parquet
    )

    metrics = {}
    for scenario in sorted(group_mapping.values()):
        pcc_by_cpg[scenario].rename("pcc").to_frame().to_parquet(output_dir / f"pcc_by_cpg_id-{scenario}.parquet")
        pcc_by_sample[scenario].rename("pcc").to_frame().to_csv(output_dir / f"pcc_by_sample_id-{scenario}.csv")
        metrics[scenario] = {
            "mas_pcc": float(pcc_by_cpg[scenario].median()),
            "mac_pcc": float(pcc_by_sample[scenario].median()),
            "mse": float(losses[scenario]["mse"]),
            "mae": float(losses[scenario]["mae"]),
            "n_pairs": int(losses[scenario]["n"]),
            "pcc_cpg_groups": int(len(pcc_by_cpg[scenario])),
            "pcc_sample_groups": int(len(pcc_by_sample[scenario])),
        }
    with open(output_dir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    with open(output_dir / "split_summary.json", "w") as handle:
        json.dump(split_summary, handle, indent=2)
    input_summary = {"input_files": len(files), "mode": "all" if stage == "all" else "resumable"}
    with open(result_dir / "cpg_audit.json") as handle:
        cpg_audit = json.load(handle)
    with open(result_dir / "sample_audit.json") as handle:
        sample_audit = json.load(handle)
    audit = {"input": input_summary, "cpg_partitions": cpg_audit, "sample_partitions": sample_audit}
    with open(output_dir / "audit.json", "w") as handle:
        json.dump(audit, handle, indent=2)

    comparison = compare_with_paper(metrics, infer_dataset_name(output_dir))
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    comparison.to_csv(output_dir / "metrics.csv", index=False)
    write_audit_report(output_dir, metrics, split_summary, comparison)
    logging.info("Metrics:\n%s", comparison.to_string(index=False))
    if not FLAGS.keep_workdir:
        shutil.rmtree(work_dir)


if __name__ == "__main__":
    app.run(main)
