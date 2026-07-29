import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "src/methylation_predictor/diagnostics/methylprophet/reproduce_paper_metrics.py"
pytestmark = pytest.mark.regression
# The project evaluation module imports the full training visualization stack.  The
# wrapper is integration-tested against it in the MethylProphet environment; this
# unit test substitutes its pandas implementation so it remains lightweight.
EVAL_STUB = types.ModuleType("src.eval")


def compute_pcc_by_group(df, group_key, backend="pandas"):
    if backend != "pandas":
        raise ValueError("Test stub supports the original pandas backend only")
    result = df.groupby(group_key)[["pred_methyl", "gt_methyl"]].corr().unstack().iloc[:, 1]
    result.name = "Me PCC"
    result.index.name = f"By {group_key}"
    return result


EVAL_STUB.compute_pcc_by_group = compute_pcc_by_group
sys.modules.setdefault("src", types.ModuleType("src"))
sys.modules["src.eval"] = EVAL_STUB
SPEC = importlib.util.spec_from_file_location("reproduce_paper_metrics", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReproducePaperMetricsTest(unittest.TestCase):
    def test_partitioned_metrics_match_original_pcc_function(self):
        rows = []
        for group_idx, samples, cpgs in [(0, [10, 11], [100, 101]), (1, [20, 21], [200, 201]), (2, [20, 21], [200, 201])]:
            for sample_idx in samples:
                for cpg_idx in cpgs:
                    gt = (sample_idx + cpg_idx) / 1000
                    rows.append(
                        {
                            "group_idx": group_idx,
                            "sample_idx": sample_idx,
                            "cpg_idx": cpg_idx,
                            "pred_methyl": gt * 0.9 + 0.01,
                            "gt_methyl": gt,
                        }
                    )
        # This row verifies that the wrapper follows the original dropna rule.
        rows.append({"group_idx": 0, "sample_idx": 10, "cpg_idx": 100, "pred_methyl": 0.5, "gt_methyl": None})
        df = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source = tmpdir / "source"
            source.mkdir()
            df.to_parquet(source / "000000.parquet", index=False)
            mapping_path = tmpdir / "mapping.json"
            mapping_path.write_text(
                json.dumps(
                    {
                        "0": "train_cpg-val_sample.parquet",
                        "1": "val_cpg-train_sample.parquet",
                        "2": "val_cpg-val_sample.parquet",
                    }
                )
            )
            work_dir = tmpdir / "work"
            work_dir.mkdir()
            MODULE.write_partitioned_predictions([source / "000000.parquet"], work_dir, 2)
            MODULE.partition_by_sample(work_dir, 2)
            mapping = MODULE.load_group_mapping(mapping_path)
            by_cpg, _ = MODULE.compute_partitioned_pcc(work_dir, "by_cpg", mapping)
            by_sample, _ = MODULE.compute_partitioned_pcc(work_dir, "by_sample", mapping)
            expected = df.dropna()
            for group_idx, scenario in mapping.items():
                group_df = expected[expected.group_idx == group_idx]
                pd.testing.assert_series_equal(
                    by_cpg[scenario], MODULE.compute_pcc_by_group(group_df, "cpg_idx", backend="pandas").sort_index()
                )
                pd.testing.assert_series_equal(
                    by_sample[scenario], MODULE.compute_pcc_by_group(group_df, "sample_idx", backend="pandas").sort_index()
                )

    def test_conflicting_duplicate_is_rejected(self):
        df = pd.DataFrame(
            [
                {"group_idx": 0, "cpg_idx": 1, "sample_idx": 2, "pred_methyl": 0.1, "gt_methyl": 0.2},
                {"group_idx": 0, "cpg_idx": 1, "sample_idx": 2, "pred_methyl": 0.3, "gt_methyl": 0.2},
            ]
        )
        with self.assertRaisesRegex(ValueError, "conflicting"):
            MODULE.deduplicate_partition(df)


if __name__ == "__main__":
    unittest.main()
