from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / "rna_branch" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hallmark_alias_matching_is_unambiguous():
    module = _load_script("prepare_hallmark_modules.py")
    mapping = module._unique_rna_id_map(np.asarray([
        "TP53;ENSG00000141510.18", "BRCA1;ENSG00000012048.23", "TP53;ENSG99999999999.1",
    ]))
    assert mapping["BRCA1"] == "BRCA1;ENSG00000012048.23"
    assert mapping["ENSG00000141510"] == "TP53;ENSG00000141510.18"
    assert "TP53" not in mapping


def test_bulkrnabert_normalises_composite_rna_labels():
    module = _load_script("extract_bulkrnabert_embeddings.py")
    assert module._normalise_gene_id("TSPAN6;ENSG00000000003.15") == "ENSG00000000003"
    assert module._normalise_gene_id("ENSG00000000003.15") == "ENSG00000000003"


def test_representation_configs_warm_start_by_matching_seed(tmp_path, monkeypatch):
    module = _load_script("make_representation_configs.py")
    base = {
        "run_name": "base", "output_dir": "unused", "data": {},
        "model": {}, "training": {"seed": 17, "learning_rate": 1e-4},
    }
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base))
    monkeypatch.setattr("sys.argv", [
        "make_representation_configs.py", "--base", str(base_path), "--output-root", str(tmp_path / "out"),
        "--seeds", "17,23", "--f2-checkpoint", "17=f2_17.pt", "--f2-checkpoint", "23=f2_23.pt",
        "--module-weights", "hallmark=modules.npz", "--gene-embeddings", "genes.npz",
    ])
    module.main()
    r1_17 = yaml.safe_load((tmp_path / "out/configs/r1_experts4_s17.yaml").read_text())
    r1_23 = yaml.safe_load((tmp_path / "out/configs/r1_experts4_s23.yaml").read_text())
    r2_23 = yaml.safe_load((tmp_path / "out/configs/r2_hallmark_module_query_s23.yaml").read_text())
    r4_17 = yaml.safe_load((tmp_path / "out/configs/r4_gene_query_k64_s17.yaml").read_text())
    assert r1_17["training"]["warm_start_checkpoint"] == "f2_17.pt"
    assert r1_23["training"]["warm_start_checkpoint"] == "f2_23.pt"
    assert r2_23["training"]["warm_start_checkpoint"] == "f2_23.pt"
    assert r4_17["training"]["warm_start_checkpoint"] == "f2_17.pt"
    assert not (tmp_path / "out/configs/r0_zscore_s17.yaml").exists()


def test_representation_configs_r5b_nested_and_shuffled(tmp_path, monkeypatch):
    module = _load_script("make_representation_configs.py")
    base = {
        "run_name": "base", "output_dir": "unused", "data": {},
        "model": {}, "training": {"seed": 17, "learning_rate": 1e-4},
    }
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base))

    monkeypatch.setattr("sys.argv", [
        "make_representation_configs.py", "--base", str(base_path), "--output-root", str(tmp_path / "out"),
        "--seeds", "17", "--f2-checkpoint", "17=f2_17.pt",
        "--gene-embeddings", "genes.npz",
        "--pretrained-rna", "bulkrnabert=pretrained_v2.h5",
        "--r5-seeds", "17", "--r5-modes", "concat",
        "--r5b-layers", "1,3",
    ])
    module.main()

    layer1 = yaml.safe_load((tmp_path / "out/configs/r5b_bulkrnabert_layer1_s17.yaml").read_text())
    layer3 = yaml.safe_load((tmp_path / "out/configs/r5b_bulkrnabert_layer3_s17.yaml").read_text())
    for config in (layer1, layer3):
        assert config["training"]["warm_start_checkpoint"] == "f2_17.pt"
        assert config["training"]["freeze_warm_start_params"] is True
        assert config["training"]["seed_initial_checkpoint"] is True
        assert config["model"]["encoder"]["kind"] == "pretrained_embedding"
        assert config["model"]["interaction"]["kind"] == "cpg_pretrained_f2"
        assert config["data"]["pretrained_mode"] == "concat"
        assert config["data"]["pretrained_control"] == "real"
    assert layer1["data"]["pretrained_rna"]["values_key"] == "embeddings_layer1"
    assert layer3["data"]["pretrained_rna"]["values_key"] == "embeddings_layer3"

    # Legacy r5_* family (mode=concat) must stay unchanged: still never warm-started.
    legacy = yaml.safe_load((tmp_path / "out/configs/r5_bulkrnabert_concat_s17.yaml").read_text())
    assert legacy["training"]["warm_start_checkpoint"] is None

    monkeypatch.setattr("sys.argv", [
        "make_representation_configs.py", "--base", str(base_path), "--output-root", str(tmp_path / "out"),
        "--seeds", "17", "--f2-checkpoint", "17=f2_17.pt",
        "--gene-embeddings", "genes.npz",
        "--pretrained-rna", "bulkrnabert=pretrained_v2.h5",
        "--r5-seeds", "17", "--r5-modes", "concat",
        "--r5b-shuffled-layer", "3",
    ])
    module.main()
    shuffled = yaml.safe_load(
        (tmp_path / "out/configs/r5b_bulkrnabert_concat_shuffled_layer3_s17.yaml").read_text()
    )
    assert shuffled["data"]["pretrained_mode"] == "concat"
    assert shuffled["data"]["pretrained_control"] == "shuffle_within_cancer"
    assert shuffled["data"]["pretrained_rna"]["values_key"] == "embeddings_layer3"
    assert shuffled["training"]["warm_start_checkpoint"] == "f2_17.pt"
    assert shuffled["training"]["freeze_warm_start_params"] is True


def test_bulkrnabert_scale_report_detects_log2_tpm_and_corruption():
    module = _load_script("audit_bulkrnabert_inputs.py")
    rng = np.random.default_rng(0)
    raw = rng.random((5, 40)).astype(np.float32)
    tpm = raw / raw.sum(axis=1, keepdims=True) * 1_000_000.0
    log2p1 = np.log2(tpm + 1.0).astype(np.float32)

    reconstructed = module.to_raw_tpm(log2p1, "log2p1")
    good = module.scale_report(reconstructed, log2p1)
    assert good["verified_log2_tpm_plus_one"] is True
    assert good["tpm_sum_mean_abs_error_from_1e6"] < 1.0

    not_tpm = np.log2(raw * 3.0 + 1.0).astype(np.float32)
    reconstructed_bad = module.to_raw_tpm(not_tpm, "log2p1")
    bad = module.scale_report(reconstructed_bad, not_tpm)
    assert bad["verified_log2_tpm_plus_one"] is False


def test_bulkrnabert_custom_tokens_deterministic_and_detects_scale_bug():
    module = _load_script("audit_bulkrnabert_inputs.py")
    rng = np.random.default_rng(1)
    raw = rng.random((6, 50)).astype(np.float32)
    tpm = raw / raw.sum(axis=1, keepdims=True) * 1_000_000.0
    checkpoint_config = {"normalization_factor": 5.873553674586928, "n_expressions_bins": 64}

    tokens_a, _ = module.custom_tokens(tpm, checkpoint_config)
    tokens_b, _ = module.custom_tokens(tpm, checkpoint_config)
    np.testing.assert_array_equal(tokens_a, tokens_b)

    # A log2-instead-of-log10 style scale bug shifts the normalization factor by
    # ln(2)/ln(10); this must be caught as a nonzero token_mismatch_count, not
    # merely a high correlation.
    buggy_config = {**checkpoint_config, "normalization_factor": checkpoint_config["normalization_factor"] / 3.321928}
    tokens_buggy, _ = module.custom_tokens(tpm, buggy_config)
    assert int((tokens_a != tokens_buggy).sum()) > 0


def test_bulkrnabert_align_to_common_genes_zero_fills_missing():
    module = _load_script("audit_bulkrnabert_inputs.py")
    values = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    source_ids = ["SYM;ENSG00000000001.3", "SYM;ENSG00000000002.1", "SYM;ENSG00000000003.9"]
    common_genes = ["ENSG00000000002", "ENSG00000000004", "ENSG00000000001"]
    aligned, overlap = module.align_to_common_genes(values, source_ids, common_genes)
    np.testing.assert_array_equal(aligned, np.asarray([[2.0, 0.0, 1.0], [5.0, 0.0, 4.0]], dtype=np.float32))
    assert overlap == pytest.approx(2.0 / 3.0)


@pytest.mark.data
def test_bulkrnabert_custom_tokens_match_official_pipeline():
    module = _load_script("audit_bulkrnabert_inputs.py")
    repo = Path("artifacts/models/multiomics-open-research")
    checkpoint_dir = repo / "checkpoints" / "bulk_rna_bert_gtex_encode"
    if not (checkpoint_dir / "config.json").is_file():
        pytest.skip("official multiomics-open-research checkout not available")
    import sys

    sys.path.insert(0, str(repo.resolve()))
    checkpoint_config = json.loads((checkpoint_dir / "config.json").read_text())
    rng = np.random.default_rng(2)
    raw = rng.random((5, checkpoint_config["n_genes"])).astype(np.float32)
    tpm = raw / raw.sum(axis=1, keepdims=True) * 1_000_000.0

    tokens_custom, _ = module.custom_tokens(tpm, checkpoint_config)
    tokens_official = module.official_tokens(tpm, checkpoint_dir)
    mismatch = int((tokens_custom != tokens_official).sum())
    assert mismatch == 0


def test_dedup_genes_tpm_space_keeps_identical_and_sums_differing_duplicates():
    module = _load_script("build_bulkrnabert_gene_source.py")

    def _log2p1(tpm_values):
        return np.log2(np.asarray(tpm_values, dtype=np.float64) + 1.0)

    raw_ids = np.asarray(
        [
            "X;ENSG00000000001.1",
            "X;ENSG00000000001.2",
            "Y;ENSG00000000002.5",
            "Z;ENSG00000000003.1",
            "Z;ENSG00000000003.9",
        ],
        dtype=object,
    )
    values_log2p1 = np.stack(
        [
            _log2p1([100.0, 50.0]),  # ENSG...0001, row a (identical duplicate)
            _log2p1([100.0, 50.0]),  # ENSG...0001, row b
            _log2p1([20.0, 10.0]),  # ENSG...0002, unique
            _log2p1([30.0, 5.0]),  # ENSG...0003, row a (differing duplicate)
            _log2p1([40.0, 15.0]),  # ENSG...0003, row b
        ]
    )

    resolved, gene_ids, policy = module.dedup_genes_tpm_space(raw_ids, values_log2p1)

    assert gene_ids.tolist() == ["ENSG00000000001", "ENSG00000000002", "ENSG00000000003"]
    np.testing.assert_allclose(resolved[0], _log2p1([100.0, 50.0]), atol=1e-5)
    np.testing.assert_allclose(resolved[1], _log2p1([20.0, 10.0]), atol=1e-5)
    # Differing duplicates must be summed in TPM space, not log2 space: log2(30+1)+log2(40+1)
    # != log2(70+1), so this also checks the aggregation happens before the log transform.
    np.testing.assert_allclose(resolved[2], _log2p1([70.0, 20.0]), atol=1e-5)

    assert policy["duplicate_stable_ids"] == ["ENSG00000000001", "ENSG00000000003"]
    assert policy["resolved_by_keep_identical_row"] == ["ENSG00000000001"]
    assert policy["resolved_by_tpm_space_sum"] == ["ENSG00000000003"]


def test_jax_parity_compare_detects_pass_and_mismatch(tmp_path):
    module = _load_script("audit_bulkrnabert_jax_parity.py")
    rng = np.random.default_rng(3)
    token_ids = rng.integers(0, 64, size=(1, 50)).astype(np.int64)
    subset_positions = np.arange(10)
    pooled = rng.normal(size=(4, 8)).astype(np.float32)
    subset = rng.normal(size=(4, 10, 8)).astype(np.float32)

    pytorch_path = tmp_path / "pytorch.npz"
    jax_path = tmp_path / "jax.npz"
    np.savez(pytorch_path, pooled=pooled, subset=subset, token_ids=token_ids, subset_positions=subset_positions)
    np.savez(jax_path, pooled=pooled, subset=subset, token_ids=token_ids, subset_positions=subset_positions)

    args = argparse.Namespace(
        pytorch_output=str(pytorch_path),
        jax_output=str(jax_path),
        report_output=str(tmp_path / "report.json"),
    )
    module.compare(args)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["all_layers_passed"] is True
    assert all(layer["pooled_cosine"] == pytest.approx(1.0) for layer in report["per_layer"])

    # A mismatched pooled embedding on one layer must be caught, not averaged away.
    pooled_bad = pooled.copy()
    pooled_bad[2] += 5.0
    np.savez(pytorch_path, pooled=pooled_bad, subset=subset, token_ids=token_ids, subset_positions=subset_positions)
    with pytest.raises(SystemExit):
        module.compare(args)
    report_bad = json.loads((tmp_path / "report.json").read_text())
    assert report_bad["all_layers_passed"] is False
    assert report_bad["per_layer"][2]["passed"] is False
    assert report_bad["per_layer"][0]["passed"] is True

    # Mismatched token_ids between the two backends must hard-fail before any metric is computed.
    np.savez(pytorch_path, pooled=pooled, subset=subset, token_ids=token_ids + 1, subset_positions=subset_positions)
    with pytest.raises(ValueError, match="token_ids differ"):
        module.compare(args)
