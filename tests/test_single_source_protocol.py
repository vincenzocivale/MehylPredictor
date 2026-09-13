import json

import h5py
import numpy as np

from methylation_predictor.scopes import scope_protocol
from methylation_predictor.tcga_canonical import TCGACanonicalBundle


def test_single_source_bundle_and_protocol_do_not_require_array(tmp_path):
    (tmp_path / "rna").mkdir()
    (tmp_path / "methylation").mkdir()
    (tmp_path / "cpg" / "splits").mkdir(parents=True)
    with h5py.File(tmp_path / "rna" / "rna.h5", "w") as h:
        h["X"] = np.ones((2, 3), dtype="f4")
        h["sample_idx"] = np.array([10, 11], dtype="i8")
        h["gene_ids"] = np.array([b"a", b"b", b"c"])
    with h5py.File(tmp_path / "methylation" / "source.h5", "w") as h:
        h["X"] = np.ones((2, 4), dtype="f4")
        h["sample_idx"] = np.array([10, 11], dtype="i8")
        h["measurement_idx"] = np.array([0, 1], dtype="i8")
        h["cpg_idx"] = np.array([100, 101, 102, 103], dtype="i8")
    for name, values in {
        "train_sample_idx": [10], "val_sample_idx": [11],
        "train_cpg_idx": [100, 101], "val_cpg_idx": [102, 103],
    }.items():
        np.save(tmp_path / "cpg" / "splits" / f"{name}.npy", np.array(values, dtype="i8"))
    bundle = TCGACanonicalBundle.from_root(
        tmp_path, rna_file="rna/rna.h5", source_files={"encode_wgbs": "methylation/source.h5"},
        primary_source="encode_wgbs", expected_shapes={"rna": (2, 3), "encode_wgbs": (2, 4)},
    )
    try:
        protocol = scope_protocol("genomewide", bundle, canonical_root=tmp_path)
        assert protocol.primary_source == "encode_wgbs"
        assert np.array_equal(protocol.train_cpg_idx, [100, 101])
        assert list(protocol.evaluation_views())[0] == "train_cpg_x_val_sample"
        assert [p.name for p in protocol._build_pools()] == ["encode_wgbs"]
    finally:
        bundle.close()
