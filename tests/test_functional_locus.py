from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from methylation_predictor.config import EncoderConfig, ModelConfig
from methylation_predictor.modeling import EfficientSingleAttentionPredictor
from methylation_predictor.storage import FunctionalLocusCache, open_functional_locus_cache


def _write_cache(tmp_path, *, mismatch=False):
    atlas = tmp_path / "atlas"
    annotations = tmp_path / "annotations"
    csr = atlas / "all_primary_overlap_csr"
    csr.mkdir(parents=True)
    annotations.mkdir()

    ids = np.array([91, 12, 45, 77], dtype=np.int64)
    np.save(atlas / "cpg_idx.npy", ids)
    np.save(annotations / "cpg_idx.npy", ids + (1 if mismatch else 0))

    core = np.arange(4 * 18, dtype=np.float32).reshape(4, 18)
    breadth = (100 + np.arange(4 * 5, dtype=np.float32)).reshape(4, 5)
    np.save(annotations / "annotation_core.f32.npy", core)
    np.save(atlas / "breadth_features.f32.npy", breadth)

    # rows: [2,7], [], [1], [3,4,8]
    np.save(
        csr / "indices.npy",
        np.array([2, 7, 1, 3, 4, 8], dtype=np.int32),
    )
    np.save(
        csr / "indptr.npy",
        np.array([0, 2, 2, 3, 6], dtype=np.int64),
    )
    np.save(csr / "data.npy", np.ones(6, dtype=np.uint8))
    (csr / "shape.json").write_text(json.dumps([4, 4165]))

    return atlas, annotations, ids, core, breadth


def test_functional_cache_arbitrary_order_and_empty_bags(tmp_path):
    atlas, annotations, ids, core, breadth = _write_cache(tmp_path)
    cache = FunctionalLocusCache(atlas, annotations)

    result = cache.get(
        np.array([ids[3], ids[1], ids[0]], dtype=np.int64)
    )

    assert result["track_indices"].tolist() == [3, 4, 8, 2, 7]
    assert result["offsets"].tolist() == [0, 3, 3, 5]
    np.testing.assert_array_equal(
        result["dense"][:, :18], core[[3, 1, 0]]
    )
    np.testing.assert_array_equal(
        result["dense"][:, 18:], breadth[[3, 1, 0]]
    )

    bag = torch.nn.EmbeddingBag(
        4165,
        4,
        mode="mean",
        include_last_offset=True,
    )
    encoded = bag(
        torch.from_numpy(result["track_indices"]),
        torch.from_numpy(result["offsets"]),
    )
    assert torch.equal(encoded[1], torch.zeros_like(encoded[1]))


def test_backend_factory_rejects_ambiguous_or_missing_roots(tmp_path):
    atlas, annotations, *_ = _write_cache(tmp_path)
    assert isinstance(open_functional_locus_cache(functional_atlas=atlas, annotation_cache=annotations), FunctionalLocusCache)
    with pytest.raises(ValueError, match="provide --locus-store"):
        open_functional_locus_cache()
    with pytest.raises(ValueError, match="cannot be combined"):
        open_functional_locus_cache(locus_store=tmp_path, functional_atlas=atlas)


def test_functional_cache_rejects_axis_mismatch(tmp_path):
    atlas, annotations, *_ = _write_cache(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="do not match exactly"):
        FunctionalLocusCache(atlas, annotations)


def test_functional_cache_reads_common_locus_store(tmp_path):
    import pandas as pd
    from methylation_predictor.locus_features.ids import locus_key

    atlas, annotations, ids, core, breadth = _write_cache(tmp_path)
    root = tmp_path / "store"
    shard = root / "features" / "chr1"
    shard.mkdir(parents=True)
    (root / "aliases").mkdir()
    keys = np.array([locus_key("chr1", 100 + i) for i in range(4)], dtype=np.uint64)
    np.save(shard / "locus_key.npy", keys)
    np.save(shard / "annotation_core.f32.npy", core)
    import hashlib
    contract = __import__('pathlib').Path(__file__).resolve().parents[1] / "resources/locus_features/annotation_core_v1.json"
    (shard / "manifest.json").write_text(json.dumps({"status": "complete", "feature_dimension": 18, "annotation_schema_sha256": hashlib.sha256(contract.read_bytes()).hexdigest()}))
    (root / "manifest.json").write_text(json.dumps({"status": "complete"}))
    (root / "validation_chr1.json").write_text(json.dumps({"status": "complete"}))
    aliases = pd.DataFrame({"cpg_idx": ids, "locus_key": keys})
    for tech in ("array", "epic", "wgbs"):
        aliases.to_parquet(root / "aliases" / f"tcga_{tech}.parquet", index=False)
    result = FunctionalLocusCache(atlas, root).get(np.array([ids[3], ids[0]]))
    np.testing.assert_array_equal(result["dense"][:, :18], core[[3, 0]])
    np.testing.assert_array_equal(result["dense"][:, 18:], breadth[[3, 0]])


def test_functional_cache_reads_packed_regulatory_backend(tmp_path):
    import hashlib
    import pandas as pd
    from pathlib import Path
    from methylation_predictor.locus_features.ids import locus_key

    atlas, _, ids, core, breadth = _write_cache(tmp_path)
    root=tmp_path/'full_store';shard=root/'features'/'chr1';shard.mkdir(parents=True)
    (root/'aliases').mkdir()
    keys=np.array([locus_key('chr1',100+i) for i in range(4)],dtype=np.uint64)
    np.save(shard/'locus_key.npy',keys)
    np.save(shard/'annotation_core.f32.npy',core)
    np.save(shard/'breadth.f32.npy',breadth)
    bits=np.zeros((4,4165),np.uint8);bits[0,[2,7]]=1;bits[2,[1]]=1;bits[3,[3,4,8]]=1
    np.save(shard/'regulatory.packbits.npy',np.packbits(bits,axis=1,bitorder='big'))
    contract=Path(__file__).resolve().parents[1]/'resources/locus_features/annotation_core_v1.json'
    (shard/'manifest.json').write_text(json.dumps({'status':'complete','feature_dimension':18,'annotation_schema_sha256':hashlib.sha256(contract.read_bytes()).hexdigest(),'regulatory_status':'complete','regulatory_tracks':4165,'regulatory_track_contract_sha256':'fixture','regulatory_packbits_bitorder':'big'}))
    (root/'regulatory').mkdir()
    (root/'regulatory'/'manifest.json').write_text(json.dumps({'status':'complete','verified':4165,'track_contract_sha256':'fixture'}))
    (root/'manifest.json').write_text(json.dumps({'status':'complete','regulatory_status':'complete','regulatory_track_contract_sha256':'fixture'}))
    (root/'validation_regulatory_chr1.json').write_text(json.dumps({'status':'complete'}))
    aliases=pd.DataFrame({'cpg_idx':ids,'locus_key':keys})
    for tech in ('array','epic','wgbs'):
        aliases.to_parquet(root/'aliases'/f'tcga_{tech}.parquet',index=False)
    result=FunctionalLocusCache(atlas,root).get(np.array([ids[3],ids[1],ids[0],ids[3]]))
    assert result['track_indices'].tolist()==[3,4,8,2,7,3,4,8]
    assert result['offsets'].tolist()==[0,3,3,5,8]
    np.testing.assert_array_equal(result['dense'],np.concatenate((core[[3,1,0,3]],breadth[[3,1,0,3]]),axis=1))


def test_legacy_and_canonical_backends_are_forward_equivalent(tmp_path):
    import hashlib
    import pandas as pd
    from pathlib import Path
    from methylation_predictor.locus_features.ids import locus_key

    atlas, annotations, ids, core, breadth = _write_cache(tmp_path)
    legacy = FunctionalLocusCache(atlas, annotations)
    root = tmp_path / "store"; shard = root / "features" / "chr1"; shard.mkdir(parents=True)
    (root / "aliases").mkdir()
    keys = np.array([locus_key("chr1", 100 + i) for i in range(4)], dtype=np.uint64)
    np.save(shard / "locus_key.npy", keys); np.save(shard / "annotation_core.f32.npy", core); np.save(shard / "breadth.f32.npy", breadth)
    bits = np.zeros((4, 4165), np.uint8); bits[0, [2, 7]] = 1; bits[2, [1]] = 1; bits[3, [3, 4, 8]] = 1
    np.save(shard / "regulatory.packbits.npy", np.packbits(bits, axis=1, bitorder="big"))
    contract = Path(__file__).resolve().parents[1] / "resources/locus_features/annotation_core_v1.json"
    digest = hashlib.sha256(contract.read_bytes()).hexdigest()
    (shard / "manifest.json").write_text(json.dumps({"status":"complete", "feature_dimension":18, "annotation_schema_sha256":digest, "regulatory_status":"complete", "regulatory_tracks":4165, "regulatory_track_contract_sha256":"fixture", "regulatory_packbits_bitorder":"big"}))
    (root / "regulatory").mkdir(); (root / "regulatory/manifest.json").write_text(json.dumps({"status":"complete", "verified":4165, "track_contract_sha256":"fixture"}))
    (root / "manifest.json").write_text(json.dumps({"status":"complete", "regulatory_status":"complete", "regulatory_track_contract_sha256":"fixture"}))
    (root / "validation_regulatory_chr1.json").write_text(json.dumps({"status":"complete"}))
    aliases = pd.DataFrame({"cpg_idx": ids, "locus_key": keys})
    for tech in ("array", "epic", "wgbs"): aliases.to_parquet(root / "aliases" / f"tcga_{tech}.parquet", index=False)
    canonical = FunctionalLocusCache(locus_store=root)
    query = np.array([ids[3], ids[0], ids[2]], dtype=np.int64)
    cfg = ModelConfig(encoder=EncoderConfig(kind="locus_attention", latent_dim=8, n_programs=6, program_dim=256, n_heads=4, dropout=0.0))
    model = EfficientSingleAttentionPredictor(48, cfg, final_regressor_dropout=0.0).eval()
    legacy_f = legacy.get(query); canonical_f = canonical.get(query)
    x = torch.randn(2, 48)
    with torch.no_grad():
        old = model(x, functional_track_indices=torch.from_numpy(legacy_f["track_indices"]), functional_offsets=torch.from_numpy(legacy_f["offsets"]), functional_dense=torch.from_numpy(legacy_f["dense"]))["beta"]
        new = model(x, functional_track_indices=torch.from_numpy(canonical_f["track_indices"]), functional_offsets=torch.from_numpy(canonical_f["offsets"]), functional_dense=torch.from_numpy(canonical_f["dense"]))["beta"]
    torch.testing.assert_close(old, new, rtol=0.0, atol=1e-6)


def test_cache_output_feeds_paper_candidate_without_dense_track_matrix(tmp_path):
    atlas, annotations, ids, *_ = _write_cache(tmp_path)
    cache = FunctionalLocusCache(atlas, annotations)

    requested = np.array([ids[3], ids[1], ids[0]], dtype=np.int64)
    functional = cache.get(requested)

    cfg = ModelConfig(
        encoder=EncoderConfig(
            kind="locus_attention",
            latent_dim=8,
            n_programs=6,
            program_dim=256,
            n_heads=4,
            dropout=0.0,
        ),
    )
    model = EfficientSingleAttentionPredictor(
        48,
        cfg,
        final_regressor_dropout=0.15,
    ).eval()

    with torch.no_grad():
        out = model(
            torch.randn(2, 48),
            functional_track_indices=torch.from_numpy(
                functional["track_indices"]
            ),
            functional_offsets=torch.from_numpy(
                functional["offsets"]
            ),
            functional_dense=torch.from_numpy(
                functional["dense"]
            ),
        )

    assert out["beta"].shape == (2, 3)
    assert torch.isfinite(out["beta"]).all()
