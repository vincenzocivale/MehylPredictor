from types import SimpleNamespace
import json
import numpy as np
import torch


def test_genomewide_inference_is_chromosome_chunked(tmp_path, monkeypatch):
    from methylation_predictor.rna_training import inference

    root = tmp_path / "store"
    for chrom, n in (("chr1", 3), ("chr2", 2)):
        shard = root / "features" / chrom
        shard.mkdir(parents=True)
        np.save(shard / "locus_key.npy", np.arange(1, n + 1, dtype=np.uint64))
        np.save(shard / "position.npy", np.arange(10, 10 + n, dtype=np.int64))
    (root / "manifest.json").write_text('{"status":"complete"}\n')

    class FakeRNA:
        values = np.zeros((2, 4), dtype=np.float16)
        index = SimpleNamespace(positions_of=lambda ids: np.arange(len(ids)))
        def rows(self, ids):
            return np.asarray([[float(i)] * 4 for i in ids], dtype=np.float32)

    class FakeStore:
        def __init__(self, root):
            pass
        def lookup_full(self, keys):
            n = len(keys)
            return (np.zeros((n, 18), np.float32), np.zeros((n, 5), np.float32), np.zeros((n, 521), np.uint8))

    class FakeModel(torch.nn.Module):
        def forward(self, rna, **kwargs):
            n = kwargs["functional_dense"].shape[0]
            return {"beta": torch.sigmoid(rna[:, :1].expand(-1, n))}

    monkeypatch.setattr(inference, "LocusFeatureStore", FakeStore)
    monkeypatch.setattr(inference, "RNACache", lambda _: FakeRNA())
    monkeypatch.setattr(inference, "load_rna_recipe", lambda _: SimpleNamespace(model=SimpleNamespace(functional_fusion_variant="fake"), raw={"locus_cls": {}}))
    monkeypatch.setattr(inference, "build_functional_predictor", lambda **_: (FakeModel(), "fake"))
    ckpt = tmp_path / "model.pt"
    torch.save({"model_state": {}, "locus_cls": {"functional_locus": {"n_tracks": 4165, "dense_dim": 23}, "use_mean_branch": True}}, ckpt)
    result = inference.infer_tcga_genomewide(checkpoint=ckpt, recipe="recipe.yaml", locus_store=root, rna_cache="rna", output=tmp_path / "out", sample_ids=np.array([0, 1]), chromosomes=("chr1", "chr2"), sample_chunk=1, locus_chunk=1, device="cpu")
    assert result["status"] == "complete"
    assert np.load(tmp_path / "out/chr1.prediction.f32.npy").shape == (2, 3)
    assert np.load(tmp_path / "out/chr2.prediction.f32.npy").shape == (2, 2)
    assert json.loads((tmp_path / "out/chr1.manifest.json").read_text())["status"] == "complete"
    resumed = inference.infer_tcga_genomewide(checkpoint=ckpt, recipe="recipe.yaml", locus_store=root, rna_cache="rna", output=tmp_path / "out", sample_ids=np.array([0, 1]), chromosomes=("chr1", "chr2"), sample_chunk=1, locus_chunk=1, device="cpu")
    assert len(resumed["chromosomes"]) == 2
    import pytest
    with pytest.raises(ValueError, match="incompatible"):
        inference.infer_tcga_genomewide(checkpoint=ckpt, recipe="recipe.yaml", locus_store=root, rna_cache="rna", output=tmp_path / "out", sample_ids=np.array([0, 1]), chromosomes=("chr1",), sample_chunk=2, locus_chunk=1, device="cpu")
