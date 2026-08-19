import json
from methylation_predictor.run_store import RunStore


def test_run_store_layout(tmp_path):
    store=RunStore.create(tmp_path,model="rna_methylation",train_scope="chr123",seed=17,learning_rate=5e-5,scheduler="constant",epochs=80,run_id="test")
    store.save_resolved_config({"training":{"epochs":80}}); store.save_metadata({"training":{"epochs":80}})
    assert (store.path/"checkpoints").is_dir()
    assert (store.path/"evaluation").is_dir()
    meta=json.loads((store.path/"metadata.json").read_text())
    assert meta["model"]=="rna_methylation" and meta["training_scope"]=="chr123"
