from pathlib import Path
import time

from methylation_predictor.experiment_coordination import (
    SharedExperimentRegistry,
)


def registry(tmp_path: Path, stale=60):
    return SharedExperimentRegistry(
        data_root=tmp_path,
        campaign="test",
        relative_root="experiments/coordination/test",
        claim_stale_after_seconds=stale,
    )


def test_atomic_single_claim(tmp_path):
    r = registry(tmp_path)
    a = r.try_claim("main/main/seed17", owner={"host": "a", "pid": 1, "gpu": "0"})
    assert a is not None
    b = r.try_claim("main/main/seed17", owner={"host": "b", "pid": 2, "gpu": "0"})
    assert b is None
    a.release()


def test_stale_claim_is_reclaimable(tmp_path):
    r = registry(tmp_path, stale=0.01)
    a = r.try_claim("job", owner={"host": "a", "pid": 1, "gpu": "0"})
    assert a is not None
    time.sleep(0.03)
    b = r.try_claim("job", owner={"host": "b", "pid": 2, "gpu": "0"})
    assert b is not None
    assert b.reclaimed_stale is True
    b.release()


def test_snapshot_counts_states(tmp_path):
    r = registry(tmp_path)
    catalog = [
        {"job_key": "a", "study": "s", "arm": "a"},
        {"job_key": "b", "study": "s", "arm": "b"},
        {"job_key": "c", "study": "s", "arm": "c"},
    ]
    r.write_state("b", status="running", catalog_entry=catalog[1])
    r.write_state("c", status="completed", catalog_entry=catalog[2])
    snap = r.refresh_snapshot(catalog)
    assert snap["summary"]["by_status"] == {
        "pending": 1,
        "running": 1,
        "completed": 1,
    }
