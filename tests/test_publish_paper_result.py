import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/publish_paper_result.py"


def module():
    spec = importlib.util.spec_from_file_location("_publish_test", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def test_unique_result_target():
    m = module()
    assert m.target_relative_path(
        {"study": "main", "arm": "main", "seed": 17}
    ) == Path("main/main/seed17.json")
    assert m.target_relative_path(
        {"study": "cpg_prior", "arm": "cpg_prior", "seed": None}
    ) == Path("cpg_prior/cpg_prior/seed_none.json")
