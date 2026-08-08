from __future__ import annotations

import pytest

from methylation_predictor.tcga_canonical import TCGACanonicalBundle, resolve_bundle_root


@pytest.fixture(scope="session")
def bundle_root():
    root = resolve_bundle_root()
    if not root.is_dir():
        pytest.skip(f"canonical TCGA bundle not available at {root}")
    return root


@pytest.fixture(scope="session")
def bundle(bundle_root):
    handle = TCGACanonicalBundle.from_root(bundle_root)
    yield handle
    handle.close()
