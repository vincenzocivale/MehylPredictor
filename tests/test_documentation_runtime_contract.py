from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CURRENT_DOCS = [
    ROOT / "README.md",
    ROOT / "PROJECT_SPECS.md",
    ROOT / "docs" / "MODEL.md",
    ROOT / "docs" / "RNA_METHYLATION.md",
    ROOT / "docs" / "WORKFLOWS.md",
    ROOT / "docs" / "CPG_STATISTICS.md",
]


def _joined_current_docs() -> str:
    return "\n".join(path.read_text() for path in CURRENT_DOCS)


def test_current_docs_do_not_describe_ntv3_as_live_rna_model_input():
    text = _joined_current_docs()

    stale_live_claims = [
        "frozen 1,536-D NTv3 embedding e_l",
        "For the current single-stage model, embeddings are required",
        "protocol-aligned RNA/embedding caches",
        "single-stage shared-backbone model in",
    ]

    offenders = [claim for claim in stale_live_claims if claim in text]
    assert offenders == []


def test_current_docs_state_functional_locus_runtime_contract():
    specs = (ROOT / "PROJECT_SPECS.md").read_text()
    workflows = (ROOT / "docs" / "WORKFLOWS.md").read_text()

    assert "functional regulatory atlas" in specs
    assert "The RNA workflow does not load genomic/FM embeddings." in specs
    assert "--prior-cache" in specs

    assert "Functional atlas and annotation caches are mandatory." in workflows
    assert "The genomic/FM feature cache is not a model input." in workflows


def test_historical_benchmark_doc_may_keep_ntv3_provenance_but_labels_it_historical():
    text = (ROOT / "docs" / "BENCHMARK_METHYLPROPHET.md").read_text()

    assert "## Historical Table-5 genomic prior" in text
    assert "current RNA predictor does not consume the NTv3 embedding" in text
