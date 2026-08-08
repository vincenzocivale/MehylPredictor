"""Clean data/protocol layer for MethylProphet-compatible TCGA training.

    from methylation_predictor.tcga_canonical import TCGACanonicalBundle, load_protocol

    bundle = TCGACanonicalBundle.from_root(root)
    protocol = load_protocol("tcga_mix_chr1", bundle=bundle)
    train_dataset = protocol.train_dataset(batch_sample_size=32, batch_cpg_size=256)
    batch = train_dataset.sample_batch()
    views = protocol.evaluation_views()

See docs/data/TCGA_CANONICAL_DATA.md and docs/data/METHYLPROPHET_PROTOCOLS.md.
"""
# pyarrow (parquet CpG-index files) must be imported before h5py: h5py's
# bundled HDF5 wheel ships its own libstdc++, and if it loads first it shadows
# the newer libstdc++ symbol version pyarrow needs (GLIBCXX_3.4.31), breaking
# every parquet read for the rest of the process (pandas only imports pyarrow
# lazily on first read_parquet call, so importing pandas alone isn't enough).
# Harmless if pyarrow is never installed; import order here just makes both
# safe together when it is.
try:
    import pyarrow as _pyarrow  # noqa: F401
except ImportError:
    pass

from .batch import RNAFeatures, TrainingBatch
from .bundle import MethylationSource, RNASource, TCGACanonicalBundle
from .config import ProtocolRunConfig, load_protocol_run_config, resolve_bundle_root
from .protocol import (
    EvaluationView,
    KNOWN_PROTOCOLS,
    Protocol,
    ProtocolTrainDataset,
    SOURCE_SAMPLING_POLICIES,
    load_protocol,
)
from .sampler import BalancedPairSampler, SourceSamplingPool

__all__ = [
    "TCGACanonicalBundle",
    "resolve_bundle_root",
    "ProtocolRunConfig",
    "load_protocol_run_config",
    "SOURCE_SAMPLING_POLICIES",
    "RNASource",
    "MethylationSource",
    "RNAFeatures",
    "TrainingBatch",
    "Protocol",
    "ProtocolTrainDataset",
    "EvaluationView",
    "KNOWN_PROTOCOLS",
    "load_protocol",
    "BalancedPairSampler",
    "SourceSamplingPool",
]
