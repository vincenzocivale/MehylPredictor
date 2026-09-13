#!/usr/bin/env bash
# Mirror the ENCODE-study data this machine needs from the slow SFTP-mounted
# dune_data/ (METHYL_DATA_ROOT) to fast local disk under local_methyl_data/,
# so training reads (repeated every epoch) don't hit the network mount.
#
# Re-run any time dune_data/ changes (e.g. after the shared locus_features_v1
# store is rebuilt, or the ENCODE bundle is regenerated) -- rsync only
# transfers deltas, so repeat syncs are cheap.
#
# local_methyl_data/ mirrors the same relative layout as dune_data/, so any
# path under dune_data/... can be swapped for local_methyl_data/... as a
# drop-in replacement (e.g. --locus-store, --rna-cache, --canonical-root).
set -euo pipefail
cd "$(dirname "$0")/.."

rsync -a --stats \
  dune_data/datasets/methylprophet_encode_v1/ local_methyl_data/datasets/methylprophet_encode_v1/

rsync -a --stats --exclude='*.bak_pre_encode' \
  dune_data/derived/locus_features_v1/ local_methyl_data/derived/locus_features_v1/

echo "sync complete: $(date -u +%FT%TZ)"
