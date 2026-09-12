"""Portable ``methyl-data://`` artifact URI resolver.

Curated paper results (``results/paper/**``) must never embed machine-specific
absolute paths (``/dune/...``, ``/data2/...``, ``/raid/...``). Instead they
carry a portable ``methyl-data://<relative-path>`` reference that each machine
resolves against its own ``METHYL_DATA_ROOT``.

See ``configs/data/storage_v2.yaml`` (``artifact_uri`` section) for the frozen
contract this module implements.
"""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

SCHEME = "methyl-data"
_PREFIX = f"{SCHEME}://"


def _data_root(data_root: str | Path | None) -> Path:
    raw = data_root if data_root is not None else os.environ.get("METHYL_DATA_ROOT")
    if not raw:
        raise ValueError(
            "data_root is required unless METHYL_DATA_ROOT is set"
        )
    root = Path(raw).expanduser().resolve()
    return root


def _normalize_relative(relative: str) -> str:
    """Validate and normalize a URI's relative-path portion.

    Rejects absolute paths, empty segments after normalization, and any
    ``..`` traversal component.
    """
    if relative == "":
        raise ValueError("methyl-data:// URI is missing a path")
    if relative.startswith("/"):
        raise ValueError(f"methyl-data:// path must be relative, got {relative!r}")

    posix = PurePosixPath(relative)
    parts: list[str] = []
    for part in posix.parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError(f"path traversal is not allowed: {relative!r}")
        parts.append(part)
    if not parts:
        raise ValueError(f"methyl-data:// path is empty after normalization: {relative!r}")
    return "/".join(parts)


def to_uri(path: str | Path, *, data_root: str | Path | None = None) -> str:
    """Convert a local path under ``METHYL_DATA_ROOT`` into a ``methyl-data://`` URI."""
    root = _data_root(data_root)
    resolved = Path(path).expanduser().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{resolved} is not located under the data root {root}"
        ) from exc
    normalized = _normalize_relative(relative.as_posix())
    return f"{_PREFIX}{normalized}"


def from_uri(uri: str, *, data_root: str | Path | None = None) -> Path:
    """Resolve a ``methyl-data://`` URI into a local path under ``METHYL_DATA_ROOT``."""
    if not uri.startswith(_PREFIX):
        raise ValueError(f"malformed methyl-data URI (bad scheme): {uri!r}")
    relative = uri[len(_PREFIX):]
    normalized = _normalize_relative(relative)
    root = _data_root(data_root)
    resolved = (root / normalized).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:  # defense in depth; _normalize_relative already blocks this
        raise ValueError(f"{uri} escapes the data root {root}") from exc
    return resolved


def is_uri(value: str) -> bool:
    return isinstance(value, str) and value.startswith(_PREFIX)
