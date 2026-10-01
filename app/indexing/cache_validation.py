"""Validation of cached vectors without changing the retrieval recipe."""

from __future__ import annotations

import hashlib
import sqlite3

import numpy as np


APPS_DATASET = {"path": "CoIR-Retrieval/apps", "revision": "f22508f96b7a36c2415181ed8bb76f76e04ae2d5"}
APPS_DOCUMENTS = 8765
APPS_UNIQUE_DOCUMENTS = 8754
APPS_TEST_QUERIES = 3765
APPS_DIMENSION = 2048


def text_fingerprint(texts: list[str]) -> str:
    """The ordered text fingerprint used by prepare_document_cache.py."""
    return hashlib.sha256("\n".join(hashlib.sha256(t.encode("utf-8")).hexdigest()
                                    for t in texts).encode("ascii")).hexdigest()


def validate_vector(vector, dimension: int, *, normalized: bool = True, label: str = "vector") -> None:
    value = np.asarray(vector)
    if value.shape != (dimension,):
        raise ValueError(f"{label}: expected dimension {dimension}, got shape {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{label}: non-finite values")
    norm = float(np.linalg.norm(value.astype(np.float64)))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"{label}: zero or non-finite norm")
    if normalized and abs(norm - 1.0) > 0.001:
        raise ValueError(f"{label}: expected a normalized vector, norm={norm}")


def validate_store(connection: sqlite3.Connection, *, dimension: int,
                   expected_keys: set[str] | None = None, normalized: bool = True,
                   shard: tuple[int, int] | None = None) -> set[str]:
    """Reject corrupt vectors and, optionally, incomplete or misassigned keys."""
    keys = set()
    for key, dim, blob, is_real in connection.execute(
            "SELECT cache_key, dim, vector, is_real FROM embedding_cache"):
        if is_real != 1:
            raise ValueError(f"cache contains a non-real vector: {key}")
        if dim != dimension or not isinstance(blob, bytes) or len(blob) != dimension * 4:
            raise ValueError(f"cache vector {key}: expected dimension {dimension} and matching float32 bytes")
        validate_vector(np.frombuffer(blob, dtype=np.float32), dimension,
                        normalized=normalized, label=f"cache vector {key}")
        if shard is not None:
            index, count = shard
            if not 0 <= index < count:
                raise ValueError(f"invalid shard {shard}")
            if int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16) % count != index:
                raise ValueError(f"cache key {key} does not belong to declared shard {index}/{count}")
        keys.add(key)
    if expected_keys is not None and keys != expected_keys:
        raise ValueError(f"cache key coverage mismatch: {len(expected_keys - keys)} missing, "
                         f"{len(keys - expected_keys)} unexpected")
    return keys
