from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3
from typing import Optional
import numpy as np


def compute_embedding_key(
    encoder_input: str,
    encoder_revision: str = "v1",
    tokenizer_revision: str = "v1",
    template_revision: str = "v1",
) -> str:
    """
    Computes the invariant embedding cache key:
    K_emb = H( encoder_revision, tokenizer_revision, template_revision, encoder_input_bytes )

    PATH AND LINE NUMBER ARE DELIBERATELY EXCLUDED.
    A function that moves keeps its vector; a function whose body changes does not.
    """
    hasher = hashlib.sha256()
    hasher.update(encoder_revision.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(tokenizer_revision.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(template_revision.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(encoder_input.encode("utf-8"))
    return hasher.hexdigest()


class EmbeddingCache:
    """
    Persistent embedding cache keyed by K_emb.
    Vectors are stored as float32 binary blobs with an is_real flag.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                cache_key TEXT PRIMARY KEY,
                dim INTEGER,
                vector BLOB,
                is_real INTEGER DEFAULT 1
            )
            """
        )
        try:
            self.conn.execute("ALTER TABLE embedding_cache ADD COLUMN is_real INTEGER DEFAULT 1")
        except sqlite3.OperationalError:
            pass
        self.conn.commit()

    def get(self, cache_key: str, real_only: bool = True) -> Optional[np.ndarray]:
        if real_only:
            row = self.conn.execute(
                "SELECT vector, dim FROM embedding_cache WHERE cache_key = ? AND is_real = 1",
                (cache_key,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT vector, dim FROM embedding_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if not row:
            return None
        blob = row["vector"]
        dim = row["dim"]
        return np.frombuffer(blob, dtype=np.float32).reshape(dim)

    def put(self, cache_key: str, vector: np.ndarray, is_real: bool = True) -> None:
        arr = np.asarray(vector, dtype=np.float32)
        blob = arr.tobytes()
        dim = arr.size
        is_real_val = 1 if is_real else 0
        self.conn.execute(
            """
            INSERT OR REPLACE INTO embedding_cache (cache_key, dim, vector, is_real)
            VALUES (?, ?, ?, ?)
            """,
            (cache_key, dim, blob, is_real_val),
        )
        self.conn.commit()

    def has(self, cache_key: str, real_only: bool = True) -> bool:
        if real_only:
            row = self.conn.execute(
                "SELECT 1 FROM embedding_cache WHERE cache_key = ? AND is_real = 1",
                (cache_key,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT 1 FROM embedding_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        return row is not None

    def count(self, real_only: bool = False) -> int:
        if real_only:
            row = self.conn.execute("SELECT COUNT(*) FROM embedding_cache WHERE is_real = 1").fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()
        return row[0] if row else 0

    def all_keys(self) -> list[str]:
        rows = self.conn.execute("SELECT cache_key FROM embedding_cache").fetchall()
        return [r[0] for r in rows]

    def clear(self) -> None:
        self.conn.execute("DELETE FROM embedding_cache")
        self.conn.commit()

