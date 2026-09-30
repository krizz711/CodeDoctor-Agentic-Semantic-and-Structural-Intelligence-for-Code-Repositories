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


def recipe_cache_key(encoder: object, text: str, mode: str = "document") -> str:
    """Content address of one vector under ``encoder``'s full recipe.

    Everything that changes the vector is in the key: model and revision, the
    exact prompt string handed to the tokenizer, the sequence cap, EOS and
    normalisation.  The ``mode`` namespace (``document`` by default, or
    ``query``) means a query can never be served from a document cache even if
    its text equals a document's.

    This is the one key for every document cache -- the scored path's corpus
    cache and the demo's -- so a model, revision or prefix change can never
    serve another recipe's vectors (it did in the demo until cycle 3, E3).
    Device and dtype are deliberately not in it: GPU and CPU vectors of the
    same recipe share an address (measured cosine >= 0.9994).  Document keys
    are byte-identical to what they were before ``mode`` existed.
    """
    if mode not in ("document", "query"):
        raise ValueError(f"mode must be 'document' or 'query', got {mode!r}")
    if hasattr(encoder, "get_prompt_text"):
        prompt = encoder.get_prompt_text(text, mode)
    else:
        prefix = getattr(encoder, "query_prefix" if mode == "query" else "document_prefix", "")
        prompt = f"{prefix}{text}"
    return compute_embedding_key(
        encoder_input=f"{mode}\x00" + prompt,
        encoder_revision=f"{getattr(encoder, 'model_name', '?')}@{getattr(encoder, 'revision', None)}",
        tokenizer_revision=(
            f"max{getattr(encoder, 'max_seq_length', '?')}"
            f"|eos{int(bool(getattr(encoder, 'add_eos_token', True)))}"
            f"|norm{int(bool(getattr(encoder, 'normalize_embeddings', True)))}"
        ),
        template_revision=repr(getattr(encoder, "document_prefix", "")),
    )


def stack_cached(vectors: list, expected_dim: int | None = None) -> np.ndarray:
    """``np.vstack`` for cached vectors that refuses mixed dimensions loudly."""
    dims = sorted({int(np.asarray(v).size) for v in vectors})
    if len(dims) > 1 or (expected_dim is not None and dims and dims[0] != expected_dim):
        raise ValueError(
            f"embedding cache returned vectors of dimension {dims}"
            + (f", encoder produces {expected_dim}" if expected_dim is not None else "")
            + "; the cache mixes recipes -- rebuild it or point it at a new file"
        )
    return np.vstack([np.asarray(v, dtype=np.float32) for v in vectors])


class EmbeddingCache:
    """
    Persistent embedding cache keyed by K_emb.
    Vectors are stored as float32 binary blobs with an is_real flag.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        # The live server and a warm-up script are meant to share this file:
        # one answers queries from it while the other fills it. Default SQLite
        # would fail the reader the moment the writer held the lock, so the
        # cache is opened in WAL with a real busy timeout instead.
        if str(db_path) != ":memory:":
            try:
                self.conn.execute("PRAGMA journal_mode=WAL")
                self.conn.execute("PRAGMA busy_timeout=30000")
                self.conn.execute("PRAGMA synchronous=NORMAL")
            except sqlite3.DatabaseError:  # pragma: no cover - exotic filesystems
                pass
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

    def get_many(self, cache_keys: list[str]) -> dict[str, np.ndarray]:
        """Real vectors for whichever of ``cache_keys`` are present."""
        found: dict[str, np.ndarray] = {}
        for start in range(0, len(cache_keys), 500):
            chunk = cache_keys[start : start + 500]
            placeholders = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT cache_key, vector, dim FROM embedding_cache "
                f"WHERE is_real = 1 AND cache_key IN ({placeholders})",
                chunk,
            ).fetchall()
            for row in rows:
                found[row["cache_key"]] = np.frombuffer(row["vector"], dtype=np.float32).reshape(row["dim"])
        return found

    def put_many(self, items: list[tuple[str, np.ndarray]]) -> None:
        """Insert many vectors in one transaction (one commit, not one per row)."""
        rows = []
        for cache_key, vector in items:
            arr = np.asarray(vector, dtype=np.float32)
            rows.append((cache_key, arr.size, arr.tobytes(), 1))
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache (cache_key, dim, vector, is_real) VALUES (?, ?, ?, ?)",
                rows,
            )

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

