from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Optional, Sequence
import numpy as np

from app.indexing.cache import EmbeddingCache, compute_embedding_key
from app.indexing.sqlite_index import SQLiteIndex
from app.parser.chunker import Chunk, chunk_source
from app.retrieval.encode import CodeEncoder, get_shared_encoder

SUPPORTED_EXTENSIONS = {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".py"}
IGNORED_DIRS = {".git", "node_modules", "dist", "build", "coverage", ".cache", ".idea", ".vscode"}


@dataclass
class IncrementalStats:
    chunks_total: int = 0
    embeddings_total: int = 0
    changed_chunks: int = 0
    re_embedded: int = 0
    reused: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_deleted: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "chunks_total": self.chunks_total,
            "embeddings_total": self.embeddings_total,
            "changed_chunks": self.changed_chunks,
            "re_embedded": self.re_embedded,
            "reused": self.reused,
            "files_indexed": self.files_indexed,
            "files_skipped": self.files_skipped,
            "files_deleted": self.files_deleted,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
        }


class IncrementalIndexer:
    def __init__(
        self,
        index: SQLiteIndex,
        cache: Optional[EmbeddingCache] = None,
        encoder: Optional[CodeEncoder] = None,
    ) -> None:
        self.index = index
        self.cache = cache or EmbeddingCache(db_path=":memory:")
        self.encoder = encoder
        self.last_stats = IncrementalStats()

    def _get_encoder(self) -> CodeEncoder:
        if self.encoder is None:
            self.encoder = get_shared_encoder()
        return self.encoder

    def _collect_disk_files(self, root: Path) -> dict[str, tuple[Path, str]]:
        """Walk root, returning {rel_path: (abs_path, sha256_hash)}."""
        disk_files: dict[str, tuple[Path, str]] = {}
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in IGNORED_DIRS for part in path.parts):
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                rel = str(path.relative_to(root)).replace("\\", "/")
                content_bytes = path.read_bytes()
                f_hash = hashlib.sha256(content_bytes).hexdigest()
                disk_files[rel] = (path, f_hash)
            except (OSError, UnicodeDecodeError):
                continue
        return disk_files

    def index_repository(
        self,
        repo_path: str | Path,
        commit_hash: str = "HEAD",
        compute_embeddings: bool = True,
        force_rebuild: bool = False,
    ) -> IncrementalStats:
        start_time = time.perf_counter()
        root = Path(repo_path).resolve()
        stats = IncrementalStats()

        if force_rebuild:
            # Clear all current files/chunks in index
            self.index.conn.execute("DELETE FROM files")
            self.index.conn.execute("DELETE FROM chunks")
            self.index.conn.execute("DELETE FROM symbols")
            self.index.conn.execute("DELETE FROM chunks_fts")
            self.index.conn.commit()

        # Query existing indexed files from DB: {rel_path: (file_id, content_hash)}
        indexed_rows = self.index.conn.execute(
            "SELECT id, path, content_hash FROM files"
        ).fetchall()
        db_files: dict[str, tuple[int, str]] = {
            row["path"]: (row["id"], row["content_hash"]) for row in indexed_rows
        }

        disk_files = self._collect_disk_files(root)

        # 1. Handle deleted files
        for rel_path, (file_id, _) in db_files.items():
            if rel_path not in disk_files:
                # Remove from database
                chunk_ids = [
                    r["id"]
                    for r in self.index.conn.execute(
                        "SELECT id FROM chunks WHERE file_id = ?", (file_id,)
                    ).fetchall()
                ]
                for cid in chunk_ids:
                    self.index.conn.execute("DELETE FROM chunks WHERE id = ?", (cid,))
                    self.index.conn.execute("DELETE FROM symbols WHERE chunk_id = ?", (cid,))
                    self.index.conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (cid,))
                self.index.conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
                stats.files_deleted += 1

        self.index.conn.commit()

        # 2. Process current disk files
        for rel_path, (abs_path, f_hash) in disk_files.items():
            db_entry = db_files.get(rel_path)
            if db_entry and db_entry[1] == f_hash and not force_rebuild:
                file_id = db_entry[0]
                can_skip = True
                if compute_embeddings:
                    chunk_rows = self.index.conn.execute(
                        "SELECT signature, content FROM chunks WHERE file_id = ?", (file_id,)
                    ).fetchall()
                    for crow in chunk_rows:
                        sig = crow["signature"]
                        cnt = crow["content"]
                        enc_in = f"{sig}\n{cnt}" if sig else cnt
                        ckey = compute_embedding_key(enc_in)
                        if not self.cache.has(ckey, real_only=True):
                            can_skip = False
                            break

                if can_skip:
                    stats.files_skipped += 1
                    existing_chunks_count = self.index.conn.execute(
                        "SELECT COUNT(*) FROM chunks WHERE file_id = ?", (file_id,)
                    ).fetchone()[0]
                    stats.reused += existing_chunks_count
                    continue

            # Read source
            try:
                raw_bytes = abs_path.read_bytes()
                source = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                source = raw_bytes.decode("latin-1")

            # Check previous chunk hashes for this file if it existed
            old_chunk_hashes: set[str] = set()
            if db_entry:
                old_chunk_hashes = {
                    r["content_hash"]
                    for r in self.index.conn.execute(
                        "SELECT content_hash FROM chunks WHERE file_id = ?", (db_entry[0],)
                    ).fetchall()
                    if r["content_hash"]
                }

            # Index file in SQLiteIndex
            lang = "python" if abs_path.suffix.lower() == ".py" else "javascript"
            chunks = self.index.index_file(
                file_path=rel_path,
                source=source,
                commit_hash=commit_hash,
                language=lang,
            )
            stats.files_indexed += 1

            for chunk in chunks:
                chunk_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
                if chunk_hash not in old_chunk_hashes and db_entry:
                    stats.changed_chunks += 1

                # Invariant embedding cache key: path and line number excluded!
                encoder_input = f"{chunk.signature}\n{chunk.content}" if chunk.signature else chunk.content
                cache_key = compute_embedding_key(encoder_input)

                if compute_embeddings:
                    if self.cache.has(cache_key, real_only=True):
                        stats.reused += 1
                    else:
                        encoder = self._get_encoder()
                        vector = encoder.encode(encoder_input, mode="document")
                        self.cache.put(cache_key, vector, is_real=True)
                        stats.re_embedded += 1
                else:
                    if self.cache.has(cache_key, real_only=False):
                        stats.reused += 1
                    else:
                        # Test mode without full encoder model
                        dummy = np.zeros(8, dtype=np.float32)
                        self.cache.put(cache_key, dummy, is_real=False)
                        stats.re_embedded += 1

        stats.chunks_total = self.index.count_chunks()
        stats.embeddings_total = self.cache.count()
        stats.elapsed_seconds = time.perf_counter() - start_time
        self.last_stats = stats
        return stats
