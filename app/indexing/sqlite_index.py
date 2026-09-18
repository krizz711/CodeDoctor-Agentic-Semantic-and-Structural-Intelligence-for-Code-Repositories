from __future__ import annotations

import hashlib
from pathlib import Path
import re
import sqlite3
from typing import Any, List, Optional

from app.parser.chunker import Chunk, chunk_source


def normalize_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", value)
    return cleaned.lower()


class SQLiteIndex:
    def __init__(self, path: str = ":memory:") -> None:
        self.db_path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE,
                commit_hash TEXT,
                language TEXT,
                content_hash TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                file_id INTEGER,
                type TEXT,
                name TEXT,
                parent TEXT,
                content TEXT,
                signature TEXT,
                start_byte INTEGER,
                end_byte INTEGER,
                start_line INTEGER,
                end_line INTEGER,
                content_hash TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT,
                name TEXT,
                normalized_name TEXT,
                type TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                target TEXT,
                relation TEXT,
                confidence REAL,
                status TEXT DEFAULT 'VERIFIED',
                file TEXT,
                line INTEGER,
                start_byte INTEGER,
                end_byte INTEGER,
                extractor TEXT DEFAULT 'ast'
            )
            """
        )
        self.conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                content,
                signature,
                name,
                tokenize = 'porter unicode61'
            )
            """
        )
        self.conn.commit()

    def index_file(
        self,
        file_path: str,
        source: str,
        commit_hash: str = "head",
        language: str = "javascript",
    ) -> list[Chunk]:
        file_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()

        self.conn.execute(
            """
            INSERT INTO files (path, commit_hash, language, content_hash)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                commit_hash=excluded.commit_hash,
                language=excluded.language,
                content_hash=excluded.content_hash
            """,
            (file_path, commit_hash, language, file_hash),
        )

        file_id = self.conn.execute(
            "SELECT id FROM files WHERE path = ?",
            (file_path,),
        ).fetchone()[0]

        # Get existing chunk ids to cleanly clean up
        existing_rows = self.conn.execute(
            "SELECT id FROM chunks WHERE file_id = ?",
            (file_id,),
        ).fetchall()
        existing_ids = [row["id"] for row in existing_rows]

        for chunk_id in existing_ids:
            self.conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
            self.conn.execute("DELETE FROM symbols WHERE chunk_id = ?", (chunk_id,))
            self.conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,))

        chunks = chunk_source(source, file_path)
        for chunk in chunks:
            chunk_key = chunk.chunk_id
            chunk_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
            self.conn.execute(
                """
                INSERT INTO chunks (
                    id, file_id, type, name, parent, content, signature,
                    start_byte, end_byte, start_line, end_line, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_key,
                    file_id,
                    chunk.type,
                    chunk.name,
                    chunk.parent,
                    chunk.content,
                    chunk.signature,
                    chunk.start_byte,
                    chunk.end_byte,
                    chunk.start_line,
                    chunk.end_line,
                    chunk_hash,
                ),
            )
            self.conn.execute(
                "INSERT INTO symbols (chunk_id, name, normalized_name, type) VALUES (?, ?, ?, ?)",
                (chunk_key, chunk.name, normalize_identifier(chunk.name), chunk.type),
            )
            self.conn.execute(
                "INSERT INTO chunks_fts (chunk_id, content, signature, name) VALUES (?, ?, ?, ?)",
                (chunk_key, chunk.content, chunk.signature, chunk.name),
            )

        self.conn.commit()
        return chunks

    def search_symbols(self, query: str) -> list[dict[str, Any]]:
        normalized = normalize_identifier(query)
        rows = self.conn.execute(
            """
            SELECT DISTINCT s.name, s.type, s.chunk_id, c.content, c.signature, c.start_line, c.end_line, f.path as file_path
            FROM symbols s
            JOIN chunks c ON s.chunk_id = c.id
            JOIN files f ON c.file_id = f.id
            WHERE s.normalized_name = ?
            ORDER BY s.name
            """,
            (normalized,),
        ).fetchall()
        return [dict(row) for row in rows]

    def search_fts(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Query SQLite FTS5 index using BM25 scoring."""
        # Sanitize query to terms
        terms = re.findall(r"[A-Za-z0-9_]+", query)
        if not terms:
            return []

        fts_query = " OR ".join(f'"{t}"' for t in terms)
        try:
            rows = self.conn.execute(
                """
                SELECT fts.chunk_id, bm25(chunks_fts) AS bm25_rank,
                       c.name, c.type, c.content, c.signature, c.start_line, c.end_line,
                       f.path as file_path
                FROM chunks_fts fts
                JOIN chunks c ON fts.chunk_id = c.id
                JOIN files f ON c.file_id = f.id
                WHERE chunks_fts MATCH ?
                ORDER BY bm25_rank ASC
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.OperationalError:
            return []

    def get_chunk(self, chunk_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            """
            SELECT c.*, f.path as file_path
            FROM chunks c
            JOIN files f ON c.file_id = f.id
            WHERE c.id = ?
            """,
            (chunk_id,),
        ).fetchone()
        return dict(row) if row else None

    def all_chunks(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT c.*, f.path as file_path
            FROM chunks c
            JOIN files f ON c.file_id = f.id
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def count_chunks(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    def count_files(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM files").fetchone()
        return row[0] if row else 0
