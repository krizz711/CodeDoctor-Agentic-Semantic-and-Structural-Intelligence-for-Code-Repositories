from __future__ import annotations

import re
import sqlite3
from typing import Any

from app.parser.chunker import chunk_source


def normalize_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", value)
    return cleaned.lower()


class SQLiteIndex:
    def __init__(self, path: str = ':memory:') -> None:
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE,
                commit_hash TEXT,
                language TEXT
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
                start_byte INTEGER,
                end_byte INTEGER,
                start_line INTEGER,
                end_line INTEGER
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
                file TEXT,
                line INTEGER
            )
            """
        )
        self.conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content, signature, documentation, content='chunks', content_rowid='id'
            )
            """
        )
        self.conn.commit()

    def index_file(self, file_path: str, source: str, commit_hash: str = 'head', language: str = 'javascript') -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO files (path, commit_hash, language) VALUES (?, ?, ?)",
            (file_path, commit_hash, language),
        )

        file_id = self.conn.execute(
            "SELECT id FROM files WHERE path = ?",
            (file_path,),
        ).fetchone()[0]

        existing_ids = [row['id'] for row in self.conn.execute(
            "SELECT id FROM chunks WHERE file_id = ?",
            (file_id,),
        ).fetchall()]
        for chunk_id in existing_ids:
            self.conn.execute('DELETE FROM chunks WHERE id = ?', (chunk_id,))
            self.conn.execute('DELETE FROM symbols WHERE chunk_id = ?', (chunk_id,))
            self.conn.execute('DELETE FROM chunks_fts WHERE rowid = ?', (int(chunk_id) if chunk_id.isdigit() else 0,))

        chunks = chunk_source(source, file_path)
        for chunk in chunks:
            chunk_key = chunk.chunk_id
            self.conn.execute(
                """
                INSERT INTO chunks (id, file_id, type, name, parent, content, start_byte, end_byte, start_line, end_line)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_key,
                    file_id,
                    chunk.type,
                    chunk.name,
                    chunk.parent,
                    chunk.content,
                    chunk.start_byte,
                    chunk.end_byte,
                    chunk.start_line,
                    chunk.end_line,
                ),
            )
            self.conn.execute(
                "INSERT INTO symbols (chunk_id, name, normalized_name, type) VALUES (?, ?, ?, ?)",
                (chunk_key, chunk.name, normalize_identifier(chunk.name), chunk.type),
            )
            self.conn.execute(
                "INSERT INTO chunks_fts (rowid, content, signature, documentation) VALUES (?, ?, ?, ?)",
                (
                    hash(chunk_key) & 0x7fffffff,
                    chunk.content,
                    chunk.signature,
                    chunk.name,
                ),
            )

        self.conn.commit()

    def search_symbols(self, query: str):
        normalized = normalize_identifier(query)
        rows = self.conn.execute(
            """
            SELECT DISTINCT name, type, chunk_id
            FROM symbols
            WHERE normalized_name = ?
            ORDER BY name
            """,
            (normalized,),
        ).fetchall()
        return [dict(row) for row in rows]
