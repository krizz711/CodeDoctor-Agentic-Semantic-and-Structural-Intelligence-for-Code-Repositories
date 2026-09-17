-- V001__initial_schema.sql
-- Initial database schema for CodeDoctor Prototype (Theme 1)
-- Anticipates Module 4 (Evolutionary Retrieval) by including commit_hash and semantic_hash.

CREATE TABLE files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    commit_hash TEXT NOT NULL,
    content TEXT NOT NULL,
    UNIQUE(repo_name, file_path, commit_hash)
);

CREATE TABLE chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    start_byte INTEGER NOT NULL,
    end_byte INTEGER NOT NULL,
    semantic_hash TEXT NOT NULL, -- Semantic boundary hash for incremental re-indexing (Module 4)
    ast_type TEXT NOT NULL,      -- e.g., 'function_declaration', 'class_declaration'
    content TEXT NOT NULL,       -- The exact byte-extracted string
    FOREIGN KEY(file_id) REFERENCES files(id)
);

-- FTS5 index for BM25 (Module 3)
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content,
    content='chunks',
    content_rowid='id'
);

-- Triggers to keep FTS5 synchronized with chunks table
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
END;

CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
  INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;
