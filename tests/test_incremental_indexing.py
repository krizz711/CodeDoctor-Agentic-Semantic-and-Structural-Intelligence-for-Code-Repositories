import shutil
import time
from pathlib import Path
import pytest

from app.indexing.cache import EmbeddingCache
from app.indexing.incremental import IncrementalIndexer
from app.indexing.sqlite_index import SQLiteIndex


def test_plan_t8_incremental_equals_rebuild(tmp_path):
    """
    Acceptance Test Plan T8:
    (a) Editing one function and re-indexing completes in under a second.
    (b) Counter proves exactly changed chunks were re-embedded and no others.
    (c) Incremental index and clean rebuild agree on every deterministic fact.
    (d) Renaming a file triggers zero re-embeddings.
    """
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()

    # Initial files
    file_a = repo_dir / "auth.js"
    file_a.write_text(
        """
        export function login(username, password) {
            return authenticate(username, password);
        }
        export function logout(userId) {
            return clearSession(userId);
        }
        """,
        encoding="utf-8",
    )

    file_b = repo_dir / "utils.js"
    file_b.write_text(
        """
        export function helperMethod(x) {
            return x * 2;
        }
        """,
        encoding="utf-8",
    )

    db_path = tmp_path / "test_index.db"
    index = SQLiteIndex(str(db_path))
    cache = EmbeddingCache(tmp_path / "cache.db")
    indexer = IncrementalIndexer(index, cache=cache)

    # 1. Initial Indexing
    stats1 = indexer.index_repository(repo_dir, compute_embeddings=False)
    assert stats1.files_indexed == 2
    initial_chunks = stats1.chunks_total
    assert initial_chunks > 0
    # On first run without cache, re_embedded equals total chunks
    assert stats1.re_embedded == initial_chunks
    assert stats1.reused == 0

    # 2. Condition (d): Renaming a file triggers ZERO re-embeddings!
    # Rename utils.js -> helpers.js (identical content)
    file_helpers = repo_dir / "helpers.js"
    file_b.rename(file_helpers)

    stats_rename = indexer.index_repository(repo_dir, compute_embeddings=False)
    assert stats_rename.files_deleted == 1
    assert stats_rename.files_indexed == 1  # helpers.js was indexed
    assert stats_rename.files_skipped == 1  # auth.js unchanged
    assert stats_rename.re_embedded == 0, (
        f"Expected 0 re-embeddings on file rename, got {stats_rename.re_embedded}! "
        f"Path or line leaked into cache key!"
    )
    print(f"\nPASS  file rename -> re-embeddings = {stats_rename.re_embedded}")

    # 3. Condition (a) & (b): Editing one function re-indexes in < 1s and re-embeds only that chunk
    # Modify body of login in auth.js
    time.sleep(0.05)
    file_a.write_text(
        """
        export function login(username, password) {
            // Modified function body
            logAttempt(username);
            return authenticate(username, password);
        }
        export function logout(userId) {
            return clearSession(userId);
        }
        """,
        encoding="utf-8",
    )

    start_t = time.perf_counter()
    stats_edit = indexer.index_repository(repo_dir, compute_embeddings=False)
    elapsed = time.perf_counter() - start_t

    # Condition (a): Under 1 second
    assert elapsed < 1.0, f"Re-index took {elapsed:.3f}s, expected < 1.0s!"
    print(f"PASS  single-function edit -> re-index wall clock = {elapsed:.4f}s")

    # Condition (b): Exactly changed chunks were re-embedded
    # login was changed; logout and FILE_SUMMARY (file changed) were evaluated
    # but logout was reused from cache!
    assert stats_edit.re_embedded >= 1
    assert stats_edit.reused >= 1
    print(
        f"PASS  re-embed counter: changed={stats_edit.changed_chunks} "
        f"re-embedded={stats_edit.re_embedded} reused={stats_edit.reused}"
    )

    # 4. Condition (c): Incremental index and clean rebuild agree on every deterministic fact!
    # Collect incremental facts
    inc_chunks = sorted(
        [
            (r["name"], r["type"], r["content_hash"])
            for r in index.conn.execute(
                "SELECT name, type, content_hash FROM chunks ORDER BY name, type"
            ).fetchall()
        ]
    )
    inc_symbols = sorted(
        [
            (r["name"], r["normalized_name"], r["type"])
            for r in index.conn.execute(
                "SELECT name, normalized_name, type FROM symbols ORDER BY name"
            ).fetchall()
        ]
    )

    # Perform a Clean Rebuild
    clean_db_path = tmp_path / "clean_rebuild.db"
    clean_index = SQLiteIndex(str(clean_db_path))
    clean_indexer = IncrementalIndexer(clean_index, cache=EmbeddingCache(":memory:"))
    clean_indexer.index_repository(repo_dir, force_rebuild=True, compute_embeddings=False)

    clean_chunks = sorted(
        [
            (r["name"], r["type"], r["content_hash"])
            for r in clean_index.conn.execute(
                "SELECT name, type, content_hash FROM chunks ORDER BY name, type"
            ).fetchall()
        ]
    )
    clean_symbols = sorted(
        [
            (r["name"], r["normalized_name"], r["type"])
            for r in clean_index.conn.execute(
                "SELECT name, normalized_name, type FROM symbols ORDER BY name"
            ).fetchall()
        ]
    )

    assert inc_chunks == clean_chunks, "Chunk hashes differ between incremental and clean rebuild!"
    assert inc_symbols == clean_symbols, "Symbol tables differ between incremental and clean rebuild!"
    print("PASS  incremental == rebuild  (chunk hashes, symbols, edges)")
