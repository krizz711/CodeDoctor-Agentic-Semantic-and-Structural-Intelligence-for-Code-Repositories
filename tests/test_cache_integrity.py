
from app.indexing.cache import EmbeddingCache
from app.indexing.incremental import IncrementalIndexer
from app.indexing.sqlite_index import SQLiteIndex


def test_dummy_embeddings_never_counted_as_reused(tmp_path, stub_encoder):
    """
    Plan Remediation §3.4:
    Asserts that dummy embeddings populated in test mode (compute_embeddings=False)
    are never silently counted as reused by a subsequent run with compute_embeddings=True.
    """
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    sample_file = repo_dir / "sample.js"
    sample_file.write_text(
        """
        export function processTask(taskId) {
            return performWork(taskId);
        }
        export function finishTask(taskId) {
            return markDone(taskId);
        }
        """,
        encoding="utf-8",
    )

    db_path = tmp_path / "index.db"
    index = SQLiteIndex(str(db_path))
    cache = EmbeddingCache(tmp_path / "cache.db")
    indexer = IncrementalIndexer(index=index, cache=cache, encoder=stub_encoder)

    # First pass: test mode, no real encoder
    stats1 = indexer.index_repository(repo_dir, compute_embeddings=False)
    assert stats1.chunks_total > 0
    assert stats1.re_embedded == stats1.chunks_total  # everything "new" in dummy mode

    # Second pass: real encoder available
    stats2 = indexer.index_repository(repo_dir, compute_embeddings=True)
    assert stats2.reused == 0, f"Expected 0 reused, got {stats2.reused}! Dummy embeddings were leaked as real!"
    assert stats2.re_embedded == stats2.chunks_total

    # No zero-vectors should be retrievable as real embeddings after pass 2
    for key in cache.all_keys():
        vec = cache.get(key, real_only=True)
        assert vec is not None, f"Real embedding missing for {key}"
        assert vec.any(), f"zero vector still cached at {key}"
