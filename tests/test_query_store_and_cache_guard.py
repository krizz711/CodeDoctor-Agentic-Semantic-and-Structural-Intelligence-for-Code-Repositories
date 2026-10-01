"""The final CPU evaluation's safety nets: MTEB-exact document keys, a complete-cache
guard, and a crash-safe, shardable query store.

A 3,765-query CPU run at 1.7B takes many hours. These tests pin that (a) the
document cache is keyed on the strings MTEB really passes, (b) a run refuses to
silently encode documents it should have cached, and (c) partial or sharded query
encoding reproduces exactly what a direct encode returns.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from app.indexing.cache import EmbeddingCache, recipe_cache_key
from app.pipeline.encoder import (
    CorpusCacheIncomplete,
    PrePostPipelineEncoder,
    QueryShardEncoded,
    QueryStoreIncomplete,
    mteb_document_text,
)

ROOT = Path(__file__).resolve().parent.parent


class StubEncoder:
    """Deterministic vectors per (mode, prompt); counts what it was asked to encode."""

    model_name, revision, dimension = "stub", "r1", 16
    document_prefix, query_prefix = "", "Q: "
    max_seq_length, add_eos_token, normalize_embeddings = 8192, True, True

    def __init__(self, fail_after: int | None = None):
        self.encoded: list[tuple[str, str]] = []
        self.fail_after = fail_after

    def get_prompt_text(self, text, mode="document"):
        return (self.query_prefix if mode == "query" else self.document_prefix) + text

    def encode(self, texts, mode="document", **_):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if self.fail_after is not None and len(self.encoded) + len(items) > self.fail_after:
            raise RuntimeError("simulated crash")
        rows = []
        for text in items:
            self.encoded.append((mode, text))
            seed = int.from_bytes(hashlib.sha256(f"{mode}|{text}".encode()).digest()[:8], "little")
            v = np.random.default_rng(seed).standard_normal(self.dimension)
            rows.append((v / np.linalg.norm(v)).astype(np.float32))
        out = np.vstack(rows)
        return out[0] if single else out


def batches(texts, size=3):
    return [{"text": texts[i:i + size]} for i in range(0, len(texts), size)]


QUERIES = [f"Compute the answer number {i}.\n\nInput\n{i}\n\nOutput\n{i * 2}\n" for i in range(23)]


# --- (a) document keys are MTEB's strings ------------------------------------

def test_document_text_is_what_mteb_passes_including_whitespace_and_title():
    from mteb._create_dataloaders import _corpus_to_dict

    rows = [
        {"id": "d1", "text": "  def f():\n    return 1\n\n"},
        {"id": "d2", "text": "x = 1", "title": "Title"},
        {"id": "d3", "text": "\n\tprint(2)  "},
        {"id": "d4", "text": "plain"},
    ]
    for row in rows:
        expected = _corpus_to_dict({"title": "", **row})["text"]
        assert mteb_document_text(row) == expected
    assert mteb_document_text(rows[0]) == "def f():\n    return 1"      # stripped, unlike the raw text
    assert mteb_document_text(rows[1]) == "Title x = 1"


@pytest.mark.skipif(not (ROOT / "artifacts" / "apps" / "texts_corpus.jsonl").exists(), reason="corpus texts not built")
def test_a_large_share_of_the_real_corpus_changes_under_mteb_stripping():
    """Why this matters: keying on raw text would miss all of these documents."""
    with open(ROOT / "artifacts" / "apps" / "texts_corpus.jsonl", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    changed = sum(mteb_document_text(r) != r["text"] for r in rows)
    assert len(rows) == 8765 and changed == 3687


def test_a_cache_built_from_mteb_texts_is_fully_hit_and_a_raw_one_is_not(tmp_path):
    raw = ["  a = 1\n", "b = 2  ", "c = 3"]
    rows = [{"id": f"d{i}", "text": t} for i, t in enumerate(raw)]
    seen_by_mteb = [mteb_document_text(r) for r in rows]

    encoder = StubEncoder()
    good = EmbeddingCache(tmp_path / "good.db")
    good.put_many([(recipe_cache_key(encoder, t), encoder.encode(t, mode="document")) for t in seen_by_mteb])
    pipeline = PrePostPipelineEncoder(encoder, corpus_cache=good, require_complete_corpus_cache=True)
    encoder.encoded.clear()
    pipeline.encode(batches(seen_by_mteb), prompt_type="document")
    assert pipeline.cache_stats == {"hits": 3, "misses": 0} and encoder.encoded == []

    naive = EmbeddingCache(tmp_path / "naive.db")
    naive.put_many([(recipe_cache_key(encoder, t), encoder.encode(t, mode="document")) for t in raw])
    strict = PrePostPipelineEncoder(encoder, corpus_cache=naive, require_complete_corpus_cache=True)
    encoder.encoded.clear()
    with pytest.raises(CorpusCacheIncomplete, match="2 of 3"):
        strict.encode(batches(seen_by_mteb), prompt_type="document")
    assert encoder.encoded == []                       # refused before encoding anything


def test_an_incomplete_cache_still_fills_itself_unless_completeness_is_required(tmp_path):
    encoder = StubEncoder()
    cache = EmbeddingCache(tmp_path / "c.db")
    pipeline = PrePostPipelineEncoder(encoder, corpus_cache=cache)
    pipeline.encode(batches(["x = 1", "y = 2"]), prompt_type="document")
    assert pipeline.cache_stats == {"hits": 0, "misses": 2} and cache.count() == 2


# --- (b) query store: reuse, crash safety, shards ----------------------------

def test_query_keys_are_namespaced_apart_from_document_keys():
    e = StubEncoder()
    assert recipe_cache_key(e, "same text", "query") != recipe_cache_key(e, "same text", "document")
    with pytest.raises(ValueError):
        recipe_cache_key(e, "t", "other")


def test_stored_queries_equal_a_direct_encode_and_are_reused(tmp_path):
    direct = PrePostPipelineEncoder(StubEncoder()).encode(batches(QUERIES), prompt_type="query")
    encoder = StubEncoder()
    store = EmbeddingCache(tmp_path / "q.db")
    first = PrePostPipelineEncoder(encoder, query_store=store)
    assert np.array_equal(first.encode(batches(QUERIES), prompt_type="query"), direct)
    assert first.query_stats == {"hits": 0, "encoded": 23}
    encoder.encoded.clear()
    again = PrePostPipelineEncoder(encoder, query_store=store)
    assert np.array_equal(again.encode(batches(QUERIES), prompt_type="query"), direct)
    assert again.query_stats == {"hits": 23, "encoded": 0} and encoder.encoded == []


def test_a_crash_keeps_finished_queries_and_the_rerun_encodes_only_the_rest(tmp_path):
    direct = PrePostPipelineEncoder(StubEncoder()).encode(batches(QUERIES), prompt_type="query")
    store = EmbeddingCache(tmp_path / "q.db")
    crashing = PrePostPipelineEncoder(StubEncoder(fail_after=10), query_store=store)
    with pytest.raises(RuntimeError, match="simulated crash"):
        crashing.encode(batches(QUERIES), prompt_type="query")
    saved = store.count()
    assert 0 < saved <= 10 and saved % PrePostPipelineEncoder.QUERY_CHUNK == 0

    encoder = StubEncoder()
    resumed = PrePostPipelineEncoder(encoder, query_store=store)
    assert np.array_equal(resumed.encode(batches(QUERIES), prompt_type="query"), direct)
    assert resumed.query_stats == {"hits": saved, "encoded": 23 - saved} and len(encoder.encoded) == 23 - saved


def test_shards_cover_every_query_once_and_merge_into_the_direct_result(tmp_path):
    direct = PrePostPipelineEncoder(StubEncoder()).encode(batches(QUERIES), prompt_type="query")
    merged = EmbeddingCache(tmp_path / "merged.db")
    encoded_per_shard = []
    for index in range(3):
        shard_store = EmbeddingCache(tmp_path / f"s{index}.db")
        shard = PrePostPipelineEncoder(StubEncoder(), query_store=shard_store, query_shard=(index, 3))
        shard.encode(batches(QUERIES), prompt_type="query")
        with pytest.raises(QueryShardEncoded):
            shard.similarity(np.zeros((1, 16), dtype=np.float32), np.zeros((1, 16), dtype=np.float32))
        encoded_per_shard.append(shard.query_stats["encoded"])
        keys = [k for (k,) in shard_store.conn.execute("SELECT cache_key FROM embedding_cache")]
        merged.put_many([(k, shard_store.get(k)) for k in keys])
    assert sum(encoded_per_shard) == 23 and all(n > 0 for n in encoded_per_shard)
    assert merged.count() == 23

    encoder = StubEncoder()
    final = PrePostPipelineEncoder(encoder, query_store=merged)
    assert np.array_equal(final.encode(batches(QUERIES), prompt_type="query"), direct)
    assert final.query_stats == {"hits": 23, "encoded": 0} and encoder.encoded == []


def test_verification_passes_on_honest_vectors_and_fails_on_tampered_ones(tmp_path):
    store = EmbeddingCache(tmp_path / "q.db")
    PrePostPipelineEncoder(StubEncoder(), query_store=store).encode(batches(QUERIES), prompt_type="query")

    honest = PrePostPipelineEncoder(StubEncoder(), query_store=store, query_verify=5)
    honest.encode(batches(QUERIES), prompt_type="query")
    assert honest.query_verification["sampled"] == 5 and honest.query_verification["min_cosine"] > 0.9999

    key = recipe_cache_key(StubEncoder(), QUERIES[0], "query")
    store.put(key, -store.get(key))                          # a wrong vector for one query
    victim = PrePostPipelineEncoder(StubEncoder(), query_store=store, query_verify=len(QUERIES))
    with pytest.raises(RuntimeError, match="do not reproduce"):
        victim.encode(batches(QUERIES), prompt_type="query")


def test_strict_replay_reuses_all_queries_without_encoding_missing_ones(tmp_path):
    encoder = StubEncoder()
    store = EmbeddingCache(tmp_path / "q.db")
    PrePostPipelineEncoder(encoder, query_store=store).encode(batches(QUERIES), prompt_type="query")
    encoder.encoded.clear()
    strict = PrePostPipelineEncoder(encoder, query_store=store, require_complete_query_store=True)
    strict.encode(batches(QUERIES), prompt_type="query")
    assert strict.query_stats == {"hits": 23, "encoded": 0} and encoder.encoded == []


@pytest.mark.parametrize("defect", ["missing", "extra", "nan", "dimension", "norm", "zero"])
def test_strict_replay_rejects_invalid_stores_before_any_encoding(tmp_path, defect):
    encoder = StubEncoder()
    store = EmbeddingCache(tmp_path / "q.db")
    PrePostPipelineEncoder(encoder, query_store=store).encode(batches(QUERIES), prompt_type="query")
    key = recipe_cache_key(encoder, QUERIES[0], "query")
    vector = store.get(key).copy()
    if defect == "missing":
        store.conn.execute("DELETE FROM embedding_cache WHERE cache_key = ?", (key,))
        store.conn.commit()
    elif defect == "extra":
        store.put("unexpected", vector)
    elif defect == "nan":
        vector[0] = np.nan
        store.put(key, vector)
    elif defect == "dimension":
        store.put(key, vector[:-1])
    elif defect == "norm":
        store.put(key, vector * 2)
    else:
        store.put(key, np.zeros_like(vector))
    encoder.encoded.clear()
    strict = PrePostPipelineEncoder(encoder, query_store=store, require_complete_query_store=True)
    with pytest.raises(QueryStoreIncomplete):
        strict.encode(batches(QUERIES), prompt_type="query")
    assert encoder.encoded == []


def test_live_verification_rejects_a_nan_encoder_output(tmp_path, monkeypatch):
    encoder = StubEncoder()
    store = EmbeddingCache(tmp_path / "q.db")
    PrePostPipelineEncoder(encoder, query_store=store).encode(batches(QUERIES), prompt_type="query")
    monkeypatch.setattr(encoder, "encode", lambda *a, **k: np.full((1, encoder.dimension), np.nan, dtype=np.float32))
    strict = PrePostPipelineEncoder(encoder, query_store=store, query_verify=30, require_complete_query_store=True)
    with pytest.raises(ValueError, match="non-finite"):
        strict.encode(batches(QUERIES), prompt_type="query")


def test_corpus_manifest_checks_order_and_exact_recipe_keys(tmp_path):
    from app.indexing.cache_validation import text_fingerprint

    texts = ["x=1", "y=2", "z=3"]
    encoder = StubEncoder()
    store = EmbeddingCache(tmp_path / "corpus.db")
    store.put_many([(recipe_cache_key(encoder, text), encoder.encode(text)) for text in texts])
    manifest = {"documents": 3, "texts_sha256": text_fingerprint(texts)}
    strict = PrePostPipelineEncoder(encoder, corpus_cache=store, corpus_manifest=manifest,
                                    require_complete_corpus_cache=True)
    strict.encode(batches(texts), prompt_type="document")
    with pytest.raises(CorpusCacheIncomplete, match="fingerprint"):
        strict.encode(batches(list(reversed(texts))), prompt_type="document")
    store.put("extra", encoder.encode("extra"))
    with pytest.raises(CorpusCacheIncomplete, match="unexpected"):
        strict.encode(batches(texts), prompt_type="document")


def test_strict_replay_refuses_nonfinite_scores_or_silently_skipped_verification(tmp_path, monkeypatch):
    from app.pipeline.execution_rerank import ExecutionConfig

    encoder = StubEncoder()
    store = EmbeddingCache(tmp_path / "q.db")
    strict = PrePostPipelineEncoder(encoder, query_store=store, require_complete_query_store=True,
                                    execution_config=ExecutionConfig(enabled=True))
    monkeypatch.setattr(encoder, "similarity", lambda a, b: np.full((1, 1), np.nan), raising=False)
    with pytest.raises(RuntimeError, match="non-finite retrieval"):
        strict.similarity(np.zeros((1, 16)), np.zeros((1, 16)))
    monkeypatch.setattr(encoder, "similarity", lambda a, b: np.ones((1, 1)))
    with pytest.raises(RuntimeError, match="query texts"):
        strict.similarity(np.zeros((1, 16)), np.zeros((1, 16)))
