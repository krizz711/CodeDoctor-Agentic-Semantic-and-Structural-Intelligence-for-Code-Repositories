"""The document-vector cache on the scored path: exact, incremental, documents only."""

from __future__ import annotations

import numpy as np

from app.indexing.cache import EmbeddingCache
from app.pipeline.encoder import PrePostPipelineEncoder
from app.pipeline.reranker import RerankerConfig


class CountingEncoder:
    """Deterministic vectors derived from the text; counts what it encodes."""

    model_name = "stub/counting"
    revision = "r1"
    max_seq_length = 8192
    add_eos_token = True
    normalize_embeddings = True
    document_prefix = ""
    query_prefix = "Q: "

    def __init__(self) -> None:
        self.encoded: list[tuple[str, str]] = []

    def get_prompt_text(self, text: str, mode: str = "document") -> str:
        return (self.query_prefix if mode == "query" else self.document_prefix) + text

    def encode(self, texts, mode="document", **kwargs):
        self.encoded.extend((mode, t) for t in texts)
        rows = []
        for text in texts:
            rng = np.random.default_rng(abs(hash((mode, text))) % (2**32))
            v = rng.standard_normal(8).astype(np.float32)
            rows.append(v / np.linalg.norm(v))
        return np.vstack(rows)

    def similarity(self, a, b):
        return np.atleast_2d(a) @ np.atleast_2d(b).T


def batches(texts):
    return [{"text": texts}]


DOCS = ["print(1)", "print(2)", "x = input()\nprint(x)"]


def _pipeline(cache: EmbeddingCache | None, encoder: CountingEncoder) -> PrePostPipelineEncoder:
    return PrePostPipelineEncoder(encoder, RerankerConfig(), corpus_cache=cache)


def test_cached_vectors_equal_fresh_ones(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    fresh = _pipeline(None, CountingEncoder()).encode(batches(DOCS), prompt_type="document")
    first = _pipeline(cache, CountingEncoder()).encode(batches(DOCS), prompt_type="document")
    encoder = CountingEncoder()
    second_pipeline = _pipeline(cache, encoder)
    second = second_pipeline.encode(batches(DOCS), prompt_type="document")
    assert np.array_equal(fresh, first) and np.array_equal(first, second)
    assert encoder.encoded == [], "a warm cache must encode nothing"
    assert second_pipeline.cache_stats == {"hits": 3, "misses": 0}


def test_only_changed_documents_are_encoded(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    _pipeline(cache, CountingEncoder()).encode(batches(DOCS), prompt_type="document")
    encoder = CountingEncoder()
    version_two = [DOCS[0], "print(3)", DOCS[2], "print(4)"]
    _pipeline(cache, encoder).encode(batches(version_two), prompt_type="document")
    assert [t for _, t in encoder.encoded] == ["print(3)", "print(4)"]


def test_queries_never_touch_the_cache(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    encoder = CountingEncoder()
    pipeline = _pipeline(cache, encoder)
    pipeline.encode(batches(DOCS), prompt_type="document")
    before = cache.count()
    pipeline.encode(batches(DOCS), prompt_type="query")  # same texts, as queries
    assert cache.count() == before
    assert [m for m, _ in encoder.encoded].count("query") == 3


def test_recipe_changes_invalidate(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    _pipeline(cache, CountingEncoder()).encode(batches(DOCS), prompt_type="document")
    other = CountingEncoder()
    other.revision = "r2"  # a different model revision is a different vector
    _pipeline(cache, other).encode(batches(DOCS), prompt_type="document")
    assert len(other.encoded) == 3


def test_hubness_penalty_is_subtracted_and_cached():
    encoder = CountingEncoder()
    docs = encoder.encode(DOCS, mode="document")
    queries = encoder.encode(["q1", "q2"], mode="query")
    bank = encoder.encode(["b1", "b2", "b3"], mode="query")
    plain = PrePostPipelineEncoder(encoder, RerankerConfig())
    corrected = PrePostPipelineEncoder(encoder, RerankerConfig(), hubness_bank=bank, hubness_beta=0.5, hubness_k=2)
    base = plain.similarity(queries, docs)
    got = corrected.similarity(queries, docs)
    sims = docs @ bank.T
    expected_penalty = np.sort(sims, axis=1)[:, -2:].mean(axis=1)
    assert np.allclose(got, base - 0.5 * expected_penalty[None, :], atol=1e-6)
    first = corrected._hub_penalty
    corrected.similarity(queries, docs)
    assert corrected._hub_penalty is first, "computed once per corpus"


def test_no_bank_means_no_change():
    encoder = CountingEncoder()
    docs = encoder.encode(DOCS, mode="document")
    queries = encoder.encode(["q1"], mode="query")
    assert np.array_equal(
        PrePostPipelineEncoder(encoder, RerankerConfig()).similarity(queries, docs),
        encoder.similarity(queries, docs),
    )
