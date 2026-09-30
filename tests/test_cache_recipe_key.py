"""E3 Part 1 (experiments/C3_cycle): every document cache is keyed on the full recipe.

The demo cache used to be keyed on the text alone, so a same-dimension model or
a prefix change served the previous recipe's vectors without a forward pass.
The first three tests reproduce that defect (found by the cycle-3 skeptic);
the fourth pins the scored path's keys byte for byte, so the refactor cannot
invalidate an existing corpus cache.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from app.indexing.cache import EmbeddingCache, recipe_cache_key, stack_cached
from app.indexing.incremental import IncrementalIndexer
from app.indexing.sqlite_index import SQLiteIndex
from app.retrieval.hybrid import Document
from app.workbench import Workbench


class StubEncoder:
    """Deterministic encoder whose vectors depend on its recipe and the text."""

    def __init__(self, model_name: str, dim: int = 16, document_prefix: str = "", revision: str = "r1"):
        self.model_name, self.revision, self.dimension = model_name, revision, dim
        self.document_prefix, self.query_prefix = document_prefix, "Q: "
        self.max_seq_length, self.add_eos_token, self.normalize_embeddings = 8192, True, True
        self.calls = 0

    def _vec(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(f"{self.model_name}|{self.document_prefix}|{text}".encode()).digest()
        v = np.random.default_rng(int.from_bytes(digest[:8], "little")).standard_normal(self.dimension)
        return (v / np.linalg.norm(v)).astype(np.float32)

    def encode(self, texts, mode="document", **_):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        self.calls += len(items)
        out = np.vstack([self._vec(t) for t in items])
        return out[0] if single else out

    def get_prompt_text(self, text, mode="document"):
        return (self.query_prefix if mode == "query" else self.document_prefix) + text


DOCS = [
    Document(document_id=f"c{i}", file_path="m.py", content=f"def f{i}(x):\n    return x + {i}\n",
             name=f"f{i}", type="FUNCTION", signature=f"def f{i}(x)")
    for i in range(5)
]


def bench(cache: EmbeddingCache, encoder: StubEncoder) -> Workbench:
    wb = Workbench(cache=cache)
    wb._encoder = encoder
    return wb


def test_workbench_never_serves_another_models_vectors(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.db")
    bench(cache, StubEncoder("model-a"))._encode_documents(DOCS)
    other = StubEncoder("model-b")
    vectors = bench(cache, other)._encode_documents(DOCS)
    assert other.calls == len(DOCS)
    assert np.allclose(vectors, other.encode([f"{d.signature}\n{d.content}" for d in DOCS]))


def test_workbench_prefix_change_invalidates_the_cache(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.db")
    bench(cache, StubEncoder("m", document_prefix=""))._encode_documents(DOCS)
    changed = StubEncoder("m", document_prefix="Represent this code: ")
    bench(cache, changed)._encode_documents(DOCS)
    assert changed.calls == len(DOCS)


def test_incremental_indexer_reencodes_after_a_model_swap(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "m.py").write_text("def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n",
                               encoding="utf-8")
    cache = EmbeddingCache(tmp_path / "cache.db")
    first = IncrementalIndexer(SQLiteIndex(str(tmp_path / "i1.db")), cache=cache,
                               encoder=StubEncoder("A")).index_repository(repo)
    second = IncrementalIndexer(SQLiteIndex(str(tmp_path / "i2.db")), cache=cache,
                                encoder=StubEncoder("B")).index_repository(repo)
    assert first.re_embedded > 0
    assert second.re_embedded == first.re_embedded and second.reused == 0


def test_same_recipe_still_reuses_and_a_move_costs_nothing(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.db")
    bench(cache, StubEncoder("m"))._encode_documents(DOCS)
    again = StubEncoder("m")
    moved = [Document(document_id=d.document_id, file_path="elsewhere.py", content=d.content, name=d.name,
                      type=d.type, signature=d.signature) for d in DOCS]
    bench(cache, again)._encode_documents(moved)
    assert again.calls == 0


def test_scored_path_keys_are_byte_identical_to_before():
    """The pre-refactor formula of PrePostPipelineEncoder.document_cache_key, inlined."""
    from app.indexing.cache import compute_embedding_key
    from app.pipeline.encoder import PrePostPipelineEncoder

    enc = StubEncoder("codefuse-ai/F2LLM-v2-1.7B", document_prefix="")
    text = "def f(): pass"
    legacy = compute_embedding_key(
        encoder_input="document\x00" + enc.get_prompt_text(text, "document"),
        encoder_revision=f"{enc.model_name}@{enc.revision}",
        tokenizer_revision="max8192|eos1|norm1",
        template_revision=repr(""),
    )
    assert PrePostPipelineEncoder(enc).document_cache_key(text) == legacy == recipe_cache_key(enc, text)


def test_mixed_dimensions_fail_with_a_clear_message():
    with pytest.raises(ValueError, match="mixes recipes"):
        stack_cached([np.zeros(1024), np.zeros(2048)])
    with pytest.raises(ValueError, match="encoder produces 2048"):
        stack_cached([np.zeros(1024)], expected_dim=2048)
    assert stack_cached([np.zeros(4), np.ones(4)]).shape == (2, 4)
