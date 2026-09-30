"""Skeptic probe for Experiment 3 (a): is the demo embedding cache keyed without the model/recipe?

Lives outside the repo on purpose. Run with the repo on sys.path:
    PYTHONPATH=C:/Users/Asus/manish/CodeDoctor python -m pytest -q test_cache_key_defect.py -p no:cacheprovider

Tests named test_DEFECT_* are expected to FAIL on the current code if the defect is real.
Tests named test_OBS_* record observed behaviour (they pass when the observation holds).
"""
from __future__ import annotations

import numpy as np
import pytest

from app.indexing.cache import EmbeddingCache
from app.indexing.incremental import IncrementalIndexer
from app.indexing.sqlite_index import SQLiteIndex
from app.retrieval.hybrid import Document
from app.workbench import Workbench


class StubEncoder:
    """Deterministic encoder whose vector depends only on its identity (model/prefix)."""

    def __init__(self, model_name: str, dim: int, document_prefix: str = "", revision: str = "r1",
                 torch_dtype: str = "float32", device: str = "cpu"):
        self.model_name = model_name
        self.revision = revision
        self.dim = dim
        self.dimension = dim
        self.document_prefix = document_prefix
        self.query_prefix = "Q: "
        self.max_seq_length = 8192
        self.add_eos_token = True
        self.normalize_embeddings = True
        self.torch_dtype = torch_dtype
        self.device = device
        self.calls = 0

    def _vec(self, text: str) -> np.ndarray:
        seed = abs(hash((self.model_name, self.document_prefix, text))) % (2**32)
        v = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def encode(self, texts, mode="document", **_):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        self.calls += len(items)
        out = np.vstack([self._vec(t) for t in items])
        return out[0] if single else out

    def get_prompt_text(self, text, mode="document"):
        return (self.query_prefix if mode == "query" else self.document_prefix) + text

    def similarity(self, a, b):
        a = a.reshape(1, -1) if a.ndim == 1 else a
        b = b.reshape(1, -1) if b.ndim == 1 else b
        return a @ b.T


DOCS = [
    Document(document_id=f"c{i}", file_path="m.py", content=f"def f{i}(x):\n    return x + {i}\n",
             name=f"f{i}", type="FUNCTION", signature=f"def f{i}(x)")
    for i in range(5)
]


def _bench(cache, encoder):
    bench = Workbench(cache=cache)
    bench._encoder = encoder  # what _get_encoder() would return
    return bench


def test_DEFECT_workbench_serves_other_models_vectors_same_dim(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.db")
    enc_a = StubEncoder("codefuse-ai/F2LLM-v2-0.6B", 16)
    vec_a = _bench(cache, enc_a)._encode_documents(DOCS)
    enc_b = StubEncoder("some-other/model-same-dim", 16)
    vec_b = _bench(cache, enc_b)._encode_documents(DOCS)
    print(f"\n[same-dim swap] encoder B forward passes: {enc_b.calls}; "
          f"max|vec_b - vec_a| = {np.abs(vec_b - vec_a).max():.3g}")
    expected_b = enc_b.encode([d.signature + "\n" + d.content for d in DOCS])
    assert np.allclose(vec_b, expected_b), "workbench returned encoder A's vectors for encoder B"


def test_DEFECT_workbench_serves_stale_vectors_after_prefix_change(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.db")
    enc_a = StubEncoder("codefuse-ai/F2LLM-v2-0.6B", 16, document_prefix="")
    vec_a = _bench(cache, enc_a)._encode_documents(DOCS)
    enc_b = StubEncoder("codefuse-ai/F2LLM-v2-0.6B", 16, document_prefix="Represent this code: ")
    vec_b = _bench(cache, enc_b)._encode_documents(DOCS)
    print(f"\n[prefix change] encoder B forward passes: {enc_b.calls}; identical={np.array_equal(vec_a, vec_b)}")
    assert enc_b.calls == len(DOCS), "prefix change did not invalidate the cache"


def test_DEFECT_incremental_indexer_skips_reencode_after_model_swap(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "m.py").write_text("def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n",
                               encoding="utf-8")
    cache = EmbeddingCache(tmp_path / "cache.db")
    s1 = IncrementalIndexer(SQLiteIndex(str(tmp_path / "i1.db")), cache=cache,
                            encoder=StubEncoder("A", 16)).index_repository(repo)
    enc_b = StubEncoder("B", 16)
    s2 = IncrementalIndexer(SQLiteIndex(str(tmp_path / "i2.db")), cache=cache,
                            encoder=enc_b).index_repository(repo)
    print(f"\n[incremental] pass1 re_embedded={s1.re_embedded}; pass2 (model B) "
          f"re_embedded={s2.re_embedded} reused={s2.reused} B calls={enc_b.calls}")
    assert s2.re_embedded == s1.re_embedded, "model B reused model A's vectors"


def test_OBS_dim_mismatch_all_hits_crashes_at_query_time(tmp_path):
    """1.7B (2048-d) query vs cached 0.6B (1024-d) documents: loud, not silent."""
    cache = EmbeddingCache(tmp_path / "cache.db")
    _bench(cache, StubEncoder("0.6B", 1024))._encode_documents(DOCS)
    enc_big = StubEncoder("1.7B", 2048)
    docs = _bench(cache, enc_big)._encode_documents(DOCS)
    assert docs.shape == (5, 1024) and enc_big.calls == 0  # served 0.6B vectors to the 1.7B bench
    q = enc_big.encode("query", mode="query")
    with pytest.raises(ValueError):
        enc_big.similarity(q, docs)


def test_OBS_dim_mismatch_partial_hits_crashes_at_build_time(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.db")
    _bench(cache, StubEncoder("0.6B", 1024))._encode_documents(DOCS[:3])
    with pytest.raises(ValueError):
        _bench(cache, StubEncoder("1.7B", 2048))._encode_documents(DOCS)


def test_OBS_scored_path_key_includes_model_but_not_device_or_dtype():
    from app.pipeline.encoder import PrePostPipelineEncoder

    def key(enc):
        return PrePostPipelineEncoder(enc).document_cache_key("def f(): pass")

    base = StubEncoder("codefuse-ai/F2LLM-v2-0.6B", 16)
    assert key(base) != key(StubEncoder("codefuse-ai/F2LLM-v2-1.7B", 16))
    assert key(base) != key(StubEncoder("codefuse-ai/F2LLM-v2-0.6B", 16, revision="r2"))
    assert key(base) != key(StubEncoder("codefuse-ai/F2LLM-v2-0.6B", 16, document_prefix="X"))
    # Operating point is NOT in the key: GPU fp16/bf16 and CPU fp32 vectors share one address.
    assert key(base) == key(StubEncoder("codefuse-ai/F2LLM-v2-0.6B", 16, torch_dtype="float16", device="cuda"))


def test_OBS_demo_key_ignores_model_entirely():
    from app.indexing.cache import compute_embedding_key
    # The demo path's key is a function of the text only.
    assert compute_embedding_key("x") == compute_embedding_key("x", "v1", "v1", "v1")
