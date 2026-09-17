"""Tests for PrePostPipelineEncoder.

These deliberately include guards against the failure mode found on the
manish-workspace branch: an encoder that returns random vectors produces a
structurally valid result JSON with meaningless scores.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.retrieval.mteb_encoder import (
    PrePostPipelineEncoder,
    SparseRandomProjection,
    lexical_tokens,
    preprocess_document,
    preprocess_query,
    split_identifier,
)


class FakeDenseModel:
    """Deterministic stand-in for a SentenceTransformer (no network needed).

    Maps text to a bag-of-characters vector. Crude, but genuinely content-dependent,
    which is all these tests need.
    """

    dim = 32

    def encode(self, texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True):
        rows = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for char in text.lower():
                rows[i, ord(char) % self.dim] += 1.0
        return rows


@pytest.fixture
def encoder():
    return PrePostPipelineEncoder(dense_model=FakeDenseModel())


def test_split_identifier_handles_snake_and_camel():
    assert split_identifier("parse_input_lines") == ["parse", "input", "lines"]
    assert split_identifier("parseInputLines") == ["parse", "input", "lines"]
    assert split_identifier("HTTPServer") == ["httpserver"]


def test_preprocess_document_surfaces_identifier_words():
    processed = preprocess_document("def count_pairs(nums):\n    return 0\n")
    assert "count" in processed
    assert "pairs" in processed
    # original source is preserved too
    assert "def count_pairs" in processed


def test_preprocess_query_strips_boilerplate_headers():
    raw = "Count the pairs.\n-----Input-----\nn\n-----Output-----\nk\n"
    processed = preprocess_query(raw)
    assert "Count the pairs" in processed
    assert "-----Input-----" not in processed


def test_lexical_tokens_include_split_parts():
    tokens = lexical_tokens("countPairs")
    assert "countpairs" in tokens
    assert "count" in tokens
    assert "pairs" in tokens


def test_projection_is_deterministic_across_instances():
    a = SparseRandomProjection(out_dim=64)
    b = SparseRandomProjection(out_dim=64)
    va = a.transform({"count": 1.0, "pairs": 2.0})
    vb = b.transform({"count": 1.0, "pairs": 2.0})
    np.testing.assert_allclose(va, vb)


def test_encode_is_deterministic(encoder):
    """A random-vector encoder would fail this immediately."""
    texts = ["def solve(n): return n * 2", "print(sum(map(int, input().split())))"]
    first = encoder.encode_texts(texts)
    second = encoder.encode_texts(texts)
    np.testing.assert_allclose(first, second)


def test_encode_is_content_dependent(encoder):
    """Different inputs must produce different vectors (random vectors pass this by
    luck, but combined with the determinism test above it pins the behaviour down)."""
    vectors = encoder.encode_texts(["def add(a, b): return a + b", "import sys"])
    assert not np.allclose(vectors[0], vectors[1])


def test_identical_inputs_give_identical_vectors(encoder):
    vectors = encoder.encode_texts(["same text", "same text"])
    np.testing.assert_allclose(vectors[0], vectors[1])


def test_relevant_document_outranks_irrelevant_one(encoder):
    """End-to-end smoke test of the ranking behaviour.

    This is the test a random-vector encoder cannot pass except by chance.
    """
    query = "How is the input preprocessed before going to the main function?"
    documents = [
        "function normalize(str) { const str2 = str.trim(); return forward(str2); }",
        "function check(s) { var pre = s.slice(0, 6); return pre === 'en-US'; }",
    ]

    class QueryPrompt:
        value = "query"

    class DocPrompt:
        value = "document"

    query_vector = encoder.encode_texts([query], prompt_type=QueryPrompt())[0]
    doc_vectors = encoder.encode_texts(documents, prompt_type=DocPrompt())

    scores = doc_vectors @ query_vector
    assert scores[0] > scores[1]


def test_query_and_document_paths_differ(encoder):
    class QueryPrompt:
        value = "query"

    class DocPrompt:
        value = "document"

    text = ["count_pairs total"]
    as_query = encoder.encode_texts(text, prompt_type=QueryPrompt())
    as_document = encoder.encode_texts(text, prompt_type=DocPrompt())
    assert not np.allclose(as_query, as_document)


def test_lexical_weight_zero_gives_dense_only(encoder):
    dense_only = PrePostPipelineEncoder(dense_model=FakeDenseModel(), lexical_weight=0.0)
    vectors = dense_only.encode_texts(["def f(): pass"])
    assert vectors.shape[1] == FakeDenseModel.dim


def test_hybrid_vector_is_concatenation_width(encoder):
    vectors = encoder.encode_texts(["def f(): pass"])
    assert vectors.shape[1] == FakeDenseModel.dim + encoder.projection.out_dim


def test_encode_accepts_dataloader_style_batches(encoder):
    batches = [{"text": ["alpha", "beta"]}, {"text": ["gamma"]}]
    vectors = encoder.encode(batches)
    assert vectors.shape[0] == 3


def test_encode_empty_input(encoder):
    assert encoder.encode_texts([]).shape[0] == 0
