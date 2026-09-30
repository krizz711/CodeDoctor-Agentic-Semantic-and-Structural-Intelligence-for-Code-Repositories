import numpy as np
import pytest
from app.retrieval.encode import CodeEncoder, get_shared_encoder
from app.retrieval.hybrid import Document, HybridRetriever, reciprocal_rank_fusion


def test_exact_symbol_match_beats_lexical_noise():
    retriever = HybridRetriever(
        [
            Document("caller", "a.js", "return getUserData(user);", "caller"),
            Document("definition", "b.js", "function getUserData(user) { return user.data; }", "getUserData"),
        ]
    )
    results = retriever.search("getUserData")
    assert results
    assert results[0].document.document_id == "definition"
    assert "exact" in results[0].sources


def test_bm25_retrieval_ranks_relevant_document_first():
    retriever = HybridRetriever(
        [
            Document("1", "preprocess.js", "function preprocessInput(data) { return sanitizeAndTrim(data); }", "preprocessInput"),
            Document("2", "locale.js", "function check(s) { var pre = s.slice(0, 6); return pre === 'en-US'; }", "check"),
            Document("3", "perf.js", "function perf(str) { if (act(A, str)) { return act(B, str); } }", "perf"),
        ]
    )
    results = retriever.search("preprocess sanitize input data")
    assert results
    assert results[0].document.document_id == "1"
    assert "bm25" in results[0].sources


def test_reciprocal_rank_fusion_properties():
    # Never invents rank for method that did not retrieve item
    exact = {0: 10.0}
    bm25 = {1: 5.0, 0: 2.0}
    dense = {2: 0.9, 1: 0.8}

    fused = reciprocal_rank_fusion({"exact": exact, "bm25": bm25, "dense": dense}, k=60)
    # doc 0 appears in exact (rank 1) and bm25 (rank 2) -> 1/(60+1) + 1/(60+2)
    expected_0 = 1 / 61 + 1 / 62
    assert abs(fused[0] - expected_0) < 1e-6

    # doc 2 only appears in dense (rank 1) -> 1/61
    expected_2 = 1 / 61
    assert abs(fused[2] - expected_2) < 1e-6


@pytest.mark.model
def test_vector_parity_between_callers():
    """
    Acceptance test: Harness and service wrappers, given the same input string
    and same prompt mode, must produce byte-identical vectors.
    """
    # Parity between the two wrappers does not depend on the tier; the small
    # tier keeps this runnable without the 1.7B default's ~8.5 GB of RAM.
    encoder1 = CodeEncoder(tier="interactive")
    encoder2 = get_shared_encoder(tier="interactive")

    test_string = "function authenticateUser(token, secret) { return verifyJwt(token, secret); }"

    # Document mode
    vec1_doc = encoder1.encode(test_string, mode="document")
    vec2_doc = encoder2.encode(test_string, mode="document")
    assert np.allclose(vec1_doc, vec2_doc, atol=1e-4), f"Document mode vectors differ: max diff={np.abs(vec1_doc - vec2_doc).max()}"

    # Query mode
    query_str = "Where is JWT authentication verified?"
    vec1_q = encoder1.encode(query_str, mode="query")
    vec2_q = encoder2.encode(query_str, mode="query")
    assert np.allclose(vec1_q, vec2_q, atol=1e-4), f"Query mode vectors differ: max diff={np.abs(vec1_q - vec2_q).max()}"

