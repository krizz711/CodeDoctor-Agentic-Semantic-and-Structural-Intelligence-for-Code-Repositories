"""The live service must score densely from cached vectors, never re-encode."""

from __future__ import annotations

import numpy as np
import pytest

from app.retrieval.hybrid import Document, HybridRetriever


def make_documents(n: int = 6) -> list[Document]:
    return [
        Document(
            document_id=f"c{i}",
            file_path=f"src/mod{i}.js",
            content=f"export function handler{i}(input) {{ return transform{i}(input); }}",
            name=f"handler{i}",
            type="FUNCTION",
            signature=f"handler{i}(input)",
        )
        for i in range(n)
    ]


def unit_rows(n: int, dimension: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, dimension)).astype(np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


class TestPrecomputedEmbeddings:
    def test_supplying_vectors_enables_the_dense_pass(self):
        documents = make_documents()
        retriever = HybridRetriever(documents, document_embeddings=unit_rows(len(documents)))
        assert retriever.use_dense is True

    def test_dense_pass_never_calls_the_encoder(self):
        """Regression guard: re-encoding per query cost minutes, not milliseconds."""

        class ExplodingEncoder:
            def encode(self, *a, **k):  # pragma: no cover - must never run
                raise AssertionError("the dense pass re-encoded the corpus")

            def similarity(self, a, b):
                return np.asarray(a) @ np.asarray(b).T

        documents = make_documents()
        embeddings = unit_rows(len(documents))
        retriever = HybridRetriever(
            documents, encoder=ExplodingEncoder(), document_embeddings=embeddings
        )
        # The query still needs encoding, so stub that too and assert the
        # corpus side came from the supplied matrix.
        retriever.encoder = type(
            "QueryOnlyEncoder",
            (),
            {
                "encode": staticmethod(lambda *a, **k: embeddings[2].copy()),
                "similarity": staticmethod(lambda a, b: np.atleast_2d(a) @ np.atleast_2d(b).T),
            },
        )()
        results = retriever.search("handler2", limit=3)
        assert results
        assert results[0].document.document_id == "c2"

    def test_mismatched_row_count_is_rejected(self):
        documents = make_documents(6)
        with pytest.raises(ValueError, match="precomputed embeddings"):
            HybridRetriever(documents, document_embeddings=unit_rows(5))

    def test_lexical_only_still_works_without_vectors(self):
        retriever = HybridRetriever(make_documents())
        assert retriever.use_dense is False
        assert retriever.search("handler3", limit=3)


class TestSearchEndpoint:
    def test_response_reports_which_passes_ran(self):
        from fastapi.testclient import TestClient

        from app.server import app

        with TestClient(app) as client:
            response = client.post("/api/search", json={"query": "parse command options", "limit": 3})
        assert response.status_code == 200
        body = response.json()
        assert "retrieval" in body
        assert set(body["retrieval"]) == {"dense", "lexical", "indexed_chunks"}
        assert isinstance(body["retrieval"]["dense"], bool)

    def test_response_reports_the_query_classification(self):
        from fastapi.testclient import TestClient

        from app.server import app

        with TestClient(app) as client:
            response = client.post(
                "/api/search",
                json={"query": "write a function ```parseOptions(argv)``` that splits flags", "limit": 3},
            )
        analysis = response.json()["query_analysis"]
        assert analysis["family"] in {
            "competitive_stdin", "leetcode_method", "kata_function", "unknown"
        }
        assert 0.0 <= analysis["confidence"] <= 1.0
        assert "parseOptions" in analysis["identifiers"]

    def test_empty_query_short_circuits(self):
        from fastapi.testclient import TestClient

        from app.server import app

        with TestClient(app) as client:
            body = client.post("/api/search", json={"query": "   "}).json()
        assert body["total"] == 0
        assert body["results"] == []
