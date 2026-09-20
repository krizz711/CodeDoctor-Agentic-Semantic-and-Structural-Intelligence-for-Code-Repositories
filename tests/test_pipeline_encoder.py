"""The MTEB contract: encode(document) -> encode(query) -> similarity().

MTEB's ``SearchEncoderWrapper`` with no index backend calls the encoder once
per prompt type and then hands the *full* query and document matrices to
``similarity()``.  That is the reranking hook, and this module pins our side of
it: the texts recorded during ``encode()`` must still line up with the score
matrix when ``similarity()`` is called, or the second pass silently no-ops.
"""

from __future__ import annotations

import numpy as np

from app.pipeline.encoder import PrePostPipelineEncoder
from app.pipeline.features import FEATURE_NAMES
from app.pipeline.reranker import RerankerConfig

DOCUMENTS = [
    "n = int(input())\nprint(n * 2)",
    "def count_squares(lines):\n    return len(lines)",
    "class Solution:\n    def twoSum(self, nums):\n        return nums[0]",
    "import sys\nfor line in sys.stdin:\n    print(line)",
]
QUERIES = [
    "Write a function ```count_squares(lines)``` that counts squares.",
    "Read n from input and print twice n.",
]


class StubEncoder:
    """Deterministic stand-in; the vectors are fixed so ranking is predictable."""

    model_name = "stub/deterministic"
    tier = "interactive"
    device = "cpu"
    similarity_metric = "cosine"
    normalize_embeddings = True
    query_prefix = "Q: "
    document_prefix = ""

    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0

    def encode(self, texts, mode="document", **kwargs):
        if mode == "query":
            self.query_calls += 1
            rows = np.eye(len(texts), 4, dtype=np.float32)
        else:
            self.document_calls += 1
            rows = np.eye(len(texts), 4, dtype=np.float32)
        return rows

    def similarity(self, a, b):
        return np.atleast_2d(a) @ np.atleast_2d(b).T


def batches(texts):
    """Mimic MTEB's DataLoader: an iterable of dicts with a 'text' key."""
    return [{"text": texts}]


def promoting_config(feature: str = "defined_name_hit", weight: float = 5.0) -> RerankerConfig:
    weights = [0.0] * len(FEATURE_NAMES)
    weights[FEATURE_NAMES.index("dense")] = 1.0
    weights[FEATURE_NAMES.index(feature)] = weight
    return RerankerConfig(weights=tuple(weights), candidate_depth=4)


class TestScoringUsesRawQueryText:
    """The encoder scores the text MTEB gave it, unmodified.

    ``QueryFacts.cleaned_text`` strips boilerplate and is deliberately not
    used for scoring: it changes 82% of test queries and removes a median 15%
    of their characters, which needs the same paired-bootstrap gate as any
    other pass before it can be switched on.  These pin that it stays out of
    the scoring path, so nobody quietly wires it in as an obvious improvement.
    """

    def test_encoder_receives_the_query_verbatim(self):
        seen: list[str] = []

        class RecordingEncoder(StubEncoder):
            def encode(self, texts, mode="document", **kwargs):
                items = [texts] if isinstance(texts, str) else list(texts)
                if mode == "query":
                    seen.extend(items)
                return super().encode(texts, mode=mode, **kwargs)

        noisy = [q + "\n-----Note-----\nThis note is retrieval-inert.\n" for q in QUERIES]
        pipeline = PrePostPipelineEncoder(RecordingEncoder(), promoting_config())
        pipeline.encode(batches(noisy), prompt_type="query")
        assert seen == noisy, "the scoring path must not silently clean the query"
        assert all("-----Note-----" in text for text in seen)

    def test_cleaned_text_is_computed_but_differs_from_what_was_scored(self):
        from app.pipeline.query_analysis import analyze_query

        text = QUERIES[0] + "\n-----Note-----\nInert.\n"
        facts = analyze_query(text)
        assert facts.cleaned_text != text, "cleaning is still available for inspection"
        assert "-----Note-----" not in facts.cleaned_text


class TestMtebContract:
    def test_encode_returns_one_row_per_text(self):
        pipeline = PrePostPipelineEncoder(StubEncoder(), promoting_config())
        documents = pipeline.encode(batches(DOCUMENTS), prompt_type="document")
        queries = pipeline.encode(batches(QUERIES), prompt_type="query")
        assert documents.shape[0] == len(DOCUMENTS)
        assert queries.shape[0] == len(QUERIES)

    def test_prompt_type_selects_the_mode(self):
        encoder = StubEncoder()
        pipeline = PrePostPipelineEncoder(encoder, promoting_config())
        pipeline.encode(batches(DOCUMENTS), prompt_type="PromptType.document")
        pipeline.encode(batches(QUERIES), prompt_type="PromptType.query")
        assert encoder.document_calls == 1
        assert encoder.query_calls == 1

    def test_second_pass_reorders_the_matrix(self):
        pipeline = PrePostPipelineEncoder(StubEncoder(), promoting_config())
        documents = pipeline.encode(batches(DOCUMENTS), prompt_type="document")
        queries = pipeline.encode(batches(QUERIES), prompt_type="query")
        scores = pipeline.similarity(queries, documents)
        assert scores.shape == (len(QUERIES), len(DOCUMENTS))
        # Query 0 names count_squares, which document 1 defines. The stub gives
        # it no dense advantage, so only the second pass can surface it.
        assert int(np.argmax(scores[0])) == 1

    def test_identity_config_leaves_the_matrix_untouched(self):
        pipeline = PrePostPipelineEncoder(StubEncoder(), RerankerConfig())
        documents = pipeline.encode(batches(DOCUMENTS), prompt_type="document")
        queries = pipeline.encode(batches(QUERIES), prompt_type="query")
        scores = pipeline.similarity(queries, documents)
        assert np.array_equal(scores, np.atleast_2d(queries) @ np.atleast_2d(documents).T)

    def test_a_row_count_mismatch_skips_the_rerank_instead_of_crashing(self, caplog):
        pipeline = PrePostPipelineEncoder(StubEncoder(), promoting_config())
        pipeline.encode(batches(DOCUMENTS), prompt_type="document")
        pipeline.encode(batches(QUERIES), prompt_type="query")
        # A third query arrives without having been recorded.
        scores = pipeline.similarity(np.eye(3, 4, dtype=np.float32), np.eye(4, dtype=np.float32))
        assert scores.shape == (3, 4)

    def test_empty_input_returns_an_empty_matrix(self):
        class DimensionedStub(StubEncoder):
            class model:  # noqa: N801 - mimics SentenceTransformer's attribute
                @staticmethod
                def get_sentence_embedding_dimension():
                    return 4

        pipeline = PrePostPipelineEncoder(DimensionedStub(), promoting_config())
        assert pipeline.encode(batches([]), prompt_type="query").shape == (0, 4)

    def test_multiple_document_batches_are_concatenated_in_order(self):
        pipeline = PrePostPipelineEncoder(StubEncoder(), promoting_config())
        pipeline.encode(
            [{"text": DOCUMENTS[:2]}, {"text": DOCUMENTS[2:]}], prompt_type="document"
        )
        assert pipeline._document_texts == DOCUMENTS


class TestOfflineRescore:
    def test_rescore_matrix_matches_the_online_path(self):
        pipeline = PrePostPipelineEncoder(StubEncoder(), promoting_config())
        documents = pipeline.encode(batches(DOCUMENTS), prompt_type="document")
        queries = pipeline.encode(batches(QUERIES), prompt_type="query")
        online = pipeline.similarity(queries, documents)

        offline_pipeline = PrePostPipelineEncoder(StubEncoder(), promoting_config())
        base = np.atleast_2d(queries) @ np.atleast_2d(documents).T
        offline = offline_pipeline.rescore_matrix(base, QUERIES, DOCUMENTS)
        assert np.allclose(online, offline)

    def test_disabled_rerank_is_a_passthrough(self):
        pipeline = PrePostPipelineEncoder(StubEncoder(), RerankerConfig())
        base = np.array([[0.9, 0.1, 0.5, 0.2]], dtype=np.float32)
        assert np.array_equal(pipeline.rescore_matrix(base, QUERIES[:1], DOCUMENTS), base)


CODEFORCES_QUERY = """Sum each test case.

-----Example-----
Input
2
3
1 2 3
2
4 5

Output
6
9
"""


class TestExecutionStage:
    """The third pass runs only when asked, and only ever promotes."""

    @staticmethod
    def _pipeline(**kwargs):
        from app.pipeline.execution_rerank import ExecutionConfig

        return PrePostPipelineEncoder(
            StubEncoder(), RerankerConfig(), execution_config=ExecutionConfig(**kwargs)
        )

    def test_disabled_by_default(self):
        pipeline = PrePostPipelineEncoder(StubEncoder(), RerankerConfig())
        assert pipeline.execution_config.enabled is False

    def test_disabled_pipeline_is_an_exact_passthrough(self):
        pipeline = self._pipeline(enabled=False)
        documents = pipeline.encode(batches(DOCUMENTS), prompt_type="document")
        queries = pipeline.encode(batches([CODEFORCES_QUERY]), prompt_type="query")
        scores = pipeline.similarity(queries[:1], documents)
        assert np.array_equal(scores, np.atleast_2d(queries[:1]) @ np.atleast_2d(documents).T)
        assert pipeline.execution_audit == {}

    def test_a_verified_candidate_is_promoted_to_rank_one(self):
        solver = (
            "n = int(input())\n"
            "for _ in range(n):\n"
            "    input()\n"
            "    print(sum(map(int, input().split())))"
        )
        documents = [*DOCUMENTS, solver]
        pipeline = self._pipeline(enabled=True, max_candidates=5, timeout_seconds=5.0)
        doc_emb = pipeline.encode(batches(documents), prompt_type="document")
        q_emb = pipeline.encode(batches([CODEFORCES_QUERY]), prompt_type="query")
        scores = pipeline.similarity(q_emb[:1], doc_emb)
        assert int(np.argmax(scores[0])) == len(documents) - 1
        assert pipeline.execution_audit["candidates_verified"] == 1

    def test_a_statement_without_examples_is_skipped_entirely(self):
        pipeline = self._pipeline(enabled=True, max_candidates=3, timeout_seconds=5.0)
        doc_emb = pipeline.encode(batches(DOCUMENTS), prompt_type="document")
        q_emb = pipeline.encode(batches(["just some prose with no examples"]), prompt_type="query")
        pipeline.similarity(q_emb[:1], doc_emb)
        assert pipeline.execution_audit["queries_with_examples"] == 0
        assert pipeline.execution_audit["candidates_executed"] == 0

    def test_nothing_is_demoted_when_no_candidate_verifies(self):
        pipeline = self._pipeline(enabled=True, max_candidates=4, timeout_seconds=5.0)
        doc_emb = pipeline.encode(batches(DOCUMENTS), prompt_type="document")
        q_emb = pipeline.encode(batches([CODEFORCES_QUERY]), prompt_type="query")
        before = np.atleast_2d(q_emb[:1]) @ np.atleast_2d(doc_emb).T
        after = pipeline.similarity(q_emb[:1], doc_emb)
        assert pipeline.execution_audit["candidates_verified"] == 0
        assert np.argsort(-before[0]).tolist() == np.argsort(-after[0]).tolist()
