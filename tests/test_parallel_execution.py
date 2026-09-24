"""Parallel execution changes the wall clock and nothing else.

Each candidate run is an independent subprocess and each result is written back
to its own slot in dense order, so a pass with N workers must produce exactly
the scores and audit counts of a sequential pass.  These tests pin that, plus
the corpus-smaller-than-depth case that used to crash ``argpartition``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from app.pipeline.encoder import PrePostPipelineEncoder
from app.pipeline.execution_rerank import ExecutionConfig, apply_execution_rerank
from app.pipeline.query_analysis import analyze_query
from app.pipeline.reranker import RerankerConfig

STATEMENT = """Print the sum of two numbers.

-----Example-----
Input
2 3

Output
5
"""
OTHER_STATEMENT = """Print the product of two numbers.

-----Example-----
Input
2 3

Output
6
"""
PROSE = "No worked examples here at all."

SUM = "a, b = map(int, input().split())\nprint(a + b)"
PRODUCT = "a, b = map(int, input().split())\nprint(a * b)"
SUM_AGAIN = "print(sum(map(int, input().split())))"
CRASH = "raise SystemExit(2)"
WRONG = "print(0)"
CANDIDATES = [WRONG, SUM, CRASH, PRODUCT, SUM_AGAIN, WRONG, "print(5)", CRASH]
SCORES = [0.9, 0.85, 0.8, 0.75, 0.7, 0.4, 0.3, 0.2]


def _config(**kwargs) -> ExecutionConfig:
    base = ExecutionConfig(enabled=True, max_candidates=8, timeout_seconds=5.0, verified_bonus=1.0)
    return replace(base, **kwargs)


class TestWithinQuery:
    def test_parallel_equals_sequential(self):
        sequential = apply_execution_rerank(STATEMENT, CANDIDATES, SCORES, _config(workers=1))
        parallel = apply_execution_rerank(STATEMENT, CANDIDATES, SCORES, _config(workers=4))
        assert parallel == sequential
        assert sequential[1]["candidates_verified"] == 3  # SUM, SUM_AGAIN, print(5)

    def test_parallel_respects_the_dense_gate(self):
        sequential = apply_execution_rerank(
            STATEMENT, CANDIDATES, SCORES, _config(workers=1, max_dense_gap=0.25)
        )
        parallel = apply_execution_rerank(
            STATEMENT, CANDIDATES, SCORES, _config(workers=4, max_dense_gap=0.25)
        )
        assert parallel == sequential
        assert sequential[1]["candidates_executed"] == 5
        assert sequential[1]["candidates_skipped_by_gate"] == 3

    def test_parallel_respects_the_promotion_gate(self):
        sequential = apply_execution_rerank(
            STATEMENT, CANDIDATES, SCORES, _config(workers=1, max_promotion_gap=0.1)
        )
        parallel = apply_execution_rerank(
            STATEMENT, CANDIDATES, SCORES, _config(workers=4, max_promotion_gap=0.1)
        )
        assert parallel == sequential

    def test_early_exit_stays_sequential(self):
        sequential = apply_execution_rerank(
            STATEMENT, CANDIDATES, SCORES, _config(workers=1, stop_after_first_match=True)
        )
        parallel = apply_execution_rerank(
            STATEMENT, CANDIDATES, SCORES, _config(workers=4, stop_after_first_match=True)
        )
        assert parallel == sequential
        assert sequential[1]["candidates_executed"] == 2


def _pipeline(workers: int, max_candidates: int = 4) -> PrePostPipelineEncoder:
    config = ExecutionConfig(
        enabled=True, max_candidates=max_candidates, timeout_seconds=5.0, workers=workers
    )
    pipeline = PrePostPipelineEncoder(object(), RerankerConfig(), execution_config=config)
    return pipeline


def _load(pipeline: PrePostPipelineEncoder, queries: list[str], documents: list[str]) -> None:
    pipeline._query_texts = list(queries)
    pipeline._query_facts = [analyze_query(q) for q in queries]
    pipeline._document_texts = list(documents)


class TestAcrossQueries:
    def test_parallel_rows_equal_sequential_rows(self):
        queries = [STATEMENT, OTHER_STATEMENT, PROSE, STATEMENT, OTHER_STATEMENT]
        documents = [WRONG, SUM, PRODUCT, CRASH, SUM_AGAIN, "print(6)"]
        rng = np.random.default_rng(3)
        scores = rng.random((len(queries), len(documents))).astype(np.float32)

        sequential = _pipeline(workers=1)
        _load(sequential, queries, documents)
        expected = sequential._apply_execution(scores)

        parallel = _pipeline(workers=4)
        _load(parallel, queries, documents)
        got = parallel._apply_execution(scores)

        assert np.array_equal(got, expected)
        for key in ("queries_with_examples", "candidates_executed", "candidates_verified"):
            assert parallel.execution_audit[key] == sequential.execution_audit[key]
        assert sequential.execution_audit["queries_with_examples"] == 4

    def test_a_corpus_smaller_than_the_depth_does_not_crash(self):
        pipeline = _pipeline(workers=2, max_candidates=20)
        _load(pipeline, [STATEMENT], [WRONG, SUM])
        scores = np.array([[0.9, 0.1]], dtype=np.float32)
        out = pipeline._apply_execution(scores)
        assert int(np.argmax(out[0])) == 1
        assert pipeline.execution_audit["candidates_executed"] == 2

    def test_an_empty_corpus_is_a_no_op(self):
        pipeline = _pipeline(workers=2)
        _load(pipeline, [STATEMENT], [])
        scores = np.zeros((1, 0), dtype=np.float32)
        assert pipeline._apply_execution(scores).shape == (1, 0)


class TestChunkedCorpus:
    """MTEB scores corpora over 50k documents in chunks, calling encode() and
    similarity() once per chunk.  Each chunk must be self-consistent: recorded
    texts line up with the chunk's score columns, and a verified document in
    any chunk still outranks that chunk's unverified documents."""

    def test_each_chunk_is_scored_against_its_own_texts(self):
        pipeline = _pipeline(workers=2, max_candidates=3)
        pipeline._query_texts = [STATEMENT]
        pipeline._query_facts = [analyze_query(STATEMENT)]
        chunks = [[WRONG, CRASH, PRODUCT], [SUM, WRONG, "print(7)"]]
        verified_at = []
        for chunk in chunks:
            pipeline._document_texts = list(chunk)  # what encode() records per chunk
            scores = np.array([[0.9, 0.8, 0.7]], dtype=np.float32)
            out = pipeline._apply_execution(scores)
            verified_at.append(int(np.argmax(out[0])))
            assert pipeline.execution_audit["candidates_executed"] == 3
        # chunk 1 has nothing that passes: dense order kept; chunk 2 promotes SUM
        assert verified_at == [0, 0]
        assert pipeline.execution_audit["candidates_verified"] == 1


class TestAdaptiveDeepening:
    """POST-7: execute ranks beyond the window only when nothing in it verified."""

    DEEP = [WRONG, CRASH, WRONG, SUM, "print(5)"]  # first window: 3; SUM at rank 4
    SCORES = [0.9, 0.85, 0.8, 0.75, 0.3]

    def test_deepens_only_when_the_window_found_nothing(self):
        config = _config(max_candidates=3, deepen_to=5)
        scores, audit = apply_execution_rerank(STATEMENT, self.DEEP, self.SCORES, config)
        assert audit["deepened"] == 2
        assert scores[3] > scores[0] and scores[4] > scores[0]  # SUM and print(5) promoted

    def test_no_deepening_when_the_window_verified(self):
        candidates = [SUM, WRONG, WRONG, "print(5)", SUM_AGAIN]
        scores, audit = apply_execution_rerank(STATEMENT, candidates, self.SCORES, _config(max_candidates=3, deepen_to=5))
        assert "deepened" not in audit
        assert audit["candidates_executed"] == 3

    def test_the_gate_limits_the_deep_window(self):
        config = _config(max_candidates=3, deepen_to=5, max_dense_gap=0.2)
        scores, audit = apply_execution_rerank(STATEMENT, self.DEEP, self.SCORES, config)
        assert audit["deepened"] == 1  # rank 4 is within 0.2 of the best, rank 5 (0.3) is not
        assert scores[3] > scores[0] and scores[4] == self.SCORES[4]

    def test_parallel_deepening_equals_sequential(self):
        sequential = apply_execution_rerank(STATEMENT, self.DEEP, self.SCORES, _config(max_candidates=3, deepen_to=5, workers=1))
        parallel = apply_execution_rerank(STATEMENT, self.DEEP, self.SCORES, _config(max_candidates=3, deepen_to=5, workers=4))
        assert parallel == sequential

    def test_encoder_passes_the_deep_window_through(self):
        pipeline = PrePostPipelineEncoder(
            object(), RerankerConfig(),
            execution_config=ExecutionConfig(enabled=True, max_candidates=2, deepen_to=4, timeout_seconds=5.0),
        )
        _load(pipeline, [STATEMENT], [WRONG, CRASH, SUM, WRONG])
        out = pipeline._apply_execution(np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32))
        assert int(np.argmax(out[0])) == 2
