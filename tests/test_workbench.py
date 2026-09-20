"""The hands-on demo has to be as honest as the leaderboard run.

These tests are mostly about what the workbench *says*, not what it ranks: a
demo whose badges are decorative is worse than no demo, because a judge cannot
tell the difference by looking. So the assertions here are that a pass which
did not run is reported as not having run, that the score breakdown adds up to
the score, and that a version boundary is a real boundary.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from app.indexing.cache import EmbeddingCache
from app.indexing.versioned import VersionedIndex
from app.retrieval.hybrid import Document, HybridRetriever, RRF_K
from app.workbench import EdgeIndex, Workbench, _hit_to_dict


def documents(count: int = 6, suffix: str = "") -> list[Document]:
    return [
        Document(
            document_id=f"c{i}",
            file_path=f"src/module{i}.py",
            content=f"def handler{i}(payload):\n    return transform{i}(payload){suffix}\n",
            name=f"handler{i}",
            type="FUNCTION",
            signature=f"handler{i}(payload)",
        )
        for i in range(count)
    ]


class TestFusionIsShownNotAsserted:
    """The evidence panel claims the terms add up. They have to."""

    def test_contributions_sum_to_the_reported_score(self):
        retriever = HybridRetriever(documents())
        for hit in retriever.search("handler3 transform payload", limit=5):
            total = sum((hit.contributions or {}).values())
            assert total == pytest.approx(hit.score, abs=1e-12)

    def test_each_contribution_is_the_rrf_term_for_its_rank(self):
        retriever = HybridRetriever(documents())
        hit = retriever.search("handler2", limit=3)[0]
        for method, rank in (hit.ranks or {}).items():
            assert hit.contributions[method] == pytest.approx(1.0 / (RRF_K + rank))

    def test_only_methods_that_retrieved_it_appear(self):
        """RRF must never invent a rank for a method that missed the document."""
        retriever = HybridRetriever(documents())
        for hit in retriever.search("handler1", limit=6):
            assert set(hit.ranks or {}) == set(hit.sources)
            assert set(hit.contributions or {}) == set(hit.sources)

    def test_serialised_shares_are_relative_to_the_score(self):
        retriever = HybridRetriever(documents())
        payload = _hit_to_dict(retriever.search("handler4", limit=3)[0])
        assert sum(signal["share"] for signal in payload["signals"]) == pytest.approx(1.0, abs=2e-3)


class TestPostingsListDidNotChangeTheRanking:
    """The inverted index is a speed change. It must not be a ranking change."""

    def test_bm25_matches_the_document_scan_it_replaced(self):
        from collections import Counter
        import math

        corpus = documents(12)
        retriever = HybridRetriever(corpus)
        tokens = ["handler", "payload", "transform", "handler7"]

        expected: dict[int, float] = {}
        query_frequency = Counter(tokens)
        total = len(corpus)
        for index, document_tokens in enumerate(retriever._tokens):
            frequencies = Counter(document_tokens)
            length = len(document_tokens)
            score = 0.0
            for token, query_count in query_frequency.items():
                term_frequency = frequencies[token]
                if not term_frequency:
                    continue
                document_frequency = retriever._document_frequency[token]
                inverse = math.log(
                    1 + (total - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = term_frequency + 1.5 * (
                    0.25 + 0.75 * length / max(1.0, retriever._average_length)
                )
                score += inverse * (term_frequency * 2.5 / denominator) * (1 + math.log1p(query_count))
            if score > 0:
                expected[index] = score

        assert retriever._bm25(tokens) == expected


class TestSnapshotsAreCachedButNotShared:
    def test_reindexing_a_commit_drops_its_cached_retriever(self):
        index = VersionedIndex()
        index.add_snapshot("v1", documents(3))
        first = index.retrieve("handler1", "v1", limit=3)
        assert first

        index.add_snapshot("v1", [documents(3)[0]])
        again = index.retrieve("handler1", "v1", limit=3)
        assert len(again) == 1, "a stale retriever would still see three documents"

    def test_warm_builds_the_retriever_without_querying(self):
        index = VersionedIndex()
        index.add_snapshot("v1", documents(2))
        assert index.warm("v1") is True
        assert index.warm("missing") is False


class TestEdgeIndex:
    def edges(self) -> list[dict]:
        return [
            {"source": "main", "target": "parse", "relation": "CALLS", "file_path": "a.py",
             "line": 10, "status": "VERIFIED", "confidence": 1.0},
            {"source": "cli", "target": "parse", "relation": "CALLS", "file_path": "b.py",
             "line": 3, "status": "CANDIDATE", "confidence": 0.5},
            {"source": "parse", "target": "tokenize", "relation": "CALLS", "file_path": "a.py",
             "line": 12, "status": "VERIFIED", "confidence": 1.0},
            {"source": "a.py", "target": "os", "relation": "IMPORTS", "file_path": "a.py",
             "line": 1, "status": "VERIFIED", "confidence": 1.0},
        ]

    def test_callers_and_callees_are_distinct_directions(self):
        index = EdgeIndex(self.edges())
        around = index.around("parse", "a.py")
        assert sorted(e["symbol"] for e in around["callers"]) == ["cli", "main"]
        assert [e["symbol"] for e in around["calls"]] == ["tokenize"]

    def test_candidate_edges_keep_their_status(self):
        index = EdgeIndex(self.edges())
        statuses = {e["symbol"]: e["status"] for e in index.around("parse", "a.py")["callers"]}
        assert statuses == {"main": "VERIFIED", "cli": "CANDIDATE"}

    def test_an_unnamed_chunk_asks_for_nothing(self):
        index = EdgeIndex(self.edges())
        assert index.around("", "a.py") == {"callers": [], "calls": [], "imports": []}

    def test_total_is_reported_for_the_pass_strip(self):
        assert EdgeIndex(self.edges()).total == 4
        assert EdgeIndex([]).total == 0


class TestWorkbenchOnARealSubtree:
    """End to end against this repository's own JavaScript fixture."""

    @pytest.fixture(scope="class")
    def bench(self) -> Workbench:
        workbench = Workbench(cache=EmbeddingCache(":memory:"))
        workbench.start_build("commander", versions=1, dense=False)
        deadline = time.time() + 180
        while time.time() < deadline:
            state = workbench.state()
            if state["phase"] in {"ready", "error"}:
                break
            time.sleep(0.25)
        state = workbench.state()
        if state["phase"] != "ready":
            pytest.skip(f"commander fixture did not build: {state.get('error')}")
        return workbench

    def test_a_pass_that_could_not_run_says_so(self, bench: Workbench):
        """The execution pass runs Python. This corpus is JavaScript."""
        result = bench.search("commander", "how does it print the help text", limit=5)
        execution = next(p for p in result["passes"] if p["name"] == "Execution")
        assert execution["ran"] is False
        assert "JavaScript" in execution["detail"]
        assert all(r["verification"]["state"] == "not_run" for r in result["results"])

    def test_a_lexical_only_build_does_not_claim_a_dense_pass(self, bench: Workbench):
        result = bench.search("commander", "where are options parsed", limit=5)
        dense = next(p for p in result["passes"] if p["name"] == "Dense")
        assert dense["ran"] is False
        assert result["encoder"]["active"] is False
        assert all("dense" not in r["sources"] for r in result["results"])

    def test_results_carry_the_span_they_were_cited_from(self, bench: Workbench):
        result = bench.search("commander", "how does it print the help text", limit=5)
        assert result["results"], "the fixture should match this query"
        for row in result["results"]:
            assert row["file_path"].endswith((".js", ".ts", ".mjs", ".cjs"))
            assert row["end_line"] >= row["start_line"] >= 1
            assert row["code"]

    def test_retrieval_is_pinned_and_leak_free(self, bench: Workbench):
        """Isolation is checked over every candidate, not just the shown page."""
        result = bench.search("commander", "unknown option", limit=10)
        assert result["isolation"]["leaked"] == 0
        assert result["isolation"]["checked"] >= len(result["results"])
        assert result["isolation"]["checked"] == result["depth"]

    def test_nothing_is_promoted_when_the_execution_pass_cannot_run(self, bench: Workbench):
        result = bench.search("commander", "how does it print the help text", limit=10)
        assert all(row["promoted_from"] is None for row in result["results"])
        assert all(row["promotion_bonus"] == 0.0 for row in result["results"])
        assert all(row["rank"] == row["retrieved_rank"] for row in result["results"])
        for row in result["results"]:
            assert row["score"] == row["retrieval_score"]

    def test_timings_are_measured_not_declared(self, bench: Workbench):
        result = bench.search("commander", "how does it print the help text", limit=5)
        timings = result["timings"]
        parts = timings["candidates_ms"] + timings["structural_ms"] + timings["verification_ms"]
        assert timings["total_ms"] >= parts - 1.0
        assert timings["total_ms"] < 2000, "a warm lexical query should not take seconds"

    def test_an_unknown_version_is_refused_rather_than_answered(self, bench: Workbench):
        with pytest.raises(KeyError):
            bench.search("commander", "help", version="deadbeef", limit=3)

    def test_across_versions_returns_both_rankings(self, bench: Workbench):
        data = bench.across("commander", "how does it print the help text", limit=5)
        assert data["versions_searched"] == 1
        assert data["collapsed"] and data["ungrouped"]
        assert data["distinct_collapsed"] >= data["distinct_ungrouped"]


class TestUnknownTargets:
    def test_an_unknown_target_is_a_key_error(self):
        workbench = Workbench(cache=EmbeddingCache(":memory:"))
        with pytest.raises(KeyError):
            workbench.start_build("not-a-corpus")

    def test_searching_before_building_names_the_fix(self):
        workbench = Workbench(cache=EmbeddingCache(":memory:"))
        with pytest.raises(KeyError, match="build"):
            workbench.search("commander", "anything")

    def test_targets_report_why_they_are_unavailable(self):
        workbench = Workbench(cache=EmbeddingCache(":memory:"))
        for target in workbench.targets():
            assert target["available"] or target["unavailable_reason"]


class TestDenseSnapshotsReportThemselves:
    def test_supplying_vectors_flips_uses_dense_for_that_snapshot_only(self):
        index = VersionedIndex()
        rows = np.eye(3, 8, dtype=np.float32)
        index.add_snapshot("v1", documents(3), embeddings=rows)
        index.add_snapshot("v2", documents(3))
        assert index.uses_dense("v1") is True
        assert index.uses_dense("v2") is False
        assert index.uses_dense() is False, "a partly-dense index is not a dense index"


class TestExecutionPromotion:
    """The pass the submitted score rests on, shown end to end on a fixture.

    No artifacts required: the corpus is four tiny programs, one of which
    actually solves the problem in the query. Whether it reaches rank 1 is
    decided by running it, not by anything the fixture asserts.
    """

    #: The layout CoIR actually ships AppsRetrieval statements in.
    QUERY = (
        "Read an integer n from standard input and print its double.\n\n"
        "-----Examples-----\n"
        "Input\n21\n\n"
        "Output\n42\n"
    )

    def corpus(self) -> list[Document]:
        bodies = {
            # Ranked above the answer lexically -- it repeats the words, and
            # is wrong.
            "wrong_double_integer_input": "n = int(input())\nprint(n + 2)\n",
            "also_wrong_integer": "n = int(input())\nprint(n * n)\n",
            "correct": "n = int(input())\nprint(n * 2)\n",
            "unrelated": "print('hello')\n",
        }
        return [
            Document(
                document_id=name,
                file_path=f"solutions/{name}.py",
                content=body,
                name=name,
                type="SOLUTION",
                end_line=body.count("\n"),
            )
            for name, body in bodies.items()
        ]

    def bench(self) -> Workbench:
        from app.workbench import TARGETS, TargetIndex

        workbench = Workbench(cache=EmbeddingCache(":memory:"))
        index = VersionedIndex()
        index.add_snapshot("v1", self.corpus())
        built = TargetIndex(target=TARGETS["apps"], index=index)
        built.versions = [
            __import__("app.workbench", fromlist=["VersionInfo"]).VersionInfo(
                ref="v1", sha="", subject="fixture", author="", timestamp=0,
                chunks=len(self.corpus()),
            )
        ]
        built.edges_by_version["v1"] = EdgeIndex([])
        workbench._built["fixture"] = built
        return workbench

    def test_the_program_that_reproduces_the_example_reaches_rank_one(self):
        result = self.bench().search("fixture", self.QUERY, limit=4)
        assert result["results"][0]["name"] == "correct"
        assert result["results"][0]["verification"]["state"] == "verified"

    def test_the_promotion_is_reported_as_movement(self):
        result = self.bench().search("fixture", self.QUERY, limit=4)
        winner = result["results"][0]
        assert winner["promotion_bonus"] > 0
        assert winner["score"] == pytest.approx(
            winner["retrieval_score"] + winner["promotion_bonus"]
        )
        if winner["retrieved_rank"] != 1:
            assert winner["promoted_from"] == winner["retrieved_rank"]

    def test_switching_verification_off_leaves_the_retrieval_order_alone(self):
        workbench = self.bench()
        verified = workbench.search("fixture", self.QUERY, limit=4)
        plain = workbench.search("fixture", self.QUERY, limit=4, verify=False)

        assert [r["name"] for r in plain["results"]] == sorted(
            (r["name"] for r in plain["results"]),
            key=lambda n: -next(x["retrieval_score"] for x in plain["results"] if x["name"] == n),
        )
        assert all(r["promotion_bonus"] == 0.0 for r in plain["results"])
        execution = next(p for p in plain["passes"] if p["name"] == "Execution")
        assert execution["ran"] is False
        # And the promoted run genuinely differs from the un-promoted one.
        assert [r["name"] for r in verified["results"]] != [r["name"] for r in plain["results"]]

    def test_a_query_without_worked_examples_never_executes_anything(self):
        result = self.bench().search("fixture", "how do I double a number", limit=4)
        execution = next(p for p in result["passes"] if p["name"] == "Execution")
        assert execution["ran"] is False
        assert "no machine-readable worked examples" in execution["detail"]
        assert all(r["verification"]["state"] == "not_run" for r in result["results"])


class TestOversizedChunksGetNoDenseVector:
    """Past the encoder's cap, truncation under last-token pooling is wrong.

    So those chunks keep their place in the index and get a zero row: the
    dense pass returns no rank for them, fusion invents none, and the lexical
    passes still find them.
    """

    def corpus(self) -> list[Document]:
        """One ordinary chunk, and one comfortably past the cap."""
        return [
            Document(document_id="small", file_path="a.py", content="def tiny(): pass\n",
                     name="tiny", type="FUNCTION"),
            Document(document_id="huge", file_path="b.py",
                     content="def enormous():\n" + ("    x = 1  # padding\n" * 900),
                     name="enormous", type="FUNCTION"),
        ]

    def test_only_the_oversized_chunk_is_marked_ineligible(self):
        eligible = Workbench._dense_eligible(self.corpus())
        assert eligible == [True, False]

    def test_the_oversized_row_is_zero_and_the_matrix_stays_aligned(self):
        workbench = Workbench(cache=EmbeddingCache(":memory:"))
        corpus = self.corpus()

        class StubEncoder:
            dimension = 4

            @staticmethod
            def encode(texts, mode="document"):
                return np.ones((len(texts), 4), dtype=np.float32)

        workbench._encoder = StubEncoder()
        matrix = workbench._encode_documents(corpus, Workbench._dense_eligible(corpus))

        assert matrix.shape == (len(corpus), 4), "one row per document, always"
        assert matrix[0].any(), "the small chunk was encoded"
        assert not matrix[1].any(), "the oversized chunk carries no vector"

    def test_a_zero_row_scores_exactly_zero_and_is_dropped_from_the_dense_list(self):
        corpus = self.corpus()
        embeddings = np.zeros((2, 4), dtype=np.float32)
        embeddings[0] = [1.0, 0.0, 0.0, 0.0]

        class QueryEncoder:
            @staticmethod
            def encode(text, mode="query"):
                return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

            @staticmethod
            def similarity(a, b):
                return np.atleast_2d(a) @ np.atleast_2d(b).T

        retriever = HybridRetriever(corpus, encoder=QueryEncoder(), document_embeddings=embeddings)
        dense = retriever._dense("anything", top_k=10)
        assert 0 in dense
        assert 1 not in dense, "a zero row must not earn a dense rank"

    def test_the_oversized_chunk_is_still_findable_lexically(self):
        corpus = self.corpus()
        embeddings = np.zeros((2, 4), dtype=np.float32)
        embeddings[0] = [1.0, 0.0, 0.0, 0.0]
        retriever = HybridRetriever(corpus, document_embeddings=embeddings)
        retriever.use_dense = False  # isolate the lexical passes
        names = [hit.document.name for hit in retriever.search("enormous", limit=2)]
        assert "enormous" in names


class TestVersionsAreOnlyComparedLikeForLike:
    """A dense side against a lexical side is a different system, not a diff."""

    def index(self, dense_left: bool, dense_right: bool) -> VersionedIndex:
        class StubEncoder:
            """Keeps the real 0.6B model out of a test about reporting."""

            @staticmethod
            def encode(text, mode="query"):
                return np.eye(1, 8, dtype=np.float32)[0]

            @staticmethod
            def similarity(a, b):
                return np.atleast_2d(a) @ np.atleast_2d(b).T

        index = VersionedIndex(encoder=StubEncoder())
        rows = np.eye(4, 8, dtype=np.float32)
        index.add_snapshot("a", documents(4), embeddings=rows if dense_left else None)
        index.add_snapshot("b", documents(4), embeddings=rows if dense_right else None)
        return index

    def bench(self, index: VersionedIndex) -> Workbench:
        from app.workbench import TARGETS, TargetIndex, VersionInfo

        workbench = Workbench(cache=EmbeddingCache(":memory:"))
        built = TargetIndex(target=TARGETS["self"], index=index)
        built.versions = [
            VersionInfo(ref=ref, sha="", subject="", author="", timestamp=0, chunks=4)
            for ref in ("b", "a")
        ]
        built.edges_by_version = {"a": EdgeIndex([]), "b": EdgeIndex([])}
        workbench._built["fixture"] = built
        return workbench

    def test_matched_passes_compare_cleanly(self):
        data = self.bench(self.index(False, False)).compare("fixture", "handler1", "a", "b")
        assert data["comparable"] is True
        assert data["comparable_reason"] == ""

    def test_a_dense_side_against_a_lexical_side_is_flagged(self):
        data = self.bench(self.index(False, True)).compare("fixture", "handler1", "a", "b")
        assert data["comparable"] is False
        assert "lexical only" in data["comparable_reason"]
        assert "not the code" in data["comparable_reason"]
