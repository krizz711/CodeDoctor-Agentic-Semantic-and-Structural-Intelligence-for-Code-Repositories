"""The Bonus goal: retrieval across every indexed version at once.

The brief names the hard part rather than leaving it implied — across versions
the snippets "would still be very similar, which would make them hard to rank
properly".  Searching all snapshots naively makes that worse: a function that
survived ten commits unchanged returns ten near-identical rows and pushes
everything else out of the top-k.

These tests pin the behaviour that fixes it, and the failure mode it fixes.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.indexing.versioned import CrossVersionHit, VersionedIndex
from app.retrieval.hybrid import Document


def doc(doc_id: str, name: str, content: str, path: str = "lib/parse.js") -> Document:
    return Document(
        document_id=doc_id,
        file_path=path,
        content=content,
        name=name,
        type="FUNCTION",
        signature=f"{name}()",
    )


PARSE_V1 = "function parseOptions(argv) { return argv.filter(isFlag); }"
PARSE_V2 = "function parseOptions(argv) { return argv.filter(isFlag).map(trim); }"
HELP = "function printHelp() { console.log(helpText); }"
UNRELATED = "function connectDatabase(url) { return pool.acquire(url); }"


@pytest.fixture
def three_versions() -> VersionedIndex:
    """parseOptions is unchanged in v1/v2 and edited in v3."""
    index = VersionedIndex()
    for commit, parse in (("v1", PARSE_V1), ("v2", PARSE_V1), ("v3", PARSE_V2)):
        index.add_snapshot(commit, [
            doc("parse", "parseOptions", parse),
            doc("help", "printHelp", HELP, path="lib/help.js"),
            doc("db", "connectDatabase", UNRELATED, path="lib/db.js"),
        ])
    return index


class TestCrossVersionRetrieval:
    def test_it_searches_every_version(self, three_versions):
        hits = three_versions.retrieve_across_versions("parse options", limit=5)
        assert hits
        top = hits[0]
        assert isinstance(top, CrossVersionHit)
        assert top.document.name == "parseOptions"
        assert top.versions == ["v1", "v2", "v3"], "must report every version it appears in"

    def test_near_identical_snippets_collapse_to_one_row(self, three_versions):
        """The failure the Bonus goal is warning about."""
        collapsed = three_versions.retrieve_across_versions("parse options", limit=5)
        raw = three_versions.retrieve_across_versions(
            "parse options", limit=5, collapse_identical=False
        )
        names = [h.document.name for h in collapsed]
        assert len(names) == len(set(names)), "collapsed results must be distinct snippets"
        raw_names = [h.document.name for h in raw]
        assert len(raw_names) != len(set(raw_names)), (
            "without collapsing, the same snippet should repeat across versions -- "
            "if this stops being true the test no longer proves anything"
        )

    def test_a_distinct_snippet_is_not_crowded_out(self, three_versions):
        """Three versions x three snippets, a query matching all of them.

        With a budget of 3, collapsing returns the three distinct snippets.
        Without it, one snippet takes every slot -- which is the concrete
        failure the Bonus goal describes.
        """
        collapsed = three_versions.retrieve_across_versions("function", limit=3)
        raw = three_versions.retrieve_across_versions(
            "function", limit=3, collapse_identical=False
        )
        assert {h.document.name for h in collapsed} == {
            "parseOptions", "printHelp", "connectDatabase",
        }
        assert len({h.document.name for h in raw}) == 1, (
            "uncollapsed, the top 3 slots go to a single snippet repeated"
        )

    def test_variants_counts_distinct_contents_not_versions(self, three_versions):
        hits = three_versions.retrieve_across_versions("function", limit=5)
        parse = next(h for h in hits if h.document.name == "parseOptions")
        assert len(parse.versions) == 3
        assert parse.variants == 2, "v1 and v2 are byte-identical; v3 differs"
        assert parse.unchanged_across_versions is False

        helper = next(h for h in hits if h.document.name == "printHelp")
        assert helper.variants == 1
        assert helper.unchanged_across_versions is True

    def test_the_reported_content_is_the_best_scoring_variant(self, three_versions):
        hits = three_versions.retrieve_across_versions("map trim", limit=5)
        parse = next(h for h in hits if h.document.name == "parseOptions")
        assert "map(trim)" in parse.document.content, (
            "the query matches the v3 form, so v3's content should be the one shown"
        )

    def test_empty_index_returns_nothing_rather_than_raising(self):
        assert VersionedIndex().retrieve_across_versions("anything") == []

    def test_single_version_still_works(self):
        index = VersionedIndex()
        index.add_snapshot("only", [doc("parse", "parseOptions", PARSE_V1)])
        hits = index.retrieve_across_versions("parse options")
        assert len(hits) == 1
        assert hits[0].versions == ["only"]
        assert hits[0].unchanged_across_versions is False, "one version is not 'unchanged across'"


class TestDensePassReachesTheVersionedPath:
    """P1 and the demo must be able to use the encoder P0 is built on."""

    def test_lexical_only_is_reported_honestly(self, three_versions):
        assert three_versions.uses_dense() is False

    def test_supplying_embeddings_turns_the_dense_pass_on(self):
        index = VersionedIndex()
        docs = [doc("a", "alpha", PARSE_V1), doc("b", "beta", HELP)]
        vectors = np.eye(2, 8, dtype=np.float32)
        index.add_snapshot("v1", docs, embeddings=vectors)
        assert index.uses_dense("v1") is True
        assert index.uses_dense() is True

    def test_row_count_mismatch_is_refused(self):
        index = VersionedIndex()
        with pytest.raises(ValueError, match="embedding rows"):
            index.add_snapshot(
                "v1",
                [doc("a", "alpha", PARSE_V1), doc("b", "beta", HELP)],
                embeddings=np.eye(1, 8, dtype=np.float32),
            )

    def test_mixed_snapshots_report_per_version(self):
        index = VersionedIndex()
        index.add_snapshot("v1", [doc("a", "alpha", PARSE_V1)],
                           embeddings=np.eye(1, 8, dtype=np.float32))
        index.add_snapshot("v2", [doc("a", "alpha", PARSE_V2)])
        assert index.uses_dense("v1") is True
        assert index.uses_dense("v2") is False
        assert index.uses_dense() is False, "all() over versions, not any()"
