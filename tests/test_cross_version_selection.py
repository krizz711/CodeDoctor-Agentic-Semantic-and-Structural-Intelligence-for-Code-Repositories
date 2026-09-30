"""Bonus path: verification picks each lineage's representative, not recency.

Every version of a function shares one identity (file path and name). The
representative of that group must be the variant that reproduces the worked
example, wherever it sits in history, while the group keeps the list of every
version the identity appears in.
"""

from dataclasses import replace

import pytest

from app.indexing.versioned import VersionedIndex
from app.retrieval.hybrid import Document
from app.submission import execution_config

QUERY = "Print the sum.\n\n-----Example-----\nInput\n2 3\n\nOutput\n5\n"
GOOD = "a,b=map(int,input().split()); print(a+b)"
WRONG_SUB = "a,b=map(int,input().split()); print(a-b)"
WRONG_MUL = "a,b=map(int,input().split()); print(a*b)"
VERSIONS = ("v1", "v2", "v3")


def doc(code):
    return Document(document_id=code, file_path="solve.py", content=code, name="solve", type="FUNCTION")


def lineage(codes):
    index = VersionedIndex()
    for version, code in zip(VERSIONS, codes):
        index.add_snapshot(version, [doc(code)])
    return index


def config(**overrides):
    return replace(execution_config(), workers=1, **overrides)


@pytest.mark.parametrize("position", [0, 1, 2])
def test_the_correct_variant_represents_the_lineage_wherever_it_sits(position):
    codes = [WRONG_SUB, WRONG_MUL, WRONG_SUB]
    codes[position] = GOOD
    grouped, raw = lineage(codes).retrieve_across_versions(
        QUERY, execution=config(), return_ungrouped=True)
    top = grouped[0]
    assert top.document.content == GOOD
    assert top.versions == list(VERSIONS)          # provenance: every version, in history order
    assert top.variants == len({*codes})
    assert "execution" in top.sources
    assert len(raw) == 3 and raw[0].document.content == GOOD


def test_the_verified_content_is_run_once_even_when_it_repeats_across_versions():
    import app.pipeline.execution_rerank as er

    calls = []
    original = er.verify_candidate

    def observed(code, *args, **kwargs):
        calls.append(code)
        return original(code, *args, **kwargs)

    index = lineage([GOOD, GOOD, WRONG_SUB])
    try:
        er.verify_candidate = observed
        grouped = index.retrieve_across_versions(QUERY, execution=config())
    finally:
        er.verify_candidate = original
    assert sorted(calls) == sorted({GOOD, WRONG_SUB})       # two distinct contents, two runs
    assert grouped[0].document.content == GOOD and grouped[0].versions == list(VERSIONS)


def test_verification_off_or_impossible_leaves_the_ranking_exactly_as_it_was():
    index = lineage([WRONG_SUB, GOOD, WRONG_MUL])
    plain = index.retrieve_across_versions(QUERY)
    off = index.retrieve_across_versions(QUERY, execution=config(enabled=False))
    no_examples = index.retrieve_across_versions("Add two numbers.", execution=config())
    baseline = index.retrieve_across_versions("Add two numbers.")

    def view(hits):
        return [(h.content_hash, h.score, tuple(h.versions), h.variants) for h in hits]

    assert view(off) == view(plain)
    assert view(no_examples) == view(baseline)
    assert all("execution" not in hit.sources for hit in no_examples)


def test_a_lineage_with_no_correct_variant_is_not_promoted_by_recency():
    grouped = lineage([WRONG_SUB, WRONG_MUL, WRONG_SUB]).retrieve_across_versions(QUERY, execution=config())
    assert "execution" not in grouped[0].sources
    assert grouped[0].versions == list(VERSIONS)
