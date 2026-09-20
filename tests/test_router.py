"""The router decides which passes run. These tests pin the decisions.

A planner is only worth having if its choices are inspectable and stable. Each
test below names the signal it is pinning, so a regression says which piece of
reasoning broke rather than just "routing changed".
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.agent import QueryRouter, RetrievalPlan


# --------------------------------------------------------------------------
# Minimal stand-ins. The router only reads `.score`, `.sources` and
# `.document.name`, so binding these tests to the real Document/RankedDocument
# would couple them to fields they do not exercise.
# --------------------------------------------------------------------------
@dataclass
class FakeDoc:
    name: str


@dataclass
class FakeHit:
    document: FakeDoc
    score: float
    sources: tuple[str, ...]


WORKED_EXAMPLES = "-----Examples-----\nInput\n21\n\nOutput\n42"


def test_free_passes_always_run():
    plan = QueryRouter().plan("anything at all")
    assert plan.ran("exact")
    assert plan.ran("bm25")


def test_prose_query_plans_dense_up_front():
    """No identifier to match on, so lexical has nothing to anchor to."""
    plan = QueryRouter().plan(
        "how does the system avoid leaking the test split", dense_available=True
    )
    assert plan.ran("dense")


def test_identifier_query_defers_dense():
    plan = QueryRouter().plan("verify_candidate", dense_available=True)
    assert not plan.ran("dense")
    assert any("deferred" in step.reason for step in plan.skipped if step.name == "dense")


def test_dense_skipped_entirely_when_unavailable():
    """A cold snapshot must still answer, not block on warming."""
    plan = QueryRouter().plan("how is the cache invalidated", dense_available=False)
    assert not plan.ran("dense")
    assert any("no warm embeddings" in s.reason for s in plan.skipped if s.name == "dense")


@pytest.mark.parametrize(
    "query",
    ["what calls verify_candidate", "who uses the encoder", "which module imports yaml"],
)
def test_structural_intent_adds_the_graph_pass(query: str):
    assert QueryRouter().plan(query).ran("structural")


def test_no_structural_intent_skips_the_graph_pass():
    plan = QueryRouter().plan("how is the embedding cache invalidated")
    assert not plan.ran("structural")


def test_execution_gated_on_worked_examples():
    """97.6% of AppsRetrieval queries pass this gate; 0% of CodeTransOceanDL."""
    assert QueryRouter().plan(WORKED_EXAMPLES).ran("execution")
    assert not QueryRouter().plan("where are options parsed").ran("execution")


def test_escalates_when_lexical_is_ambiguous():
    router = QueryRouter()
    plan = router.plan("batch size", dense_available=True)
    hits = [
        FakeHit(FakeDoc("plan_batches"), 0.0166, ("bm25",)),
        FakeHit(FakeDoc("select_batch"), 0.0165, ("bm25",)),
    ]
    decision = router.escalate(plan, hits)
    assert decision.escalated
    assert plan.ran("dense")


def test_stops_when_rank_one_is_the_named_symbol():
    """The strong stop signal: we already found the function by name."""
    router = QueryRouter()
    plan = router.plan("verify_candidate", dense_available=True)
    hits = [
        FakeHit(FakeDoc("verify_candidate"), 0.0166, ("exact", "bm25")),
        FakeHit(FakeDoc("run_candidate"), 0.0165, ("bm25",)),
    ]
    decision = router.escalate(plan, hits)
    assert not decision.escalated
    assert not plan.ran("dense")
    assert "exact name match" in decision.reason


def test_escalates_when_lexical_returns_nothing():
    router = QueryRouter()
    plan = router.plan("zzz_nonexistent_thing", dense_available=True)
    decision = router.escalate(plan, [])
    assert decision.escalated
    assert plan.ran("dense")


def test_cannot_escalate_without_dense():
    router = QueryRouter()
    plan = router.plan("something ambiguous here", dense_available=False)
    decision = router.escalate(plan, [])
    assert not decision.escalated
    assert not plan.ran("dense")


def test_always_dense_disables_routing():
    plan = QueryRouter(always_dense=True).plan("verify_candidate", dense_available=True)
    assert plan.ran("dense")


@pytest.mark.parametrize(
    "query",
    [
        "verify_candidate",
        "what calls verify_candidate",
        "how does version isolation work",
        WORKED_EXAMPLES,
        "",
    ],
)
def test_a_pass_is_never_both_run_and_skipped(query: str):
    """The invariant that made the first draft's output unreadable."""
    router = QueryRouter()
    plan = router.plan(query, dense_available=True)
    router.escalate(plan, [FakeHit(FakeDoc("x"), 0.01, ("bm25",))])
    ran = {step.name for step in plan.steps}
    skipped = {step.name for step in plan.skipped}
    assert not (ran & skipped), f"{ran & skipped} appear in both lists"


def test_plan_serialises_with_reasons():
    """The UI renders this; every entry must carry a why."""
    router = QueryRouter()
    plan = router.plan("how is the cache invalidated", dense_available=True)
    payload = plan.as_dict()
    assert payload["ran"] and payload["skipped"]
    for entry in payload["ran"] + payload["skipped"]:
        assert entry["reason"], entry
        assert entry["cost"] in {"free", "cheap", "expensive"}


def test_empty_query_does_not_crash():
    router = QueryRouter()
    plan = router.plan("", dense_available=True)
    assert isinstance(plan, RetrievalPlan)
    router.escalate(plan, [])
