"""Cross-version snippet retrieval: content addressing, lineage, evidence ranking."""

from __future__ import annotations

from app.indexing.lineage import LineageStore, content_hash

GOOD = "a, b = map(int, input().split())\nprint(a + b)\n"
BUGGY = "a, b = map(int, input().split())\nprint(a - b)\n"
OTHER = "n = int(input())\nprint(n * n)\n"


def _store() -> LineageStore:
    store = LineageStore()
    store.add_version("v1", [("sum", BUGGY), ("square", OTHER)])
    store.add_version("v2", [("sum", GOOD), ("square", OTHER)])
    return store


def test_unchanged_content_is_stored_once():
    store = _store()
    hashes, texts = store.unique_texts()
    assert len(hashes) == 3  # BUGGY, OTHER (once), GOOD
    assert store.contents[content_hash(OTHER)].versions == ["v1", "v2"]


def test_a_new_version_reports_only_what_changed():
    store = _store()
    stats = store.add_version("v3", [("sum", GOOD), ("square", OTHER)])
    assert stats == {"new_contents": 0, "reused_contents": 2}


def test_versions_collapse_into_one_lineage_with_history():
    store = _store()
    dense = {content_hash(BUGGY): 0.90, content_hash(GOOD): 0.89, content_hash(OTHER): 0.5}
    hits = store.rank(dense)
    assert [h.lineage for h in hits] == ["sum", "square"]
    assert hits[0].variants == 2 and hits[0].all_versions == ("v1", "v2")


def test_dense_alone_picks_the_buggy_near_clone():
    store = _store()
    dense = {content_hash(BUGGY): 0.90, content_hash(GOOD): 0.89}
    assert store.rank(dense)[0].best_text == BUGGY


def test_execution_evidence_picks_the_correct_version():
    store = _store()
    dense = {content_hash(BUGGY): 0.90, content_hash(GOOD): 0.89}
    hit = store.rank(dense, tiers={content_hash(GOOD): "strong"})[0]
    assert hit.best_text == GOOD and hit.reason == "execution: strong"
    assert hit.best_versions == ("v2",)


def test_ties_go_to_the_most_recent_version():
    store = _store()
    dense = {content_hash(BUGGY): 0.5, content_hash(GOOD): 0.5}
    hit = store.rank(dense)[0]
    assert hit.best_text == GOOD and "recent" in hit.reason


def test_similarity_clustering_without_keys():
    store = LineageStore()
    store.add_version("v1", [(content_hash(BUGGY), BUGGY), (content_hash(OTHER), OTHER)])
    store.add_version("v2", [(content_hash(GOOD), GOOD)])
    order = [content_hash(BUGGY), content_hash(GOOD), content_hash(OTHER)]
    assignment = store.cluster_by_similarity(order, threshold=0.5)
    assert assignment[content_hash(GOOD)] == assignment[content_hash(BUGGY)]
    assert assignment[content_hash(OTHER)] != assignment[content_hash(BUGGY)]
    dense = {h: s for h, s in zip(order, (0.9, 0.89, 0.5))}
    hits = store.rank(dense, lineage=assignment.__getitem__)
    assert len(hits) == 2
