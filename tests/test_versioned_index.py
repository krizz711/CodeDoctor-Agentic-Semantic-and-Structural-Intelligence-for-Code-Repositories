import pytest

from app.indexing.versioned import VersionedIndex
from app.retrieval.hybrid import Document


def test_version_filter_prevents_cross_commit_leakage(tmp_path):
    index = VersionedIndex()
    index.add_snapshot("commit-a", [Document("a", "auth.js", "function oldAuth() {}", "oldAuth")])
    index.add_snapshot("commit-b", [Document("b", "auth.js", "function newAuth() {}", "newAuth")])

    results = index.retrieve("authentication", "commit-a")
    assert all(result.document.document_id == "a" for result in results)
    assert index.verify_no_cross_commit_leakage(results, "commit-a")

    with pytest.raises(KeyError, match="not indexed"):
        index.retrieve("authentication", "missing")

    path = tmp_path / "versions.json"
    index.save(path)
    restored = VersionedIndex.load(path)
    assert restored.commits() == ["commit-a", "commit-b"]
    assert restored.retrieve("newAuth", "commit-b")[0].document.document_id == "b"
