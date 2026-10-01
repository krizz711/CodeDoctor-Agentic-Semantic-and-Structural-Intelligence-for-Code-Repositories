"""A demo artifact must match the active tier, recipe and corpus row order."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from app.indexing.cache import recipe_cache_key
from app.retrieval.encode import CodeEncoder
from app.workbench import TARGETS, Workbench


def artifact(tmp_path, monkeypatch, tier="accuracy"):
    monkeypatch.setenv("CODEDOCTOR_TIER", tier)
    monkeypatch.setattr("app.workbench.APPS_ARTIFACTS", tmp_path)
    tag = "f2llm17b" if tier == "accuracy" else "f2llm06b"
    encoder = CodeEncoder(tier=tier, device="cpu")  # reads YAML, never loads weights
    dim = 4
    ids = ["c0", "c1"]
    (tmp_path / "ids_corpus.json").write_text(json.dumps(ids), encoding="utf-8")
    (tmp_path / "texts_corpus.jsonl").write_text(
        "".join(json.dumps({"id": key, "text": f"print({i})"}) + "\n" for i, key in enumerate(ids)),
        encoding="utf-8")
    (tmp_path / "texts_queries_train.jsonl").write_text(
        json.dumps({"id": "q0", "text": "print zero"}) + "\n", encoding="utf-8")
    (tmp_path / "qrels_train.json").write_text(json.dumps({"q0": {"c0": 1}}), encoding="utf-8")
    np.save(tmp_path / f"emb_{tag}_corpus.npy", np.eye(2, dim, dtype=np.float32))
    record = {"model_name": encoder.model_name, "revision": encoder.revision, "tier": tier, "dim": dim,
              "document_recipe_key": recipe_cache_key(encoder, "", mode="document"),
              "query_recipe_key": recipe_cache_key(encoder, "", mode="query"),
              "corpus": {"rows": len(ids)}}
    for name, filename in (("array_sha256", f"emb_{tag}_corpus.npy"),
                           ("ids_sha256", "ids_corpus.json"), ("texts_sha256", "texts_corpus.jsonl")):
        record["corpus"][name] = hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
    manifest = {"models": {tag: record}}
    save = lambda: (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    save()
    bench = Workbench()
    encoder._model = SimpleNamespace(get_sentence_embedding_dimension=lambda: dim)
    bench._encoder = encoder
    monkeypatch.setattr(bench, "_warm_encoder", lambda: None)
    return bench, manifest, tag, save


@pytest.mark.parametrize("tier", ["accuracy", "interactive"])
def test_loader_uses_the_requested_tier_with_matching_provenance(tmp_path, monkeypatch, tier):
    bench, _, _, _ = artifact(tmp_path, monkeypatch, tier)
    assert bench._availability(TARGETS["apps"])[0]
    built = bench._build_apps(TARGETS["apps"])
    assert built.index._encoder.tier == tier
    assert built.index._embeddings["train"].shape == (2, 4)


def test_accuracy_does_not_fall_back_to_interactive_arrays(tmp_path, monkeypatch):
    bench, _, _, _ = artifact(tmp_path, monkeypatch, "interactive")
    monkeypatch.setenv("CODEDOCTOR_TIER", "accuracy")
    ready, reason = bench._availability(TARGETS["apps"])
    assert not ready and "f2llm17b" in reason


@pytest.mark.parametrize("field", ["model_name", "revision", "tier", "document_recipe_key", "query_recipe_key"])
def test_wrong_recipe_or_model_is_rejected_before_loading_weights(tmp_path, monkeypatch, field):
    bench, manifest, tag, save = artifact(tmp_path, monkeypatch)
    manifest["models"][tag][field] = "wrong"
    save()
    with pytest.raises(RuntimeError, match="does not match"):
        bench._build_apps(TARGETS["apps"])
    assert not bench._availability(TARGETS["apps"])[0]


def test_legacy_manifest_cannot_certify_array_provenance(tmp_path, monkeypatch):
    bench, manifest, tag, save = artifact(tmp_path, monkeypatch)
    del manifest["models"][tag]["corpus"]
    save()
    with pytest.raises(RuntimeError, match="legacy arrays"):
        bench._build_apps(TARGETS["apps"])


def test_reordered_ids_fail_the_saved_alignment_checksum(tmp_path, monkeypatch):
    bench, _, _, _ = artifact(tmp_path, monkeypatch)
    (tmp_path / "ids_corpus.json").write_text('["c1", "c0"]', encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum"):
        bench._build_apps(TARGETS["apps"])


@pytest.mark.parametrize("vectors", [np.ones((2, 5)), np.array([[np.nan, 0, 0, 0], [1, 0, 0, 0]]), np.zeros((2, 4))])
def test_wrong_dimensions_or_invalid_vectors_fail_even_with_fresh_checksum(tmp_path, monkeypatch, vectors):
    bench, manifest, tag, save = artifact(tmp_path, monkeypatch)
    path = tmp_path / f"emb_{tag}_corpus.npy"
    np.save(path, vectors)
    manifest["models"][tag]["corpus"]["array_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    save()
    with pytest.raises(RuntimeError, match="embedding shape|finite nonzero"):
        bench._build_apps(TARGETS["apps"])


def test_manifest_dimension_must_equal_the_loaded_encoder(tmp_path, monkeypatch):
    bench, _, _, _ = artifact(tmp_path, monkeypatch)
    bench._encoder._model = SimpleNamespace(get_sentence_embedding_dimension=lambda: 8)
    with pytest.raises(RuntimeError, match="loaded encoder"):
        bench._build_apps(TARGETS["apps"])


def test_duplicate_text_ids_cannot_silently_replace_corpus_rows(tmp_path, monkeypatch):
    bench, manifest, tag, save = artifact(tmp_path, monkeypatch)
    path = tmp_path / "texts_corpus.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + json.dumps({"id": "c0", "text": "different"}) + "\n",
                    encoding="utf-8")
    manifest["models"][tag]["corpus"]["texts_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    save()
    with pytest.raises(RuntimeError, match="duplicate id"):
        bench._build_apps(TARGETS["apps"])
