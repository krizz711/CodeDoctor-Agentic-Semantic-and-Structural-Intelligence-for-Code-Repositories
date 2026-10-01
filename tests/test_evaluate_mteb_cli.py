"""scripts/evaluate_mteb.py: shard -> merge -> final run, and the incomplete-cache abort.

MTEB itself is faked (it would open the test split); everything else is the real
entry point: argument handling, the pipeline, the stores, the audit records. The fake
``mteb.evaluate`` drives the real MTEB-facing wrapper exactly as MTEB does: encode the
queries, encode the corpus, then ask for the similarity matrix.
"""

import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.evaluate_mteb as ev
from app.indexing.cache import EmbeddingCache, recipe_cache_key
from app.indexing.cache_validation import APPS_DATASET
from app.pipeline.encoder import CorpusCacheIncomplete, QueryStoreIncomplete, mteb_document_text

QUERIES = [f"Return the number {i}.\n\nInput\n{i}\n\nOutput\n{i}\n" for i in range(11)]
DOCS = [{"id": f"d{i}", "text": f"  print({i})\n"} for i in range(7)]


class Stub:
    """Stands in for CodeEncoder: deterministic vectors, no model."""

    model_name, revision, dimension = "stub", "r1", 16
    document_prefix, query_prefix = "", "Q: "
    max_seq_length, add_eos_token, normalize_embeddings = 8192, True, True
    model = object()

    def __init__(self):
        self.encoded = []

    def get_prompt_text(self, text, mode="document"):
        return (self.query_prefix if mode == "query" else self.document_prefix) + text

    def encode(self, texts, mode="document", **_):
        items = [texts] if isinstance(texts, str) else list(texts)
        rows = []
        for text in items:
            self.encoded.append((mode, text))
            seed = int.from_bytes(hashlib.sha256(f"{mode}|{text}".encode()).digest()[:8], "little")
            v = np.random.default_rng(seed).standard_normal(self.dimension)
            rows.append((v / np.linalg.norm(v)).astype(np.float32))
        out = np.vstack(rows)
        return out[0] if isinstance(texts, str) else out

    def similarity(self, a, b):
        return np.dot(np.atleast_2d(a), np.atleast_2d(b).T)


@pytest.fixture
def harness(monkeypatch, tmp_path):
    import mteb

    from app import evaluation_harness

    stub = Stub()
    audit: list[dict] = []
    monkeypatch.setattr(ev, "CodeEncoder", lambda **kw: stub)
    task = SimpleNamespace(metadata=SimpleNamespace(dataset=dict(APPS_DATASET)), dataset={"default": {"test": {
        "queries": {"id": [f"q{i}" for i in range(len(QUERIES))]},
        "corpus": {"id": [row["id"] for row in DOCS]},
        "relevant_docs": {f"q{i}": {f"d{i % len(DOCS)}": 1} for i in range(len(QUERIES))},
    }}})
    task.data_loaded = False
    task.unload_after_evaluation = False

    def load_data():
        assert audit and audit[-1]["reason"] == "final_run_started"
        task.data_loaded = True

    task.load_data = load_data
    monkeypatch.setattr(mteb, "get_task", lambda name: task)
    monkeypatch.setattr(evaluation_harness, "record_persistent_audit",
                        lambda **kw: audit.append(kw) or kw)

    def loader(texts):
        return [{"text": texts[i:i + 3]} for i in range(0, len(texts), 3)]

    def fake_evaluate(model, tasks, **kwargs):
        assert kwargs["cache"] is None                                # MTEB's result cache stays off
        queries = model.encode(loader(QUERIES), prompt_type="query")
        docs = model.encode(loader([mteb_document_text(d) for d in DOCS]), prompt_type="document")
        model.similarity(queries, docs)
        payload = {"task_name": "AppsRetrieval", "dataset_revision": APPS_DATASET["revision"], "mteb_version": "2.21.0",
                   "scores": {"test": [{"ndcg_at_10": 0.5, "mrr_at_10": 0.25}]}}
        if task.unload_after_evaluation:
            for labels in task.dataset["default"]["test"]["relevant_docs"].values():
                labels.clear()
            task.dataset = None
            task.data_loaded = False
        return SimpleNamespace(task_results=[SimpleNamespace(to_dict=lambda: payload)])

    monkeypatch.setattr(mteb, "evaluate", fake_evaluate)

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["evaluate_mteb.py", "--no-execution", *map(str, args)])
        ev.main()

    def corpus_cache(rows=DOCS):
        cache = EmbeddingCache(tmp_path / "corpus.sqlite")
        cache.put_many([(recipe_cache_key(stub, mteb_document_text(r)), stub.encode(mteb_document_text(r)))
                        for r in rows])
        stub.encoded.clear()
        return tmp_path / "corpus.sqlite"

    return SimpleNamespace(stub=stub, audit=audit, run=run, corpus_cache=corpus_cache, tmp=tmp_path, task=task)


def test_three_shards_then_a_verified_final_run(harness):
    cache = harness.corpus_cache()
    reports = []
    for index in range(3):
        out = harness.tmp / f"shard{index}.json"
        harness.run("--corpus-cache", cache, "--require-complete-corpus-cache",
                    "--query-store", harness.tmp / f"s{index}.sqlite", "--query-shard", f"{index}/3",
                    "--output", out)
        reports.append(json.loads(out.read_text(encoding="utf-8")))
    assert sum(r["encoded"] for r in reports) == len(QUERIES)
    assert [r["shard"] for r in reports] == [[0, 3], [1, 3], [2, 3]]

    merged = EmbeddingCache(harness.tmp / "merged.sqlite")
    for index in range(3):
        shard = EmbeddingCache(harness.tmp / f"s{index}.sqlite")
        keys = [k for (k,) in shard.conn.execute("SELECT cache_key FROM embedding_cache")]
        merged.put_many([(k, shard.get(k)) for k in keys])
    assert merged.count() == len(QUERIES)

    harness.stub.encoded.clear()
    result = harness.tmp / "result.json"
    harness.run("--corpus-cache", cache, "--require-complete-corpus-cache", "--query-store",
                harness.tmp / "merged.sqlite", "--require-complete-query-store", "--query-verify", 4, "--output", result)
    assert json.loads(result.read_text(encoding="utf-8"))["scores"]["test"][0]["ndcg_at_10"] == 0.5
    # nothing but the 4 verification samples was encoded in the final run
    assert [mode for mode, _ in harness.stub.encoded] == ["query"] * 4

    reasons = [entry["reason"] for entry in harness.audit]
    assert reasons.count("query_shard_started") == 3 and reasons.count("final_run_started") == 1
    final = [entry for entry in harness.audit if entry["reason"] == "final_run"][0]["extra"]
    assert final["query_store"] == {"hits": len(QUERIES), "encoded": 0}
    assert final["corpus_cache"] == {"hits": len(DOCS), "misses": 0}
    assert final["query_verification"]["sampled"] == 4 and final["query_verification"]["min_cosine"] > 0.9999


def test_an_incomplete_corpus_cache_aborts_before_any_encoding(harness):
    cache = harness.corpus_cache(DOCS[:-2])
    with pytest.raises(CorpusCacheIncomplete, match="2 of 7"):
        harness.run("--corpus-cache", cache, "--require-complete-corpus-cache",
                    "--query-store", harness.tmp / "q.sqlite", "--output", harness.tmp / "r.json")
    assert [m for m, _ in harness.stub.encoded if m == "document"] == []


def test_a_shard_needs_a_store_and_a_valid_index(harness):
    for bad in (["--query-shard", "0/3"], ["--query-shard", "3/3", "--query-store", "x.sqlite"],
                ["--query-shard", "a/b", "--query-store", "x.sqlite"]):
        with pytest.raises(SystemExit):
            harness.run(*bad)


def test_an_existing_output_is_never_overwritten_by_default(harness):
    cache = harness.corpus_cache()
    existing = harness.tmp / "appsretrieval_results.json"
    existing.write_text('{"first": "release"}', encoding="utf-8")
    with pytest.raises(SystemExit, match="already exists"):
        harness.run("--corpus-cache", cache, "--output", existing)
    assert existing.read_text(encoding="utf-8") == '{"first": "release"}'
    assert harness.audit == []                          # refused before any test-split record
    harness.run("--corpus-cache", cache, "--output", existing, "--overwrite")
    assert json.loads(existing.read_text(encoding="utf-8"))["scores"]["test"][0]["ndcg_at_10"] == 0.5


def test_the_default_output_is_under_results_and_an_execution_audit_sits_beside_it(harness, monkeypatch):
    cache = harness.corpus_cache()
    monkeypatch.chdir(harness.tmp)
    harness.run("--corpus-cache", cache)
    out = harness.tmp / "results"
    assert (out / "appsretrieval_results.json").exists()
    audit = json.loads((out / "execution_audit.json").read_text(encoding="utf-8"))
    assert audit["configuration"]["configuration"] == "config/submission.yaml"
    assert audit["configuration"]["model"].endswith("F2LLM-v2-1.7B")
    assert audit["corpus_cache"] == {"hits": len(DOCS), "misses": 0}


def test_strict_replay_refuses_a_missing_query_without_live_fallback(harness):
    cache = harness.corpus_cache()
    store = EmbeddingCache(harness.tmp / "incomplete.sqlite")
    store.put_many([(recipe_cache_key(harness.stub, text, "query"), harness.stub.encode(text, mode="query"))
                    for text in QUERIES[:-1]])
    harness.stub.encoded.clear()
    with pytest.raises(QueryStoreIncomplete, match="1 missing"):
        harness.run("--corpus-cache", cache, "--query-store", harness.tmp / "incomplete.sqlite",
                    "--require-complete-query-store", "--output", harness.tmp / "incomplete.json")
    assert harness.stub.encoded == []
    assert not (harness.tmp / "incomplete.json").exists()


def test_an_unexpected_task_revision_aborts_before_encoding(harness):
    cache = harness.corpus_cache()
    harness.task.metadata.dataset["revision"] = "wrong"
    with pytest.raises(RuntimeError, match="dataset revision"):
        harness.run("--corpus-cache", cache, "--output", harness.tmp / "wrong.json")
    assert harness.stub.encoded == []


def test_strict_replay_cannot_be_used_for_a_shard_or_without_a_store(harness):
    for arguments in (["--require-complete-query-store"],
                      ["--require-complete-query-store", "--query-store", "q.sqlite", "--query-shard", "0/3"]):
        with pytest.raises(SystemExit):
            harness.run(*arguments)
    assert harness.audit == []


def test_prediction_identity_survives_evaluator_unloading_the_dataset(harness, monkeypatch):
    from app.indexing import cache_validation
    from scripts import export_submission_csv

    monkeypatch.setattr(ev, "APPS_TEST_QUERIES", len(QUERIES))
    monkeypatch.setattr(cache_validation, "APPS_DOCUMENTS", len(DOCS))
    cache = harness.corpus_cache()
    store_path = harness.tmp / "complete.sqlite"
    store = EmbeddingCache(store_path)
    store.put_many([(recipe_cache_key(harness.stub, text, "query"), harness.stub.encode(text, mode="query"))
                    for text in QUERIES])
    harness.stub.encoded.clear()
    harness.task.unload_after_evaluation = True
    exports = []

    def capture_export(folder, payload, out, **validation):
        assert harness.task.dataset is None
        assert validation["expected_query_ids"] == {f"q{i}" for i in range(len(QUERIES))}
        assert validation["valid_corpus_ids"] == {row["id"] for row in DOCS}
        assert validation["qrels"]["q0"] == {"d0": 1}
        exports.append(validation)
        return []

    monkeypatch.setattr(export_submission_csv, "export_from_predictions", capture_export)
    harness.run("--corpus-cache", cache, "--query-store", store_path, "--require-complete-query-store",
                "--predictions", harness.tmp / "predictions", "--output", harness.tmp / "result.json")
    assert len(exports) == 1 and harness.stub.encoded == []
    audit = json.loads((harness.tmp / "execution_audit.json").read_text())
    assert audit["prediction_identity_setup_seconds"] >= 0
