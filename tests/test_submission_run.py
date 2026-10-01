import argparse
import json
import sys

import numpy as np
import pytest

from app.indexing.cache import EmbeddingCache
from app.submission import read_config
from scripts import submission_run as run

ENCODER = read_config(run.ROOT / "config" / "submission.yaml")["encoder"]


def make_cache(directory, **overrides):
    directory.mkdir()
    cache = EmbeddingCache(directory / "corpus.sqlite")
    cache.put_many([(f"k{i}", np.ones(4, dtype=np.float32)) for i in range(5)])
    cache.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    cache.conn.close()
    manifest = {"model": ENCODER["production_model"], "revision": ENCODER["revision"], "limit": None,
                "unique_vectors": 5, "sha256": run.sha256_file(directory / "corpus.sqlite")}
    manifest.update(overrides)
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def test_a_complete_pinned_cache_is_accepted(tmp_path):
    assert run.verify_document_cache(make_cache(tmp_path / "ok"))["unique_vectors"] == 5


@pytest.mark.parametrize("overrides, message", [
    ({"sha256": "0" * 64}, "checksum"),
    ({"revision": "other"}, "different model or revision"),
    ({"model": "someone/else"}, "different model or revision"),
    ({"limit": 100}, "smoke test"),
    ({"unique_vectors": 9}, "holds 5 vectors"),
])
def test_every_way_a_cache_can_be_wrong_is_rejected(tmp_path, overrides, message):
    with pytest.raises(RuntimeError, match=message):
        run.verify_document_cache(make_cache(tmp_path / "bad", **overrides))


def test_a_directory_without_its_manifest_is_rejected(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(RuntimeError, match="corpus.sqlite and manifest.json"):
        run.verify_document_cache(tmp_path / "empty")


def test_source_hashes_cover_the_runtime_and_the_tree_hash_is_order_independent_and_sensitive():
    hashes = run.source_hashes()
    for name in ("app/submission.py", "config/submission.yaml", "config/frozen.yaml",
                 "scripts/evaluate_mteb.py", "requirements-submission.txt"):
        assert len(hashes[name]) == 64
    assert not any("__pycache__" in name for name in hashes)
    shuffled = dict(reversed(list(hashes.items())))
    assert run.tree_hash(hashes) == run.tree_hash(shuffled)
    changed = {**hashes, "app/submission.py": "f" * 64}
    assert run.tree_hash(changed) != run.tree_hash(hashes)


def args(**kw):
    base = dict(shard="1/4", tier="accuracy", query_verify=30)
    return argparse.Namespace(**{**base, **kw})


def test_stage_commands_are_the_documented_judge_entrypoint(tmp_path):
    cache, out = tmp_path / "cache", tmp_path / "out"
    p0 = run.build_command("p0", args(), out, cache)
    assert p0[1:3] == ["scripts/evaluate_mteb.py", "--device"] and p0[3] == "cpu"
    for flag in ("--config", "config/submission.yaml", "--require-complete-corpus-cache", "--query-verify", "30",
                 "--predictions"):
        assert flag in p0
    assert "--query-shard" not in p0
    assert "--require-complete-query-store" in p0 and "--corpus-manifest" in p0
    queries = run.build_command("queries", args(), out, cache)
    assert queries[queries.index("--query-shard") + 1] == "1/4" and "--predictions" not in queries
    assert "--require-complete-query-store" not in queries
    assert run.build_command("documents", args(), out, None)[2:4] == ["--device", "cuda"]
    assert "--tier" in run.build_command("hands-on", args(tier="interactive"), out, None)


def test_a_child_that_succeeds_fails_or_hangs_is_recorded_faithfully(tmp_path):
    ok = run.run_logged([sys.executable, "-c", "print('hello from the child')"], tmp_path, 60)
    assert ok["exit_code"] == 0 and not ok["timed_out"]
    assert "hello from the child" in (tmp_path / "run.log").read_text(encoding="utf-8")

    (tmp_path / "b").mkdir()
    bad = run.run_logged([sys.executable, "-c", "import sys; sys.exit(3)"], tmp_path / "b", 60)
    assert bad["exit_code"] == 3 and not bad["timed_out"]

    (tmp_path / "c").mkdir()
    hung = run.run_logged([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path / "c", 2)
    assert hung["timed_out"] and hung["seconds"] < 30


def test_main_leaves_a_complete_environment_record(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "build_command", lambda *a, **k: [sys.executable, "-c", "print('stage ran')"])
    out = tmp_path / "hands-on"
    assert run.main(["--stage", "hands-on", "--out", str(out)]) == 0
    record = json.loads((out / "environment.json").read_text(encoding="utf-8"))
    assert record["status"] == "completed" and record["exit_code"] == 0
    assert record["configuration"]["sha256"] == run.sha256_file(run.ROOT / "config" / "submission.yaml")
    assert len(record["source_tree_sha256"]) == 64 and record["source_sha256"]["app/submission.py"]
    assert record["packages"]["numpy"] and record["physical_cpus"] and record["ram_bytes"] > 0
    assert (out / "requirements-measured.txt").read_text(encoding="utf-8").strip()
    assert record["command"][-1] == "print('stage ran')"
    with pytest.raises(FileExistsError):                    # a stage never overwrites an earlier record
        run.main(["--stage", "hands-on", "--out", str(out)])


def test_explicit_query_input_paths_round_trip_in_the_initial_environment_record(tmp_path):
    inputs = [tmp_path / "shard0", tmp_path / "shard1"]
    arguments = args(query_store_in=inputs, corpus_cache=tmp_path / "documents",
                     nested={"paths": (tmp_path / "extra", None), "enabled": True})
    environment = run.capture_environment("p0", arguments, tmp_path)
    run.write_environment(tmp_path, environment)
    recorded = json.loads((tmp_path / "environment.json").read_text())
    assert recorded["args"]["query_store_in"] == [str(p) for p in inputs]
    assert recorded["args"]["corpus_cache"] == str(tmp_path / "documents")
    assert recorded["args"]["nested"] == {"paths": [str(tmp_path / "extra"), None], "enabled": True}


def test_the_queries_stage_needs_a_shard():
    with pytest.raises(SystemExit):
        run.main(["--stage", "queries", "--out", "unused"])


def test_query_stage_copies_read_only_input_and_passes_a_database_file(tmp_path, monkeypatch):
    source = make_cache(tmp_path / 'mounted')
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    commands = []
    original_verify = run.verify_document_cache

    def synthetic_verify(directory, *, require_official=False):
        assert require_official
        return original_verify(directory)

    monkeypatch.setattr(run, "verify_document_cache", synthetic_verify)

    def fake_run(command, out, timeout_seconds):
        commands.append(command)
        database = run.Path(command[command.index('--corpus-cache') + 1])
        assert database.is_file() and database.name == 'corpus.sqlite'
        assert database.parent != source
        return {'exit_code': 0, 'seconds': 0, 'peak_process_tree_rss_bytes': 0, 'timed_out': False}

    monkeypatch.setattr(run, 'run_logged', fake_run)
    assert run.main(['--stage', 'queries', '--shard', '0/4', '--corpus-cache', str(source),
                     '--out', str(tmp_path / 'result')]) == 0
    assert commands
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


def test_official_cache_rejects_invented_dataset_identity_and_invalid_vectors(tmp_path):
    source = make_cache(tmp_path / "not-official", dataset={"path": "other", "revision": "other"},
                        documents=8765, texts_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="dataset identity"):
        run.verify_document_cache(source, require_official=True)


def query_metadata(directory, index, count=2, **environment_overrides):
    from app.indexing.cache_validation import APPS_DATASET

    directory.mkdir()
    environment = {"stage": "queries", "status": "completed", "exit_code": 0,
                   "command": ["python", "--device", "cpu"], "args": {"shard": f"{index}/{count}"},
                   "configuration": {"sha256": run.sha256_file(run.ROOT / "config/submission.yaml")},
                   "source_sha256": run.source_hashes(), "source_tree_sha256": "a" * 64,
                   "document_cache": {"dataset": APPS_DATASET, "sha256": "b" * 64, "texts_sha256": "c" * 64},
                   "packages": {"torch": "2.8.0+cpu", "transformers": "4.57.6", "sentence-transformers": "5.0.0",
                                "mteb": "2.21.0", "datasets": "3.6.0", "numpy": "2.5.3"}}
    environment.update(environment_overrides)
    (directory / "environment.json").write_text(json.dumps(environment), encoding="utf-8")
    (directory / "shard_report.json").write_text(json.dumps({"shard": [index, count], "store_vectors": 3}), encoding="utf-8")
    (directory / "queries.sqlite").touch()
    return directory / "queries.sqlite"


def test_query_provenance_requires_all_shards_and_matching_completed_runs(tmp_path):
    a = query_metadata(tmp_path / "a", 0)
    b = query_metadata(tmp_path / "b", 1)
    records, shards = run.verify_query_inputs([a, b])
    assert shards == [(0, 2), (1, 2)] and records[0]["reported_rows"] == 3
    with pytest.raises(RuntimeError, match="different document cache"):
        run.verify_query_inputs([a, b], document_manifest={"sha256": "different"})
    with pytest.raises(RuntimeError, match="every declared shard"):
        run.verify_query_inputs([a])
    with pytest.raises(RuntimeError, match="every declared shard"):
        run.verify_query_inputs([a, a])
    e = json.loads((b.parent / "environment.json").read_text())
    e["status"] = "failed"
    (b.parent / "environment.json").write_text(json.dumps(e))
    with pytest.raises(RuntimeError, match="not a completed"):
        run.verify_query_inputs([a, b])


@pytest.mark.parametrize("defect", ["config", "recipe", "dataset", "device", "packages"])
def test_query_provenance_rejects_a_mismatched_shard_before_merging(tmp_path, defect):
    a = query_metadata(tmp_path / "a", 0)
    b = query_metadata(tmp_path / "b", 1)
    path = b.parent / "environment.json"
    e = json.loads(path.read_text())
    if defect == "config":
        e["configuration"]["sha256"] = "wrong"
    elif defect == "recipe":
        e["source_sha256"]["app/retrieval/encode.py"] = "wrong"
    elif defect == "dataset":
        e["document_cache"]["dataset"]["revision"] = "wrong"
    elif defect == "device":
        e["command"][-1] = "cuda"
    else:
        e["packages"]["torch"] = "other"
    path.write_text(json.dumps(e))
    with pytest.raises(RuntimeError):
        run.verify_query_inputs([a, b])
