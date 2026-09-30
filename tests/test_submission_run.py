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
    queries = run.build_command("queries", args(), out, cache)
    assert queries[queries.index("--query-shard") + 1] == "1/4" and "--predictions" not in queries
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


def test_the_queries_stage_needs_a_shard():
    with pytest.raises(SystemExit):
        run.main(["--stage", "queries", "--out", "unused"])
