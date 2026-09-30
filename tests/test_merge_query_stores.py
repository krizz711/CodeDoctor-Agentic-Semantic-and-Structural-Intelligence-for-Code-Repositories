import json

import numpy as np
import pytest

from app.indexing.cache import EmbeddingCache
from scripts.merge_query_stores import main, merge, sha256


def unit(seed, dim=8):
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def store(path, rows):
    cache = EmbeddingCache(path)
    cache.put_many(rows)
    cache.conn.close()
    return path


def test_merge_is_the_union_with_checksums_and_no_disagreement_for_identical_duplicates(tmp_path):
    a = store(tmp_path / "a.sqlite", [("k1", unit(1)), ("k2", unit(2))])
    b = store(tmp_path / "b.sqlite", [("k2", unit(2)), ("k3", unit(3))])
    manifest = merge([a, b], tmp_path / "out.sqlite")
    assert manifest["rows"] == 3 and manifest["duplicate_keys_disagreeing"] == 0
    assert [i["added"] for i in manifest["inputs"]] == [2, 1]
    assert manifest["inputs"][0]["sha256"] == sha256(a) and manifest["sha256"] == sha256(tmp_path / "out.sqlite")
    merged = EmbeddingCache(tmp_path / "out.sqlite")
    assert all(np.array_equal(merged.get(k), v) for k, v in (("k1", unit(1)), ("k2", unit(2)), ("k3", unit(3))))


def test_a_duplicate_key_with_a_different_vector_is_reported_and_the_first_is_kept(tmp_path):
    a = store(tmp_path / "a.sqlite", [("k", unit(1))])
    b = store(tmp_path / "b.sqlite", [("k", unit(99))])
    manifest = merge([a, b], tmp_path / "out.sqlite")
    assert manifest["rows"] == 1 and manifest["duplicate_keys_disagreeing"] == 1
    assert manifest["worst_duplicate_cosine"] < 0.9999
    assert np.array_equal(EmbeddingCache(tmp_path / "out.sqlite").get("k"), unit(1))


def test_missing_and_empty_inputs_fail_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        merge([tmp_path / "absent.sqlite"], tmp_path / "out.sqlite")
    with pytest.raises(ValueError):
        merge([], tmp_path / "out.sqlite")


def test_the_cli_writes_a_manifest_and_enforces_the_expected_row_count(tmp_path, capsys):
    a = store(tmp_path / "a.sqlite", [("k1", unit(1)), ("k2", unit(2))])
    out = tmp_path / "queries.sqlite"
    assert main(["--out", str(out), "--expect", "2", str(a)]) == 0
    manifest = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == 2
    with pytest.raises(SystemExit, match="expected 5"):
        main(["--out", str(tmp_path / "again.sqlite"), "--expect", "5", str(a)])
