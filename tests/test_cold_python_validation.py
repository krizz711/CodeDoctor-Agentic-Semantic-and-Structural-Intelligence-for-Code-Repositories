"""Check the functional validation's cold/edit/version assertions without weights."""
import hashlib

import numpy as np
import pytest

from scripts.validate_cold_python import ObservedEncoder, functional_check


class Encoder:
    model_name, revision, dimension = "synthetic-test", "fixture", 8
    document_prefix, query_prefix = "", "Q: "
    max_seq_length, add_eos_token, normalize_embeddings = 8192, True, True
    device, torch_dtype = "cpu", "float32"

    def encode(self, texts, mode="document", **kwargs):
        single = isinstance(texts, str)
        items = [texts] if single else texts
        values = []
        for text in items:
            # Positive vectors keep every fixture eligible for dense search.
            vector = np.frombuffer(hashlib.sha256(text.encode()).digest()[:8], dtype=np.uint8).astype(np.float32) + 1
            values.append(vector / np.linalg.norm(vector))
        matrix = np.vstack(values)
        return matrix[0] if single else matrix

    def similarity(self, a, b):
        return np.atleast_2d(a) @ np.atleast_2d(b).T


def test_check_indexes_fresh_content_encodes_queries_and_retains_edit_versions(tmp_path):
    report = functional_check(tmp_path, Encoder())
    assert report["initial_cache_entries"] == 0
    assert report["builds"][0]["encoded"] > report["builds"][1]["encoded"] > 0
    assert report["builds"][1]["reused"] > 0
    assert report["unknown_version_rejected"]
    assert all(row["version_isolation"] for row in report["queries"])
    assert any(hit["variants"] == 2 for hit in report["cross_version"]["hits"])
    assert any(row["mode"] == "query" for row in report["encoder_calls"])


def test_observer_refuses_nonfinite_real_encoder_outputs():
    class Invalid:
        def encode(self, *args, **kwargs):
            return np.array([np.nan, 0], dtype=np.float32)

    with pytest.raises(RuntimeError, match="invalid vectors"):
        ObservedEncoder(Invalid()).encode("new query", mode="query")
