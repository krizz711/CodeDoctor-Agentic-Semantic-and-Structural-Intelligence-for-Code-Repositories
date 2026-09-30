"""The final P1/Bonus validation script, exercised without a model.

A deterministic hashing bag-of-words encoder stands in for F2LLM so dense and lexical
evidence agree and the fixture logic (cache accounting, fresh-rebuild equivalence, the
version lineages, the shipping ``Workbench.across`` path) runs in seconds. The real model
run is the recorded stage in scripts/submission_run.py.
"""

import hashlib
import re

import numpy as np

from scripts import validate_hands_on as vh


class HashingEncoder:
    """Feature-hashed token counts, L2-normalised: similar texts get similar vectors."""

    model_name, revision, dimension = "hashing-stub", "r1", 64
    document_prefix, query_prefix = "", "Q: "
    max_seq_length, add_eos_token, normalize_embeddings = 8192, True, True
    device, torch_dtype, max_batch = "cpu", "float32", 1

    def get_prompt_text(self, text, mode="document"):
        return (self.query_prefix if mode == "query" else self.document_prefix) + text

    def _vector(self, text):
        v = np.zeros(self.dimension, dtype=np.float32)
        for token in re.findall(r"[A-Za-z_]+|\d+", text.lower()):
            v[int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % self.dimension] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def encode(self, texts, mode="document", **_):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        out = np.vstack([self._vector(t) for t in items])
        return out[0] if single else out

    def similarity(self, a, b):
        return np.dot(np.atleast_2d(a), np.atleast_2d(b).T)


def test_p1_counts_every_encode_exactly_and_matches_a_fresh_rebuild(tmp_path):
    result = vh.p1(HashingEncoder(), tmp_path, repetitions=1)
    summary = result["summary"]
    assert summary["all_encoded_counts_exact"] and summary["all_equivalent_to_fresh_rebuild"]
    by_op = {u["operation"]: u for u in result["runs"][0]["updates"]}
    assert by_op["unchanged"]["encoded"] == 0 and by_op["rename"]["encoded"] == 0 and by_op["delete"]["encoded"] == 0
    assert by_op["one"]["encoded"] >= 1 and by_op["ten"]["encoded"] >= 10 and by_op["add"]["encoded"] >= 1
    assert by_op["one"]["encoded"] < result["runs"][0]["full"]["encoded"]        # incremental, not a rebuild
    assert result["runs"][0]["edited_function_retrievable"] is True
    assert result["runs"][0]["equivalence"]["min_vector_cosine"] >= 0.99999
    assert all(u["rss_bytes"] > 0 and u["seconds"] > 0 for u in result["runs"][0]["updates"])


def test_bonus_runs_through_the_workbench_and_verification_lifts_the_correct_version(tmp_path):
    result = vh.bonus(HashingEncoder(), tmp_path)
    assert result["path"].startswith("Workbench.across") and len(result["queries"]) == vh.LINEAGES
    metrics = result["metrics"]
    assert metrics["correct_version_first"] == 1.0 and metrics["lineage_first"] == 1.0
    assert metrics["wrong_above_correct"] == 0.0 and metrics["distinct_at_10"] == 10.0   # the maximum: no crowding
    changing = [q for q in result["queries"] if q["kind"] == "changes"]
    assert all(q["verified_by_execution"] for q in changing)
    unchanged = [q for q in result["queries"] if q["kind"] == "unchanged"]
    assert unchanged and all(q["top_versions"] == ["v1", "v2", "v3"] for q in unchanged)


def test_the_fixture_has_repeated_and_distinct_wrong_variants_and_alternating_correct_versions():
    codes = {lineage: [vh.lineage_code(lineage, v) for v in range(3)] for lineage in range(vh.LINEAGES)}
    assert {next(v for v in range(3) if codes[i][v][1]) for i in range(9)} == {0, 1, 2}
    repeated = [i for i in range(9) if len({c for c, ok in codes[i] if not ok}) == 1]
    distinct = [i for i in range(9) if len({c for c, ok in codes[i] if not ok}) == 2]
    assert repeated and distinct
    assert all(len({c for c, _ in codes[i]}) == 1 for i in range(9, 12))          # never change
