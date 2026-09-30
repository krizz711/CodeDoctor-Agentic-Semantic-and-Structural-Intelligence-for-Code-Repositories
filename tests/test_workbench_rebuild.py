"""E3 Part 2 (experiments/C3_cycle): an edited folder is re-indexed in-process.

Before cycle 3, ``start_build`` returned early for a built target, so the demo
could only pick up an edit by restarting the server.  With ``rebuild=True`` the
folder is re-read, the lexical index rebuilt, and only chunks whose encoder
input changed are encoded again.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from app.indexing.cache import EmbeddingCache
from app.workbench import Workbench, _documents_from_source, _encoder_input

ALPHA = "def alpha(x):\n    return x + 1\n"
ALPHA_EDITED = "def alpha(x):\n    return x * 1000 + 7\n"
BETA = "def beta(y):\n    return y * 2\n"


class RecordingEncoder:
    """Deterministic vectors; counts document and query forward passes separately."""

    model_name, revision, dimension = "stub", "r1", 32
    document_prefix, query_prefix = "", "Q: "
    max_seq_length, add_eos_token, normalize_embeddings = 8192, True, True
    device, torch_dtype = "cpu", "float32"

    def __init__(self) -> None:
        self.documents: list[str] = []

    def _vec(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode()).digest()
        v = np.random.default_rng(int.from_bytes(digest[:8], "little")).standard_normal(self.dimension)
        return (v / np.linalg.norm(v)).astype(np.float32)

    def encode(self, texts, mode="document", **_):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if mode == "document":
            self.documents += items
        out = np.vstack([self._vec(t) for t in items])
        return out[0] if single else out

    def similarity(self, a, b):
        a = a.reshape(1, -1) if a.ndim == 1 else a
        return a @ b.T

    def get_prompt_text(self, text, mode="document"):
        return (self.query_prefix if mode == "query" else self.document_prefix) + text


def wait_ready(bench: Workbench, seconds: float = 60) -> dict:
    deadline = time.time() + seconds
    while time.time() < deadline:
        state = bench.state()
        if state["phase"] in {"ready", "error"}:
            return state
        time.sleep(0.05)
    return bench.state()


def encoder_inputs(path: Path, rel: str) -> set[str]:
    return {_encoder_input(d) for d in _documents_from_source(path.read_text(encoding="utf-8"), rel)}


def test_rebuild_picks_up_an_edit_and_encodes_only_what_changed(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text(ALPHA, encoding="utf-8")
    (root / "b.py").write_text(BETA, encoding="utf-8")
    bench = Workbench(cache=EmbeddingCache(tmp_path / "cache.db"))
    encoder = RecordingEncoder()
    bench._encoder = encoder
    key = bench.open_folder(str(root))["key"]

    bench.start_build(key, dense=True)
    assert wait_ready(bench)["phase"] == "ready"
    before = encoder_inputs(root / "a.py", "a.py")
    first_pass = len(encoder.documents)
    assert first_pass > 0

    (root / "a.py").write_text(ALPHA_EDITED, encoding="utf-8")
    after = encoder_inputs(root / "a.py", "a.py")

    # Without the flag, a built target is left alone (the old behaviour).
    bench.start_build(key, dense=True)
    assert len(encoder.documents) == first_pass

    bench.start_build(key, dense=True, rebuild=True)
    state = wait_ready(bench)
    assert state["phase"] == "ready" and state["dense"]
    re_encoded = encoder.documents[first_pass:]
    assert set(re_encoded) == after - before, "exactly the chunks whose encoder input changed"

    codes = [r["code"] for r in bench.search(key, "x * 1000 + 7", verify=False)["results"]]
    assert any("x * 1000 + 7" in c for c in codes)
    assert not any("return x + 1\n" in c for c in codes), "the stale body is gone"


def test_rebuilding_after_a_pure_move_encodes_nothing(tmp_path):
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "a.py").write_text(ALPHA, encoding="utf-8")
    (root / "b.py").write_text(BETA, encoding="utf-8")
    bench = Workbench(cache=EmbeddingCache(tmp_path / "cache.db"))
    encoder = RecordingEncoder()
    bench._encoder = encoder
    key = bench.open_folder(str(root))["key"]
    bench.start_build(key, dense=True)
    assert wait_ready(bench)["phase"] == "ready"
    first_pass = len(encoder.documents)

    (root / "b.py").rename(root / "pkg" / "b.py")  # byte-identical content, new path
    bench.start_build(key, dense=True, rebuild=True)
    assert wait_ready(bench)["phase"] == "ready"
    assert len(encoder.documents) == first_pass
