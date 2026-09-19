"""Shared pytest configuration.

Most of the suite is pure logic and runs in under a second.  A handful of
tests load the real encoder weights, which means a multi-gigabyte download on
a cold machine.  Those carry the ``model`` marker so they can be deselected::

    pytest -m "not model"      # fast: no weights, no download
    pytest                     # everything, including the encoder integration

They are *not* skipped by default: a green run should mean the real encoder
was exercised, not that it was quietly stepped over.

A second marker, ``data``, covers the tests that drive the evaluation harness
over the real CoIR AppsRetrieval split.  Those skip only when the split is
genuinely unreachable -- no ``artifacts/apps`` cache and no ``datasets``
install -- and they say which of the two to fix.  The harness raises in that
situation rather than returning placeholder rows, which is the whole point:
metrics computed on invented data are worse than no metrics at all.
"""

from __future__ import annotations

import functools

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "model: requires the real encoder weights (multi-GB download on a cold cache)",
    )
    config.addinivalue_line(
        "markers",
        "data: requires the real CoIR AppsRetrieval split, from the local "
        "artifacts/apps cache or a HuggingFace download",
    )


@functools.cache
def _apps_data_source() -> str | None:
    """Which source can supply the real APPS split, or ``None`` if neither can.

    Checked without loading anything: the point is to skip with a useful
    reason on a fresh clone, not to pay for the read twice.
    """
    from app.evaluation_harness import APPS_CACHE_DIR

    required = (
        "texts_corpus.jsonl",
        "texts_queries_train.jsonl",
        "texts_queries_test.jsonl",
        "qrels_train.json",
        "qrels_test.json",
    )
    if all((APPS_CACHE_DIR / name).exists() for name in required):
        return "local artifacts/apps cache"

    try:
        import datasets  # noqa: F401
    except ImportError:
        return None
    return "HuggingFace"


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip ``data`` tests loudly when the dataset genuinely is not there.

    The harness itself refuses to invent placeholder rows, so without the
    dataset these tests cannot run at all.  Skipping with the reason printed
    is honest; quietly passing them against synthetic data would not be.
    """
    if item.get_closest_marker("data") and _apps_data_source() is None:
        pytest.skip(
            "needs the CoIR AppsRetrieval split: populate artifacts/apps with "
            "`python scripts/encode_apps.py --tier accuracy --tag f2llm17b`, "
            "or `pip install -r requirements-eval.txt` to allow the download"
        )


@pytest.fixture
def stub_encoder():
    """A deterministic stand-in for :class:`~app.retrieval.encode.CodeEncoder`.

    Hashes the encoder input to a unit vector.  Identical text always yields an
    identical vector, which is the only property the indexing and caching
    layers actually depend on.
    """
    import hashlib

    import numpy as np

    class StubEncoder:
        model_name = "stub/deterministic"
        tier = "interactive"
        device = "cpu"
        similarity_metric = "cosine"
        normalize_embeddings = True
        query_prefix = "Instruct: Retrieve the most relevant code snippet for the given query.\nQuery: "
        document_prefix = ""
        dimension = 64

        def get_prompt_text(self, text: str, mode: str = "document") -> str:
            prefix = self.query_prefix if mode == "query" else self.document_prefix
            return f"{prefix}{text}"

        def encode(self, texts, mode: str = "document", **kwargs):
            single = isinstance(texts, str)
            items = [texts] if single else list(texts)
            out = np.empty((len(items), self.dimension), dtype=np.float32)
            for row, text in enumerate(items):
                digest = hashlib.sha256(
                    self.get_prompt_text(text, mode=mode).encode("utf-8")
                ).digest()
                raw = np.frombuffer(digest * 8, dtype=np.uint8)[: self.dimension]
                vector = raw.astype(np.float32) - 127.5
                out[row] = vector / (np.linalg.norm(vector) or 1.0)
            return out[0] if single else out

        def similarity(self, a, b):
            a = np.atleast_2d(a)
            b = np.atleast_2d(b)
            return a @ b.T

    return StubEncoder()
