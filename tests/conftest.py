"""Shared pytest configuration.

Most of the suite is pure logic and runs in under a second.  A handful of
tests load the real encoder weights, which means a multi-gigabyte download on
a cold machine.  Those carry the ``model`` marker so they can be deselected::

    pytest -m "not model"      # fast: no weights, no download
    pytest                     # everything, including the encoder integration

They are *not* skipped by default: a green run should mean the real encoder
was exercised, not that it was quietly stepped over.
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "model: requires the real encoder weights (multi-GB download on a cold cache)",
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
