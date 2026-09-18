"""Memory-bounded dense search helpers.

The full ``[n_queries, n_documents]`` score matrix for APPS is 3765 x 8765
floats.  That fits in RAM, but the tuning loop runs it hundreds of times, so
everything here works in query blocks and keeps only the candidate pool.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CandidatePool:
    """Top-``depth`` documents per query, descending by dense score."""

    #: ``[n_queries, depth]`` document indices.
    indices: np.ndarray
    #: ``[n_queries, depth]`` dense scores aligned with ``indices``.
    scores: np.ndarray

    @property
    def depth(self) -> int:
        return int(self.indices.shape[1])

    @property
    def n_queries(self) -> int:
        return int(self.indices.shape[0])

    def gold_rank(self, query_row: int, gold_index: int) -> int | None:
        """1-based rank of ``gold_index`` in the pool, or ``None`` if absent."""
        hit = np.flatnonzero(self.indices[query_row] == gold_index)
        return int(hit[0]) + 1 if hit.size else None

    def gold_ranks(self, gold_indices: np.ndarray) -> np.ndarray:
        """1-based ranks, or ``depth + 1`` where the gold document missed the pool."""
        out = np.full(self.n_queries, self.depth + 1, dtype=np.int32)
        for row in range(self.n_queries):
            hit = np.flatnonzero(self.indices[row] == gold_indices[row])
            if hit.size:
                out[row] = int(hit[0]) + 1
        return out


def build_candidate_pool(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    *,
    depth: int = 200,
    block_size: int = 256,
) -> CandidatePool:
    """Cosine top-``depth`` search in query blocks.

    Both matrices must already be L2-normalized, so the inner product *is* the
    cosine similarity; this is asserted rather than assumed.
    """
    query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    document_embeddings = np.ascontiguousarray(document_embeddings, dtype=np.float32)

    for name, matrix in (("query", query_embeddings), ("document", document_embeddings)):
        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, atol=2e-2):
            raise ValueError(
                f"{name} embeddings are not L2-normalized "
                f"(norm range {norms.min():.4f}..{norms.max():.4f}); "
                "cosine scoring would be wrong"
            )

    n_queries = query_embeddings.shape[0]
    depth = min(depth, document_embeddings.shape[0])
    indices = np.empty((n_queries, depth), dtype=np.int32)
    scores = np.empty((n_queries, depth), dtype=np.float32)

    for start in range(0, n_queries, block_size):
        stop = min(start + block_size, n_queries)
        block = query_embeddings[start:stop] @ document_embeddings.T
        part = np.argpartition(-block, depth - 1, axis=1)[:, :depth]
        block_scores = np.take_along_axis(block, part, axis=1)
        order = np.argsort(-block_scores, axis=1, kind="stable")
        indices[start:stop] = np.take_along_axis(part, order, axis=1)
        scores[start:stop] = np.take_along_axis(block_scores, order, axis=1)
        del block

    return CandidatePool(indices=indices, scores=scores)


def document_key(text: str) -> str:
    """Stable id for a document, derived from its text.

    The hub profile is computed offline against one corpus ordering and applied
    later against another (MTEB builds its own). Keying on content rather than
    row index means a reordering cannot silently shift every hub score onto the
    wrong document.
    """
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def save_hub_profile(path: Path, document_texts: Sequence[str], hubness: np.ndarray) -> Path:
    """Write a content-keyed hub profile."""
    if len(document_texts) != len(hubness):
        raise ValueError(
            f"{len(document_texts)} documents but {len(hubness)} hub scores"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        keys=np.array([document_key(t) for t in document_texts]),
        values=np.asarray(hubness, dtype=np.float32),
    )
    return path


def load_hub_profile(path: Path, document_texts: Sequence[str]) -> np.ndarray:
    """Align a saved hub profile onto ``document_texts``.

    Documents absent from the profile score 0.0 — an unknown hub score is not a
    penalty, it is simply no evidence.
    """
    with np.load(path, allow_pickle=False) as payload:
        lookup = dict(zip(payload["keys"].tolist(), payload["values"].tolist()))
    out = np.zeros(len(document_texts), dtype=np.float32)
    for row, text in enumerate(document_texts):
        value = lookup.get(document_key(text))
        if value is not None:
            out[row] = value
    return out


def hubness_from_pool(pool: CandidatePool, n_documents: int, *, top_k: int = 10) -> np.ndarray:
    """Hub score per document, from how often it enters a reference top-``k``.

    A few snippets — short ``print``-only scripts, generic templates — sit near
    the centroid of the embedding space and crowd the top of almost every
    ranking.  Counting how often each document enters a *reference* query set's
    top-``k`` measures that, and keeping the reference set disjoint from the
    split being scored keeps the statistic inductive.
    """
    k = min(top_k, pool.depth)
    counts = np.bincount(pool.indices[:, :k].reshape(-1), minlength=n_documents).astype(np.float32)
    expected = pool.n_queries * k / max(1, n_documents)
    excess = np.maximum(0.0, counts - expected)
    scale = float(excess.max())
    if scale <= 0.0:
        return np.zeros(n_documents, dtype=np.float32)
    return (excess / scale).astype(np.float32)
