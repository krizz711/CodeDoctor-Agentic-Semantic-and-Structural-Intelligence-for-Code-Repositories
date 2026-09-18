"""Fitting the reranker weights on the train split.

Coordinate ascent directly on NDCG@10.  Nine weights, one relevant document
per query, and a candidate pool that fits in a few megabytes — so the whole
objective is a vectorised matrix product and a rank count, and a full sweep
takes well under a second.

Fitting reads the **train** split only.  Nothing in this module can see a test
label: ``fit_weights`` is handed a feature tensor and gold columns, and the
caller is responsible for (and audited on) where those came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from app.pipeline.features import FEATURE_NAMES

#: ``dense`` is pinned at 1.0 so every other weight is a correction *on top of*
#: the frozen encoder score, which keeps the learned values interpretable.
_DENSE_INDEX = FEATURE_NAMES.index("dense")

#: Hard cap on any non-dense weight.  Cosine gaps inside a top-50 pool are of
#: order 0.01-0.1, so +-2.0 is already enough authority to reorder the pool
#: freely.  Beyond that the dense score stops mattering at all and the system
#: is no longer "retrieval plus corrections" — it is a hand-built classifier
#: whose behaviour on an unseen query mix is not something the dev split can
#: vouch for.  Coordinate ascent will happily run to -6 if left unbounded.
MAX_ABS_WEIGHT = 2.0

#: Search grid per feature, in units of "dense score points".  Dense cosine
#: differences between rank 1 and rank 10 are typically 0.01-0.08, so the grid
#: has to be fine near zero — but it also has to reach far enough to let a
#: decisive feature (an exact defined-name hit) override the dense order.
_DEFAULT_GRID: tuple[float, ...] = (
    -0.50, -0.30, -0.20, -0.12, -0.08, -0.05, -0.03, -0.02, -0.012, -0.006, -0.003, -0.001,
    0.0,
    0.001, 0.003, 0.006, 0.012, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30, 0.50,
)


@dataclass
class FitResult:
    weights: tuple[float, ...]
    train_ndcg: float
    history: list[tuple[str, float, float]]
    n_sweeps: int


def _neighbourhood(grid: tuple[float, ...], value: float) -> float:
    """Distance from ``value`` to its nearest grid neighbour, for refinement."""
    others = [abs(g - value) for g in grid if g != value]
    return min(others) if others else 0.001


def ranks_for_weights(
    features: np.ndarray,
    gold_columns: np.ndarray,
    weights: np.ndarray,
    *,
    pool_miss_rank: int,
) -> np.ndarray:
    """1-based rank of the gold document after scoring the pool with ``weights``.

    ``features`` is ``[n_queries, depth, n_features]``; ``gold_columns`` holds
    the gold document's position in each query's pool, or -1 when the dense
    pass missed it entirely (that query can never be recovered by reranking,
    and is scored at ``pool_miss_rank``).

    Ties are broken by the original dense order, matching the stable sort the
    runtime reranker uses.
    """
    scores = features @ weights  # [n_queries, depth]
    n_queries, depth = scores.shape
    ranks = np.full(n_queries, pool_miss_rank, dtype=np.int32)

    present = gold_columns >= 0
    if not present.any():
        return ranks

    rows = np.flatnonzero(present)
    cols = gold_columns[rows]
    gold_scores = scores[rows, cols][:, None]
    block = scores[rows]

    strictly_better = (block > gold_scores).sum(axis=1)
    positions = np.arange(depth)[None, :]
    tied_earlier = ((block == gold_scores) & (positions < cols[:, None])).sum(axis=1)
    ranks[rows] = (1 + strictly_better + tied_earlier).astype(np.int32)
    return ranks


def ndcg_at_10(ranks: np.ndarray) -> float:
    inside = ranks <= 10
    if not inside.any():
        return 0.0
    gains = np.zeros(len(ranks), dtype=np.float64)
    gains[inside] = 1.0 / np.log2(ranks[inside] + 1.0)
    return float(gains.mean())


def fit_weights(
    features: np.ndarray,
    gold_columns: np.ndarray,
    *,
    pool_miss_rank: int,
    grid: Sequence[float] = _DEFAULT_GRID,
    n_sweeps: int = 4,
    initial: np.ndarray | None = None,
    verbose: bool = True,
) -> FitResult:
    """Coordinate ascent on NDCG@10 over the non-dense feature weights."""
    n_features = features.shape[2]
    if n_features != len(FEATURE_NAMES):
        raise ValueError(
            f"feature tensor has {n_features} columns, expected {len(FEATURE_NAMES)}"
        )

    weights = np.zeros(n_features, dtype=np.float32) if initial is None else initial.astype(np.float32).copy()
    weights[_DENSE_INDEX] = 1.0

    best = ndcg_at_10(ranks_for_weights(features, gold_columns, weights, pool_miss_rank=pool_miss_rank))
    history: list[tuple[str, float, float]] = [("<start>", 0.0, best)]
    if verbose:
        print(f"  start NDCG@10 = {best:.5f}")

    order = [i for i in range(n_features) if i != _DENSE_INDEX]
    grid = tuple(grid)
    for sweep in range(n_sweeps):
        improved = False
        for feature_index in order:
            current = float(weights[feature_index])
            # Coarse pass over the grid, then a local refinement between the
            # winning grid point and its neighbours, so the optimum is not
            # pinned to a grid node.
            candidates = [c for c in grid if abs(c) <= MAX_ABS_WEIGHT]
            best_value, best_score = current, best
            for stage in range(2):
                for candidate in candidates:
                    if abs(candidate) > MAX_ABS_WEIGHT:
                        continue
                    if candidate == weights[feature_index]:
                        continue
                    weights[feature_index] = candidate
                    score = ndcg_at_10(
                        ranks_for_weights(
                            features, gold_columns, weights, pool_miss_rank=pool_miss_rank
                        )
                    )
                    if score > best_score + 1e-9:
                        best_score, best_value = score, candidate
                weights[feature_index] = best_value
                if stage == 0:
                    span = _neighbourhood(grid, best_value)
                    candidates = [
                        best_value + span * f for f in (-0.66, -0.33, 0.33, 0.66)
                    ]
                    # A weight parked on the edge of the grid means the grid was
                    # too narrow for a decisive feature: expand geometrically
                    # rather than silently clipping it.
                    if best_value in (grid[0], grid[-1]):
                        candidates += [best_value * m for m in (2.0, 4.0, 8.0, 16.0)]
            if best_score > best + 1e-9:
                if verbose:
                    print(
                        f"  sweep {sweep}: {FEATURE_NAMES[feature_index]:>20s} "
                        f"= {best_value:+.4f}  NDCG@10 {best:.5f} -> {best_score:.5f}"
                    )
                history.append((FEATURE_NAMES[feature_index], best_value, best_score))
                best = best_score
                improved = True
        if not improved:
            if verbose:
                print(f"  converged after {sweep + 1} sweep(s)")
            return FitResult(tuple(float(w) for w in weights), best, history, sweep + 1)

    return FitResult(tuple(float(w) for w in weights), best, history, n_sweeps)


def fit_length_prior(
    query_lengths: np.ndarray,
    gold_document_lengths: np.ndarray,
) -> tuple[float, float]:
    """Least-squares fit of ``log1p(len(doc)) ~ a + b * log1p(len(query))``.

    Longer statements really do get longer solutions, but the relationship is
    weak; the fitted slope is reported so the PPT can state how weak.
    """
    x = np.log1p(np.asarray(query_lengths, dtype=np.float64))
    y = np.log1p(np.asarray(gold_document_lengths, dtype=np.float64))
    if len(x) < 2:
        return 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)
