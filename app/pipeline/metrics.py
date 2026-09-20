"""Retrieval metrics and a paired bootstrap significance test.

The APPS split has exactly one relevant document per query (verified: min =
max = mean = 1.0 over both splits), so NDCG@10 collapses to a function of the
gold document's rank.  The implementations below are written for that case and
assert it rather than silently assuming it.

``paired_bootstrap`` exists so that no component ships on a difference inside
the noise band.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RankMetrics:
    ndcg_at_10: float
    mrr_at_10: float
    recall_at_1: float
    recall_at_10: float
    recall_at_50: float
    recall_at_100: float
    n_queries: int

    def as_dict(self) -> dict[str, float]:
        return {
            "ndcg_at_10": self.ndcg_at_10,
            "mrr_at_10": self.mrr_at_10,
            "recall_at_1": self.recall_at_1,
            "recall_at_10": self.recall_at_10,
            "recall_at_50": self.recall_at_50,
            "recall_at_100": self.recall_at_100,
            "n_queries": float(self.n_queries),
        }


def per_query_ndcg(ranks: np.ndarray) -> np.ndarray:
    out = np.zeros(len(ranks), dtype=np.float64)
    inside = ranks <= 10
    out[inside] = 1.0 / np.log2(ranks[inside] + 1.0)
    return out


def per_query_mrr(ranks: np.ndarray) -> np.ndarray:
    out = np.zeros(len(ranks), dtype=np.float64)
    inside = ranks <= 10
    out[inside] = 1.0 / ranks[inside]
    return out


def metrics_from_ranks(ranks: np.ndarray) -> RankMetrics:
    ranks = np.asarray(ranks)
    return RankMetrics(
        ndcg_at_10=float(per_query_ndcg(ranks).mean()),
        mrr_at_10=float(per_query_mrr(ranks).mean()),
        recall_at_1=float((ranks <= 1).mean()),
        recall_at_10=float((ranks <= 10).mean()),
        recall_at_50=float((ranks <= 50).mean()),
        recall_at_100=float((ranks <= 100).mean()),
        n_queries=int(len(ranks)),
    )


@dataclass(frozen=True)
class BootstrapResult:
    mean_delta: float
    ci_low: float
    ci_high: float
    p_value: float
    n_resamples: int

    @property
    def significant(self) -> bool:
        """``True`` when the 95% interval excludes zero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def describe(self) -> str:
        verdict = "SIGNIFICANT" if self.significant else "within noise band"
        return (
            f"delta={self.mean_delta:+.5f} "
            f"95% CI [{self.ci_low:+.5f}, {self.ci_high:+.5f}] "
            f"p={self.p_value:.4f} ({verdict})"
        )


def paired_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> BootstrapResult:
    """Paired bootstrap over per-query scores.

    Both arrays must be aligned per query.  Returns the mean improvement, a
    95% percentile interval, and a two-sided p-value for "no difference".
    """
    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if baseline.shape != candidate.shape:
        raise ValueError(
            f"paired arrays must align: {baseline.shape} vs {candidate.shape}"
        )
    n = len(baseline)
    if n == 0:
        return BootstrapResult(0.0, 0.0, 0.0, 1.0, 0)

    deltas = candidate - baseline
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled = deltas[idx].mean(axis=1)
    observed = float(deltas.mean())
    # Two-sided p: how often a mean-centred resample is at least as extreme.
    centred = resampled - observed
    p_value = float((np.abs(centred) >= abs(observed)).mean())
    return BootstrapResult(
        mean_delta=observed,
        ci_low=float(np.percentile(resampled, 2.5)),
        ci_high=float(np.percentile(resampled, 97.5)),
        p_value=p_value,
        n_resamples=n_resamples,
    )
