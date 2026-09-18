"""Development splits that actually resemble the split being scored.

Measured on the CoIR APPS queries (text only — no labels, no metadata):

    train (5000):  kata 43.0%  competitive-stdin 34.1%  leetcode 14.6%  unknown 8.2%
    test  (3765):  competitive-stdin 98.4%  leetcode 0.8%  unknown 0.8%  kata 0.1%

The corpus is ordered by source, and the test split is a contiguous tail of
that ordering, so it is almost entirely competitive-programming statements
while the train split is mostly Codewars katas.  Two consequences:

1. A *contiguous* half-split of train is not exchangeable.  Measured with a
   small encoder, the first half scores NDCG@10 0.209 and the second 0.623 —
   a split-half "held-out" number would be reporting distribution shift, not
   generalization.  Splits here are seeded-random.
2. A *uniform* sample of train is not the right development set either.  For
   tuning that has to transfer to the scored split, restrict to the query
   family that split is made of.  ``competitive_stdin`` covers 1,707 train
   queries, which is enough to fit nine weights and still hold half back.

Both facts come from query text, which the retrieval system reads at inference
time anyway.  No qrel, document id, or dataset metadata is consulted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from app.pipeline.query_analysis import QueryFamily, analyze_query

#: The family the scored test split is made of.
TEST_DOMINANT_FAMILY = QueryFamily.COMPETITIVE_STDIN


@dataclass(frozen=True)
class DevelopmentSplit:
    """Row indices into the train arrays for the tuning and held-out halves."""

    tune: np.ndarray
    dev: np.ndarray
    #: Every row that survived the distribution filter, tune and dev together.
    eligible: np.ndarray
    family_filter: str | None
    seed: int

    def describe(self) -> str:
        scope = self.family_filter or "all families"
        return (
            f"{len(self.eligible)} eligible queries ({scope}), "
            f"TUNE={len(self.tune)} DEV={len(self.dev)} seed={self.seed}"
        )


def make_development_split(
    query_texts: Sequence[str],
    *,
    match_family: QueryFamily | None = TEST_DOMINANT_FAMILY,
    seed: int = 20260919,
    dev_fraction: float = 0.5,
) -> DevelopmentSplit:
    """Split train query rows into TUNE and held-out DEV.

    ``match_family`` keeps only queries of that family, so the development set
    matches the distribution of the split that will be scored.  Pass ``None``
    to keep everything (useful as a robustness check, not as the primary
    measurement).
    """
    if match_family is None:
        eligible = np.arange(len(query_texts), dtype=np.int64)
    else:
        eligible = np.array(
            [i for i, text in enumerate(query_texts) if analyze_query(text).family is match_family],
            dtype=np.int64,
        )
    if len(eligible) < 20:
        raise ValueError(
            f"only {len(eligible)} queries match {match_family}; too few to fit and hold out. "
            "Pass match_family=None to use the whole split."
        )

    rng = np.random.default_rng(seed)
    shuffled = eligible.copy()
    rng.shuffle(shuffled)
    cut = int(round(len(shuffled) * (1.0 - dev_fraction)))
    cut = max(10, min(cut, len(shuffled) - 10))
    return DevelopmentSplit(
        tune=np.sort(shuffled[:cut]),
        dev=np.sort(shuffled[cut:]),
        eligible=np.sort(eligible),
        family_filter=match_family.value if match_family else None,
        seed=seed,
    )


def family_distribution(query_texts: Sequence[str]) -> dict[str, float]:
    """Fraction of queries in each family, for reporting the shift."""
    counts: dict[str, int] = {}
    for text in query_texts:
        key = analyze_query(text).family.value
        counts[key] = counts.get(key, 0) + 1
    total = max(1, len(query_texts))
    return {k: v / total for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}
