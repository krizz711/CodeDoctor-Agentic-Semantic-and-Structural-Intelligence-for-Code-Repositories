from __future__ import annotations

import math
from typing import Sequence


def reciprocal_rank(results: Sequence[str], relevant: set[str]) -> float:
    for position, document_id in enumerate(results, start=1):
        if document_id in relevant:
            return 1 / position
    return 0.0


def ndcg_at_k(results: Sequence[str], relevant: set[str], k: int = 10) -> float:
    ranked = results[:k]
    dcg = sum((1 / math.log2(position + 2)) for position, document_id in enumerate(ranked) if document_id in relevant)
    ideal = sum((1 / math.log2(position + 2)) for position in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def evaluate_queries(rows: Sequence[tuple[Sequence[str], set[str]]], k: int = 10) -> dict[str, float]:
    if not rows:
        return {"mrr": 0.0, "ndcg_at_10": 0.0}
    return {
        "mrr": sum(reciprocal_rank(results, relevant) for results, relevant in rows) / len(rows),
        "ndcg_at_10": sum(ndcg_at_k(results, relevant, k) for results, relevant in rows) / len(rows),
    }
