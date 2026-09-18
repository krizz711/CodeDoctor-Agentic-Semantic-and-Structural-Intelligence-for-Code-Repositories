"""Second-pass reranking over the dense candidate pool (Theme 01 axis 5).

The reranker is a linear model over :mod:`app.pipeline.features`.  It is
deliberately linear: with nine features and a single relevant document per
query, a linear score is as accurate as anything heavier, runs in microseconds
per candidate on CPU, and its weights are printable — which is what lets the
submission argue *why* a ranking changed rather than only that it did.

Weights are fitted by coordinate ascent directly on NDCG@10 over the **train**
split (``scripts/tune_reranker.py``).  The test split is never read during
fitting; :mod:`app.evaluation` enforces that with the leakage firewall.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from app.pipeline.features import (
    FEATURE_NAMES,
    FamilyCompatibility,
    FeatureExtractor,
)
from app.pipeline.query_analysis import QueryFacts

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "config" / "reranker.json"

#: Identity configuration: pass the dense score straight through.  Used as the
#: ablation baseline and as the fallback when no tuned weights are present.
_IDENTITY_WEIGHTS = tuple(1.0 if name == "dense" else 0.0 for name in FEATURE_NAMES)


@dataclass
class RerankerConfig:
    """A fitted reranker: feature weights plus the pool depth they were fit at."""

    weights: tuple[float, ...] = _IDENTITY_WEIGHTS
    candidate_depth: int = 50
    family_compat: FamilyCompatibility = field(default_factory=FamilyCompatibility)
    length_prior_slope: float = 0.0
    length_prior_intercept: float = 0.0
    hubness_top_k: int = 10
    #: Free-form record of how these weights were produced.
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.weights) != len(FEATURE_NAMES):
            raise ValueError(
                f"expected {len(FEATURE_NAMES)} weights for {FEATURE_NAMES}, "
                f"got {len(self.weights)}"
            )

    @property
    def is_identity(self) -> bool:
        return tuple(self.weights) == _IDENTITY_WEIGHTS

    def describe(self) -> str:
        parts = [f"{n}={w:+.4f}" for n, w in zip(FEATURE_NAMES, self.weights) if w]
        return f"depth={self.candidate_depth} " + " ".join(parts)

    def to_json(self) -> dict[str, object]:
        return {
            "feature_names": list(FEATURE_NAMES),
            "weights": list(self.weights),
            "candidate_depth": self.candidate_depth,
            "family_compat": self.family_compat.to_json(),
            "length_prior_slope": self.length_prior_slope,
            "length_prior_intercept": self.length_prior_intercept,
            "hubness_top_k": self.hubness_top_k,
            "provenance": self.provenance,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "RerankerConfig":
        names = payload.get("feature_names")
        if names is not None and list(names) != list(FEATURE_NAMES):
            raise ValueError(
                "reranker weights were fitted against a different feature set: "
                f"{names} != {list(FEATURE_NAMES)}. Refit with scripts/tune_reranker.py."
            )
        return cls(
            weights=tuple(float(w) for w in payload["weights"]),
            candidate_depth=int(payload.get("candidate_depth", 50)),
            family_compat=FamilyCompatibility.from_json(payload.get("family_compat", {}))
            if payload.get("family_compat")
            else FamilyCompatibility(),
            length_prior_slope=float(payload.get("length_prior_slope", 0.0)),
            length_prior_intercept=float(payload.get("length_prior_intercept", 0.0)),
            hubness_top_k=int(payload.get("hubness_top_k", 10)),
            provenance=dict(payload.get("provenance", {})),
        )

    @classmethod
    def load(cls, path: Path | None = None) -> "RerankerConfig":
        """Load fitted weights, falling back to the identity configuration.

        A missing weights file is not an error: it means "dense only", which
        is a valid and reproducible configuration.
        """
        target = path or DEFAULT_WEIGHTS_PATH
        if not target.exists():
            return cls()
        return cls.from_json(json.loads(target.read_text(encoding="utf-8")))

    def save(self, path: Path | None = None) -> Path:
        target = path or DEFAULT_WEIGHTS_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        return target


class Reranker:
    """Applies a :class:`RerankerConfig` to a dense score matrix."""

    def __init__(self, config: RerankerConfig, extractor: FeatureExtractor) -> None:
        self.config = config
        self.extractor = extractor
        self._weights = np.asarray(config.weights, dtype=np.float32)

    def rescore_row(
        self,
        query_facts: QueryFacts,
        row_scores: np.ndarray,
    ) -> np.ndarray:
        """Rerank one query's full score row in place-safe fashion.

        Only the top ``candidate_depth`` documents are rescored; everything
        below keeps its dense score, shifted down so that no untouched
        document can overtake a rescored one.  That preserves the dense tail
        ordering exactly, which is what makes recall@100 an honest ceiling.
        """
        depth = min(self.config.candidate_depth, row_scores.shape[0])
        if depth <= 0 or self.config.is_identity:
            return row_scores

        candidates = np.argpartition(-row_scores, depth - 1)[:depth]
        candidates = candidates[np.argsort(-row_scores[candidates], kind="stable")]
        candidate_scores = row_scores[candidates]

        features = self.extractor.for_query(query_facts, candidates, candidate_scores)
        new_scores = features @ self._weights

        out = row_scores.copy()
        # Shift the reranked block above the untouched tail while keeping its
        # internal order, so the pool boundary never reshuffles the tail.
        tail_max = float(np.max(np.delete(row_scores, candidates))) if depth < len(row_scores) else -np.inf
        span = float(new_scores.max() - new_scores.min()) if depth > 1 else 0.0
        offset = tail_max + 1.0 + span - float(new_scores.max()) if np.isfinite(tail_max) else 0.0
        out[candidates] = new_scores + offset
        return out

    def rescore(
        self,
        scores: np.ndarray,
        query_facts: Sequence[QueryFacts],
    ) -> np.ndarray:
        """Rerank a full ``[n_queries, n_documents]`` score matrix."""
        if self.config.is_identity:
            return scores
        if len(query_facts) != scores.shape[0]:
            raise ValueError(
                f"{len(query_facts)} query facts for {scores.shape[0]} score rows"
            )
        out = np.empty_like(scores)
        for row in range(scores.shape[0]):
            out[row] = self.rescore_row(query_facts[row], scores[row])
        return out
