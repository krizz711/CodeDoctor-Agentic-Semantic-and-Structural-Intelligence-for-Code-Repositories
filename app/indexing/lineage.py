"""Retrieval across versions of *snippets* (the Bonus goal), ranked, not hidden.

``VersionedIndex.retrieve_across_versions`` groups results by ``(path,
symbol)`` and keeps the best-scoring variant of each group.  That removes the
crowding the brief warns about, but it answers a different question: *which*
version of a snippet should win is decided by a dense score that is nearly
identical across versions -- ExecRetrieval (2026) measured that embeddings rank
a buggy near-clone above the correct code in 67-78% of queries -- and a snippet
corpus has no paths or symbols to group by.

This module keeps three things separate:

1. **Content addressing.**  Every distinct text is stored once, keyed by its
   hash, with the set of versions it appears in.  A new version adds only new
   contents, so re-indexing costs what changed (P1), and a snippet unchanged
   across ten versions is scored once, not ten times.
2. **Lineage.**  Versions of "the same" snippet form a lineage: an explicit key
   when the corpus has one (a document id, a path and symbol), otherwise
   near-duplicate clustering over token shingles.
3. **Ranking inside a lineage** by evidence the encoder cannot supply:
   execution verification first (the version that reproduces the statement's
   worked examples), then the dense score, then recency.

Nothing here runs code itself: callers pass the execution tiers they obtained
from :mod:`app.pipeline.execution_rerank`.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable(parts: Sequence[str]) -> int:
    # Not the built-in hash(): it is salted per process, so clusters would
    # differ between runs.
    return int.from_bytes(hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=8).digest(), "big")


def shingles(text: str, k: int = 5) -> set[int]:
    tokens = _TOKEN.findall(text)
    if len(tokens) < k:
        return {_stable(tokens)}
    return {_stable(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


@dataclass
class Content:
    hash: str
    text: str
    #: version ids this exact text appears in, in insertion order
    versions: list[str] = field(default_factory=list)
    #: explicit lineage keys it was registered under (usually one)
    keys: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class LineageHit:
    """One lineage in a cross-version ranking."""

    lineage: str
    #: the content chosen to represent the lineage
    best_hash: str
    best_text: str
    #: versions in which ``best_text`` appears
    best_versions: tuple[str, ...]
    #: every version the lineage appears in
    all_versions: tuple[str, ...]
    #: distinct contents in the lineage (1 = unchanged across versions)
    variants: int
    dense: float
    tier: str
    #: why this content represents the lineage
    reason: str


_TIER_ORDER = {"strong": 2, "weak": 1, "none": 0}


class LineageStore:
    """Content-addressed versions of a snippet corpus."""

    def __init__(self) -> None:
        self.contents: dict[str, Content] = {}
        self.version_order: list[str] = []
        #: explicit key -> [(version, hash)]
        self.history: dict[str, list[tuple[str, str]]] = {}

    # -- building -----------------------------------------------------------

    def add_version(self, version: str, snippets: Iterable[tuple[str, str]]) -> dict[str, int]:
        """Register ``(key, text)`` pairs as ``version``.  Returns what was new."""
        if version in self.version_order:
            raise ValueError(f"version {version!r} already indexed")
        self.version_order.append(version)
        new = reused = 0
        for key, text in snippets:
            digest = content_hash(text)
            content = self.contents.get(digest)
            if content is None:
                content = Content(digest, text)
                self.contents[digest] = content
                new += 1
            else:
                reused += 1
            if version not in content.versions:
                content.versions.append(version)
            content.keys.add(key)
            self.history.setdefault(key, []).append((version, digest))
        return {"new_contents": new, "reused_contents": reused}

    def unique_texts(self) -> tuple[list[str], list[str]]:
        """``(hashes, texts)`` of every distinct content -- what gets embedded."""
        hashes = list(self.contents)
        return hashes, [self.contents[h].text for h in hashes]

    # -- lineage --------------------------------------------------------------

    def lineage_of(self, digest: str) -> str:
        keys = self.contents[digest].keys
        return sorted(keys)[0] if keys else digest

    def cluster_by_similarity(self, hashes: Sequence[str], threshold: float = 0.7) -> dict[str, str]:
        """Near-duplicate lineage for corpora without explicit keys.

        Greedy single-link over the given hashes, visited in the given order
        (pass them best-first); a content joins the first cluster whose seed it
        overlaps at ``threshold`` Jaccard or more.
        """
        seeds: list[tuple[str, set[int]]] = []
        assignment: dict[str, str] = {}
        for digest in hashes:
            sig = shingles(self.contents[digest].text)
            for seed, seed_sig in seeds:
                if jaccard(sig, seed_sig) >= threshold:
                    assignment[digest] = seed
                    break
            else:
                seeds.append((digest, sig))
                assignment[digest] = digest
        return assignment

    # -- ranking ----------------------------------------------------------------

    def rank(
        self,
        dense: dict[str, float],
        *,
        tiers: dict[str, str] | None = None,
        limit: int = 10,
        lineage: Callable[[str], str] | None = None,
        recency: bool = True,
    ) -> list[LineageHit]:
        """Rank lineages; inside each, pick the version the evidence supports.

        ``dense`` maps content hash -> dense score for the candidates retrieved
        (usually the top-K unique contents).  ``tiers`` maps content hash ->
        ``strong``/``weak``/``none`` from execution verification.  A lineage's
        position is set by its best member's (tier, dense score); its
        representative is the member with the highest tier, then dense score,
        then the most recent version.
        """
        tiers = tiers or {}
        lineage = lineage or self.lineage_of
        position = {v: i for i, v in enumerate(self.version_order)}
        groups: dict[str, list[str]] = {}
        for digest in dense:
            groups.setdefault(lineage(digest), []).append(digest)

        def latest(digest: str) -> int:
            return max((position.get(v, -1) for v in self.contents[digest].versions), default=-1)

        hits: list[LineageHit] = []
        for key, members in groups.items():
            def order(d: str) -> tuple:
                return (
                    _TIER_ORDER.get(tiers.get(d, "none"), 0),
                    dense[d],
                    latest(d) if recency else 0,
                )

            members.sort(key=order, reverse=True)
            best = members[0]
            tier = tiers.get(best, "none")
            if tier != "none":
                reason = f"execution: {tier}"
            elif len(members) > 1 and dense[best] - max(dense[m] for m in members[1:]) < 1e-9:
                reason = "tie on dense score: most recent version"
            else:
                reason = "highest dense score"
            all_versions: list[str] = []
            for member in members:
                for version in self.contents[member].versions:
                    if version not in all_versions:
                        all_versions.append(version)
            content = self.contents[best]
            hits.append(
                LineageHit(
                    lineage=key,
                    best_hash=best,
                    best_text=content.text,
                    best_versions=tuple(content.versions),
                    all_versions=tuple(sorted(all_versions, key=lambda v: position.get(v, -1))),
                    variants=len(members),
                    dense=float(dense[best]),
                    tier=tier,
                    reason=reason,
                )
            )
        hits.sort(key=lambda h: (_TIER_ORDER.get(h.tier, 0), h.dense), reverse=True)
        return hits[:limit]
