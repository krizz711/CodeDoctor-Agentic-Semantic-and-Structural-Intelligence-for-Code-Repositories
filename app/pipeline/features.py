"""Candidate features for the second retrieval pass (Theme 01 axis 5).

The first pass is dense and returns a candidate pool.  This module turns each
``(query, candidate)`` pair into a small, fixed vector of *interpretable*
signals that the dense encoder does not model well:

* whether the query's implied answer shape matches the snippet's actual shape,
* whether identifiers the statement names are actually defined by the snippet,
* whether constraint / sample-output literals appear verbatim in the snippet,
* whether the snippet is a *hub* that scores highly against unrelated queries.

Every feature is computed from text alone.  Nothing reads dataset metadata,
document ids, or corpus partitions: those exist in the raw APPS files and
would leak, so the pipeline is built to never see them.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from app.pipeline.code_analysis import CodeFacts, CodeFamily, analyze_code
from app.pipeline.query_analysis import QueryFacts, QueryFamily, analyze_query


#: Order is load-bearing: learned weights are stored positionally.
FEATURE_NAMES: tuple[str, ...] = (
    "dense",
    "dense_margin",
    "family_compat",
    "defined_name_hit",
    "ident_idf_overlap",
    "number_idf_overlap",
    "string_idf_overlap",
    "length_prior",
    "hub_penalty",
)

_QUERY_FAMILIES: tuple[QueryFamily, ...] = (
    QueryFamily.COMPETITIVE_STDIN,
    QueryFamily.LEETCODE_METHOD,
    QueryFamily.KATA_FUNCTION,
    QueryFamily.UNKNOWN,
)
_CODE_FAMILIES: tuple[CodeFamily, ...] = (
    CodeFamily.STDIN_SCRIPT,
    CodeFamily.CLASS_SOLUTION,
    CodeFamily.FUNCTION_LIBRARY,
    CodeFamily.UNKNOWN,
)


class InverseDocumentFrequency:
    """IDF over the indexed corpus only — never over queries.

    Keeping the statistics corpus-side makes the feature *inductive*: adding a
    query changes nothing, so a tuned weight measured on one query split
    transfers to another.
    """

    def __init__(self, document_token_sets: Iterable[frozenset[str]]) -> None:
        counts: Counter[str] = Counter()
        total = 0
        for tokens in document_token_sets:
            counts.update(tokens)
            total += 1
        self._total = max(1, total)
        self._df = counts
        self._default = math.log(1.0 + self._total)

    def weight(self, token: str) -> float:
        df = self._df.get(token, 0)
        if df == 0:
            return self._default
        return math.log(1.0 + self._total / df)

    def overlap(self, left: frozenset[str], right: frozenset[str]) -> float:
        """IDF-weighted overlap, normalized by the query side's own IDF mass."""
        if not left or not right:
            return 0.0
        shared = left & right
        if not shared:
            return 0.0
        numerator = sum(self.weight(t) for t in shared)
        denominator = sum(self.weight(t) for t in left)
        if denominator <= 0.0:
            return 0.0
        return numerator / denominator


@dataclass
class CorpusStatistics:
    """Everything derived once from the indexed corpus."""

    facts: list[CodeFacts]
    identifier_idf: InverseDocumentFrequency
    number_idf: InverseDocumentFrequency
    string_idf: InverseDocumentFrequency
    #: log(1 + characters) per document, used by the length prior.
    log_length: np.ndarray
    #: Hub score per document in [0, 1]; filled by
    #: :func:`app.pipeline.search.hubness_from_pool`, which computes it from a
    #: reference query set disjoint from the split being scored.
    hubness: np.ndarray

    @classmethod
    def build(cls, documents: Sequence[str]) -> "CorpusStatistics":
        facts = [analyze_code(text) for text in documents]
        return cls(
            facts=facts,
            identifier_idf=InverseDocumentFrequency(f.identifiers for f in facts),
            number_idf=InverseDocumentFrequency(f.numbers for f in facts),
            string_idf=InverseDocumentFrequency(f.strings for f in facts),
            log_length=np.array([math.log1p(f.char_length) for f in facts], dtype=np.float32),
            hubness=np.zeros(len(facts), dtype=np.float32),
        )


#: Prior compatibility between a query's implied shape and a snippet's actual
#: shape.  Values are log-odds-ish nudges in [-1, 1]; UNKNOWN rows are all
#: zero so an unclassified query contributes nothing rather than a guess.
_DEFAULT_FAMILY_COMPAT: dict[tuple[str, str], float] = {
    (QueryFamily.COMPETITIVE_STDIN.value, CodeFamily.STDIN_SCRIPT.value): 1.0,
    (QueryFamily.COMPETITIVE_STDIN.value, CodeFamily.CLASS_SOLUTION.value): -1.0,
    (QueryFamily.COMPETITIVE_STDIN.value, CodeFamily.FUNCTION_LIBRARY.value): -0.5,
    (QueryFamily.LEETCODE_METHOD.value, CodeFamily.CLASS_SOLUTION.value): 1.0,
    (QueryFamily.LEETCODE_METHOD.value, CodeFamily.STDIN_SCRIPT.value): -1.0,
    (QueryFamily.LEETCODE_METHOD.value, CodeFamily.FUNCTION_LIBRARY.value): 0.0,
    (QueryFamily.KATA_FUNCTION.value, CodeFamily.FUNCTION_LIBRARY.value): 1.0,
    (QueryFamily.KATA_FUNCTION.value, CodeFamily.STDIN_SCRIPT.value): -1.0,
    (QueryFamily.KATA_FUNCTION.value, CodeFamily.CLASS_SOLUTION.value): -0.5,
}


class FamilyCompatibility:
    """A 4x4 table of query-shape / snippet-shape affinities.

    Starts from hand-set priors and is replaced by counts estimated on the
    train split (``scripts/tune_reranker.py``).  Estimation uses only train
    qrels, so the table never sees a test label.
    """

    def __init__(self, table: Mapping[tuple[str, str], float] | None = None) -> None:
        self.table = dict(_DEFAULT_FAMILY_COMPAT if table is None else table)

    def score(self, query_facts: QueryFacts, code_facts: CodeFacts) -> float:
        if not query_facts.is_verified():
            return 0.0
        raw = self.table.get((query_facts.family.value, code_facts.family.value), 0.0)
        return raw * query_facts.confidence

    def to_json(self) -> dict[str, float]:
        return {f"{q}|{c}": v for (q, c), v in self.table.items()}

    @classmethod
    def from_json(cls, payload: Mapping[str, float]) -> "FamilyCompatibility":
        table: dict[tuple[str, str], float] = {}
        for key, value in payload.items():
            query_family, _, code_family = key.partition("|")
            table[(query_family, code_family)] = float(value)
        return cls(table)

    @classmethod
    def estimate(
        cls,
        query_families: Sequence[QueryFamily],
        gold_code_families: Sequence[CodeFamily],
    ) -> "FamilyCompatibility":
        """Estimate from observed (query family, gold snippet family) pairs.

        Cell value is the pointwise mutual information between the two
        families, clipped to [-1, 1].
        """
        joint: Counter[tuple[str, str]] = Counter()
        q_margin: Counter[str] = Counter()
        c_margin: Counter[str] = Counter()
        total = 0
        for qf, cf in zip(query_families, gold_code_families):
            joint[(qf.value, cf.value)] += 1
            q_margin[qf.value] += 1
            c_margin[cf.value] += 1
            total += 1
        if total == 0:
            return cls()

        table: dict[tuple[str, str], float] = {}
        for qf in _QUERY_FAMILIES:
            for cf in _CODE_FAMILIES:
                n_joint = joint.get((qf.value, cf.value), 0)
                n_q = q_margin.get(qf.value, 0)
                n_c = c_margin.get(cf.value, 0)
                if n_q == 0 or n_c == 0:
                    table[(qf.value, cf.value)] = 0.0
                    continue
                # Add-one smoothing keeps empty cells finite.
                p_joint = (n_joint + 1) / (total + len(_QUERY_FAMILIES) * len(_CODE_FAMILIES))
                p_q = n_q / total
                p_c = n_c / total
                pmi = math.log(p_joint / (p_q * p_c))
                table[(qf.value, cf.value)] = max(-1.0, min(1.0, pmi))
        return cls(table)


class FeatureExtractor:
    """Builds the per-candidate feature matrix for one query."""

    def __init__(
        self,
        corpus_stats: CorpusStatistics,
        family_compat: FamilyCompatibility | None = None,
        *,
        length_prior_slope: float = 0.0,
        length_prior_intercept: float = 0.0,
    ) -> None:
        self.corpus = corpus_stats
        self.family_compat = family_compat or FamilyCompatibility()
        self.length_prior_slope = length_prior_slope
        self.length_prior_intercept = length_prior_intercept

    def for_query(
        self,
        query_facts: QueryFacts,
        candidate_indices: np.ndarray,
        candidate_scores: np.ndarray,
    ) -> np.ndarray:
        """Return a ``[len(candidates), len(FEATURE_NAMES)]`` float32 matrix."""
        n = len(candidate_indices)
        out = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
        if n == 0:
            return out

        top_score = float(candidate_scores[0]) if n else 0.0
        expected_log_length = (
            self.length_prior_intercept
            + self.length_prior_slope * math.log1p(query_facts.char_length)
        )

        ident_idf = self.corpus.identifier_idf
        number_idf = self.corpus.number_idf
        string_idf = self.corpus.string_idf

        q_idents = query_facts.identifiers
        q_numbers = query_facts.numbers
        q_strings = query_facts.strings

        for row, (doc_index, score) in enumerate(zip(candidate_indices, candidate_scores)):
            facts = self.corpus.facts[doc_index]
            out[row, 0] = score
            out[row, 1] = score - top_score
            out[row, 2] = self.family_compat.score(query_facts, facts)
            out[row, 3] = 1.0 if (q_idents & facts.defined_names) else 0.0
            out[row, 4] = ident_idf.overlap(q_idents, facts.identifiers)
            out[row, 5] = number_idf.overlap(q_numbers, facts.numbers)
            out[row, 6] = string_idf.overlap(q_strings, facts.strings)
            if self.length_prior_slope:
                delta = self.corpus.log_length[doc_index] - expected_log_length
                out[row, 7] = -abs(float(delta))
            out[row, 8] = -float(self.corpus.hubness[doc_index])
        return out


def analyze_queries(texts: Sequence[str]) -> list[QueryFacts]:
    return [analyze_query(text) for text in texts]
