from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
QUERY_EXPANSIONS = {
    "preprocessed": ("preprocess", "normalize", "trim", "sanitize"),
    "preprocessing": ("preprocess", "normalize", "trim", "sanitize"),
    "main": ("forward", "handler", "entry"),
    "input": ("argument", "value", "data"),
}


@dataclass(frozen=True)
class Document:
    document_id: str
    file_path: str
    content: str
    name: str = ""
    type: str = ""
    start_line: int = 1
    end_line: int = 1


@dataclass(frozen=True)
class RankedDocument:
    document: Document
    score: float
    sources: tuple[str, ...]


def normalize_identifier(value: str) -> str:
    return "".join(char.lower() for char in value if char.isalnum())


def tokenize(value: str) -> list[str]:
    tokens: list[str] = []
    for token in TOKEN_PATTERN.findall(value.lower()):
        tokens.append(token)
        normalized = normalize_identifier(token)
        if normalized != token:
            tokens.append(normalized)
        for part in re.findall(r"[a-z]+|[0-9]+", token):
            if part != token:
                tokens.append(part)
    return tokens


class HybridRetriever:
    def __init__(self, documents: Sequence[Document]) -> None:
        self.documents = list(documents)
        self._tokens = [tokenize(document.content + " " + document.name) for document in self.documents]
        self._document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            self._document_frequency.update(set(tokens))
        self._average_length = sum(map(len, self._tokens)) / max(1, len(self._tokens))

    def _exact(self, query_tokens: list[str]) -> dict[int, float]:
        normalized_query = {normalize_identifier(token) for token in query_tokens}
        results: dict[int, float] = {}
        for index, document in enumerate(self.documents):
            normalized_name = normalize_identifier(document.name)
            if normalized_name and normalized_name in normalized_query:
                results[index] = 1.0
        return results

    def _bm25(self, query_tokens: list[str]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        query_frequency = Counter(query_tokens)
        total_documents = len(self.documents)
        for index, tokens in enumerate(self._tokens):
            frequencies = Counter(tokens)
            length = len(tokens)
            for token, query_count in query_frequency.items():
                term_frequency = frequencies[token]
                if not term_frequency:
                    continue
                document_frequency = self._document_frequency[token]
                inverse_frequency = math.log(
                    1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = term_frequency + 1.5 * (0.25 + 0.75 * length / max(1.0, self._average_length))
                scores[index] += (
                    inverse_frequency * (term_frequency * 2.5 / denominator) * (1 + math.log1p(query_count))
                )
        return dict(scores)

    def _tfidf(self, query_tokens: list[str]) -> dict[int, float]:
        query_counts = Counter(query_tokens)
        query_vector: dict[str, float] = {}
        total_documents = len(self.documents)
        for token, count in query_counts.items():
            query_vector[token] = (1 + math.log(count)) * math.log(
                1 + total_documents / (1 + self._document_frequency[token])
            )
        query_norm = math.sqrt(sum(value * value for value in query_vector.values())) or 1.0
        scores: dict[int, float] = {}
        for index, tokens in enumerate(self._tokens):
            counts = Counter(tokens)
            vector = {
                token: (1 + math.log(count)) * math.log(1 + total_documents / (1 + self._document_frequency[token]))
                for token, count in counts.items()
                if token in query_vector
            }
            dot = sum(query_vector[token] * value for token, value in vector.items())
            norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
            score = dot / (query_norm * norm)
            if score > 0:
                scores[index] = score
        return scores

    @staticmethod
    def _rrf(rankings: Iterable[dict[int, float]], k: int = 60) -> dict[int, float]:
        fused: defaultdict[int, float] = defaultdict(float)
        for ranking in rankings:
            ordered = sorted(ranking, key=ranking.get, reverse=True)
            for rank, index in enumerate(ordered, start=1):
                fused[index] += 1 / (k + rank)
        return dict(fused)

    def search(self, query: str, limit: int = 10) -> list[RankedDocument]:
        query_tokens = tokenize(query)
        for token in list(query_tokens):
            query_tokens.extend(QUERY_EXPANSIONS.get(token, ()))
        exact = self._exact(query_tokens)
        bm25 = self._bm25(query_tokens)
        tfidf = self._tfidf(query_tokens)
        fused = self._rrf((exact, bm25, tfidf))
        ranked = sorted(fused, key=fused.get, reverse=True)[:limit]
        results: list[RankedDocument] = []
        for index in ranked:
            sources = tuple(
                source for source, ranking in (("exact", exact), ("bm25", bm25), ("tfidf", tfidf)) if index in ranking
            )
            results.append(RankedDocument(self.documents[index], fused[index], sources))
        return results
