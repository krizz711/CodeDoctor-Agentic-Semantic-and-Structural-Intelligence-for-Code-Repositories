from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, Optional, Sequence
import numpy as np

from app.retrieval.encode import CodeEncoder, get_shared_encoder


TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


@dataclass(frozen=True)
class Document:
    document_id: str
    file_path: str
    content: str
    name: str = ""
    type: str = ""
    start_line: int = 1
    end_line: int = 1
    start_byte: int = 0
    end_byte: int = 0
    signature: str = ""


@dataclass(frozen=True)
class RankedDocument:
    document: Document
    score: float
    sources: tuple[str, ...]
    signals: dict[str, float] | None = None


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


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, dict[int, float]],
    weights: Mapping[str, float] | None = None,
    k: int = 60,
) -> dict[int, float]:
    """
    Unified Reciprocal Rank Fusion rule:
    R(x) = sum_m w_m * 1[x in L_m] / (k + rank_m(x))
    Graph or specific method candidates receive only their retrieved features.
    Never invents a rank for a method that did not retrieve the item.
    """
    if weights is None:
        weights = {method: 1.0 for method in ranked_lists}

    fused_scores: defaultdict[int, float] = defaultdict(float)

    for method, scores in ranked_lists.items():
        weight = weights.get(method, 1.0)
        # Sort items by score descending to get ranks 1..N
        sorted_indices = sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)
        for rank, doc_index in enumerate(sorted_indices, start=1):
            fused_scores[doc_index] += weight * (1.0 / (k + rank))

    return dict(fused_scores)


class HybridRetriever:
    def __init__(
        self,
        documents: Sequence[Document],
        encoder: Optional[CodeEncoder] = None,
        use_dense: bool = False,
        document_embeddings: Optional[np.ndarray] = None,
    ) -> None:
        """
        ``document_embeddings`` supplies vectors the indexer already computed,
        aligned row-for-row with ``documents``.  Supplying them is what keeps
        query latency in milliseconds: without them the dense path has to
        encode the whole corpus on the first query.
        """
        self.documents = list(documents)
        self.encoder = encoder
        self.use_dense = use_dense
        self._doc_embeddings: Optional[np.ndarray] = None
        if document_embeddings is not None:
            embeddings = np.asarray(document_embeddings, dtype=np.float32)
            if embeddings.shape[0] != len(self.documents):
                raise ValueError(
                    f"{embeddings.shape[0]} precomputed embeddings for "
                    f"{len(self.documents)} documents"
                )
            self._doc_embeddings = embeddings
            self.use_dense = True

        # Build lexical index
        self._tokens = [tokenize(f"{doc.content} {doc.name} {doc.signature}") for doc in self.documents]
        self._document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            self._document_frequency.update(set(tokens))
        self._average_length = sum(map(len, self._tokens)) / max(1, len(self._tokens))

    def _ensure_dense_index(self) -> None:
        if self._doc_embeddings is None and self.documents:
            if self.encoder is None:
                self.encoder = get_shared_encoder()
            texts = [
                f"{doc.signature}\n{doc.content}" if doc.signature else doc.content
                for doc in self.documents
            ]
            self._doc_embeddings = self.encoder.encode(texts, mode="document")

    def _exact(self, query_tokens: list[str]) -> dict[int, float]:
        normalized_query = {normalize_identifier(token) for token in query_tokens if token}
        results: dict[int, float] = {}
        for index, document in enumerate(self.documents):
            normalized_name = normalize_identifier(document.name)
            if normalized_name and normalized_name in normalized_query:
                results[index] = 10.0
            elif any(token in normalized_name for token in normalized_query if len(token) >= 3):
                results[index] = 5.0
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
                inverse_frequency = math.log(1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5))
                denominator = term_frequency + 1.5 * (0.25 + 0.75 * length / max(1.0, self._average_length))
                scores[index] += inverse_frequency * (term_frequency * 2.5 / denominator) * (1 + math.log1p(query_count))
        return {idx: s for idx, s in scores.items() if s > 0}

    def _dense(self, query: str, top_k: int = 50) -> dict[int, float]:
        if not self.use_dense or not self.documents:
            return {}
        self._ensure_dense_index()
        query_emb = self.encoder.encode(query, mode="query")
        scores = self.encoder.similarity(query_emb, self._doc_embeddings)[0]
        # Return top-k dense candidates
        top_indices = np.argsort(scores)[::-1][:top_k]
        return {int(idx): float(scores[idx]) for idx in top_indices if scores[idx] > 0}

    @staticmethod
    def _rrf(rankings: Iterable[dict[int, float]], k: int = 60) -> dict[int, float]:
        """Backwards-compatible wrapper around reciprocal_rank_fusion."""
        named = {f"method_{i}": r for i, r in enumerate(rankings)}
        return reciprocal_rank_fusion(named, k=k)

    def search(
        self,
        query: str,
        limit: int = 10,
        weights: Optional[Mapping[str, float]] = None,
    ) -> list[RankedDocument]:
        query_tokens = tokenize(query)

        ranked_lists: dict[str, dict[int, float]] = {}

        exact_results = self._exact(query_tokens)
        if exact_results:
            ranked_lists["exact"] = exact_results

        bm25_results = self._bm25(query_tokens)
        if bm25_results:
            ranked_lists["bm25"] = bm25_results

        if self.use_dense:
            dense_results = self._dense(query, top_k=max(limit * 3, 20))
            if dense_results:
                ranked_lists["dense"] = dense_results

        if not ranked_lists:
            return []

        default_weights = {"exact": 1.0, "bm25": 1.0, "dense": 1.0}
        active_weights = dict(default_weights)
        if weights:
            active_weights.update(weights)

        fused = reciprocal_rank_fusion(ranked_lists, weights=active_weights, k=60)
        ranked = sorted(fused.keys(), key=lambda idx: fused[idx], reverse=True)[:limit]

        results: list[RankedDocument] = []
        for index in ranked:
            sources = tuple(
                method for method, res_dict in ranked_lists.items() if index in res_dict
            )
            signals = {
                method: res_dict[index] for method, res_dict in ranked_lists.items() if index in res_dict
            }
            results.append(RankedDocument(self.documents[index], fused[index], sources, signals))
        return results
