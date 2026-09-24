"""The MTEB-facing pre/post-processing pipeline encoder.

MTEB drives retrieval through two hooks on :class:`~mteb.models.abs_encoder.AbsEncoder`:

    encode(inputs, prompt_type=query|document) -> [n, d]
    similarity(query_embeddings, document_embeddings) -> [n_q, n_d]

The first pass lives in ``encode``; the second pass lives in ``similarity``,
which is where the reranker sees the score matrix alongside the texts recorded
during encoding.  That is the documented reranking hook, and it is the only
extra information the pipeline uses: raw text, in the order MTEB supplied it.

What the pipeline deliberately does **not** touch, although it is reachable in
the raw CoIR files:

    * ``meta_information`` (carries the original problem URL and starter code),
    * the ``partition`` column (would halve the candidate set for free),
    * document ids (the APPS qrels are the identity map ``q<N> -> d<N>``).

Any of those would lift the score without improving retrieval.  The encoder is
handed text only, and :mod:`app.evaluation` audits that the pipeline never
opens the raw dataset during a scored run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.pipeline.execution_rerank import (
    ExecutionConfig,
    apply_execution_rerank,
    has_executable_examples,
)
from app.pipeline.features import CorpusStatistics, FeatureExtractor
from app.pipeline.query_analysis import QueryFacts, analyze_query
from app.pipeline.reranker import Reranker, RerankerConfig
from app.pipeline.search import load_hub_profile
from app.retrieval.encode import CodeEncoder

logger = logging.getLogger(__name__)


def _texts_from_batch(batch: Any) -> list[str]:
    if isinstance(batch, dict):
        texts = batch.get("text", batch.get("sentences", []))
    elif isinstance(batch, (list, tuple)):
        texts = list(batch)
    else:
        texts = [batch]
    return [t if isinstance(t, str) else str(t) for t in texts]


class PrePostPipelineEncoder:
    """Two-pass retrieval: frozen dense encoder, then feature-based reranking.

    Instantiate with a :class:`~app.retrieval.encode.CodeEncoder` and a fitted
    :class:`~app.pipeline.reranker.RerankerConfig`.  With the identity config
    the class is a faithful pass-through of the published model, which makes
    the ablation exact: same vectors, reranker on or off.
    """

    def __init__(
        self,
        encoder: CodeEncoder,
        reranker_config: RerankerConfig | None = None,
        *,
        model_meta: Any = None,
        enable_rerank: bool = True,
        execution_config: ExecutionConfig | None = None,
        corpus_cache: Any = None,
        hubness_bank: np.ndarray | None = None,
        hubness_beta: float = 0.3,
        hubness_k: int = 10,
    ) -> None:
        self.encoder = encoder
        #: PRE-5 hubness correction.  ``hubness_bank`` holds *query* embeddings
        #: (same encoder, same prompt) of statements whose answers are NOT in
        #: the corpus being searched -- otherwise the penalty also pushes down
        #: documents that answer other queries, which is benchmark structure,
        #: not hubness.  Each document's penalty is its mean top-k similarity
        #: to the bank; computed once per corpus, subtracted at query time.
        self.hubness_bank = None if hubness_bank is None else np.asarray(hubness_bank, dtype=np.float32)
        self.hubness_beta = float(hubness_beta)
        self.hubness_k = int(hubness_k)
        self._hub_penalty: np.ndarray | None = None
        self._hub_penalty_key: tuple | None = None
        #: Optional content-addressed store of *document* vectors
        #: (:class:`app.indexing.cache.EmbeddingCache`).  The corpus is encoded
        #: once, offline, where the FAQ allows a GPU; every later run -- a
        #: judge's re-run, a new corpus version -- encodes only what changed.
        #: Queries never touch it: they are encoded at retrieval time, on CPU.
        self.corpus_cache = corpus_cache
        self.cache_stats: dict[str, int] = {"hits": 0, "misses": 0}
        self.mteb_model_meta = model_meta
        self.reranker_config = reranker_config or RerankerConfig()
        self.enable_rerank = enable_rerank and not self.reranker_config.is_identity
        #: Third pass. Disabled unless the caller passes an enabled config —
        #: it runs dataset-authored code, so it is never on by accident.
        self.execution_config = execution_config or ExecutionConfig()
        #: Populated by the execution pass so the caller can report what ran.
        self.execution_audit: dict[str, int] = {}

        # Texts recorded during encode(), in MTEB's own order.
        self._document_texts: list[str] = []
        self._query_texts: list[str] = []
        self._query_facts: list[QueryFacts] = []
        self._corpus_stats: CorpusStatistics | None = None
        self._reranker: Reranker | None = None
        #: Path to a content-keyed hub profile, if one was precomputed. It is
        #: aligned onto the corpus MTEB actually supplies, by document text, so
        #: a different corpus ordering cannot shift hub scores onto the wrong
        #: documents.
        self.hub_profile_path: Path | None = None

    # -- MTEB hooks --------------------------------------------------------

    def encode(
        self,
        inputs: Any,
        *,
        task_metadata: Any = None,
        hf_split: str = "test",
        hf_subset: str = "default",
        prompt_type: Any = None,
        **kwargs: Any,
    ) -> np.ndarray:
        is_query = prompt_type is not None and "query" in str(prompt_type).lower()
        mode = "query" if is_query else "document"

        texts: list[str] = []
        if hasattr(inputs, "__iter__") and not isinstance(inputs, (str, dict)):
            for batch in inputs:
                texts.extend(_texts_from_batch(batch))
        else:
            texts = _texts_from_batch(inputs)

        if is_query:
            self._query_texts = texts
            self._query_facts = [analyze_query(t) for t in texts]
            logger.info("pipeline: analyzed %d queries", len(texts))
        else:
            self._document_texts = texts
            self._corpus_stats = None  # invalidate, rebuilt lazily on demand
            logger.info("pipeline: recorded %d documents", len(texts))

        if not texts:
            dimension = self.encoder.model.get_sentence_embedding_dimension()
            return np.empty((0, dimension), dtype=np.float32)

        if is_query or self.corpus_cache is None:
            return self.encoder.encode(texts, mode=mode)
        return self._encode_documents_cached(texts)

    def document_cache_key(self, text: str) -> str:
        """Content address of one document vector under the current recipe.

        Everything that changes the vector is in the key: model and revision,
        the exact prompt string handed to the tokenizer, the sequence cap, EOS
        and normalisation.  The ``document`` namespace means a query can never
        be served from this cache even if its text equals a document's.
        """
        from app.indexing.cache import compute_embedding_key

        enc = self.encoder
        prompt = (
            enc.get_prompt_text(text, "document")
            if hasattr(enc, "get_prompt_text")
            else f"{getattr(enc, 'document_prefix', '')}{text}"
        )
        return compute_embedding_key(
            encoder_input="document\x00" + prompt,
            encoder_revision=f"{getattr(enc, 'model_name', '?')}@{getattr(enc, 'revision', None)}",
            tokenizer_revision=(
                f"max{getattr(enc, 'max_seq_length', '?')}"
                f"|eos{int(bool(getattr(enc, 'add_eos_token', True)))}"
                f"|norm{int(bool(getattr(enc, 'normalize_embeddings', True)))}"
            ),
            template_revision=repr(getattr(enc, "document_prefix", "")),
        )

    def _encode_documents_cached(self, texts: list[str]) -> np.ndarray:
        keys = [self.document_cache_key(t) for t in texts]
        found = self.corpus_cache.get_many(list(dict.fromkeys(keys)))
        missing = [i for i, k in enumerate(keys) if k not in found]
        if missing:
            fresh = np.asarray(self.encoder.encode([texts[i] for i in missing], mode="document"),
                               dtype=np.float32)
            self.corpus_cache.put_many([(keys[i], fresh[row]) for row, i in enumerate(missing)])
            for row, i in enumerate(missing):
                found[keys[i]] = fresh[row]
        self.cache_stats["hits"] += len(texts) - len(missing)
        self.cache_stats["misses"] += len(missing)
        logger.info("pipeline: corpus cache %d hits, %d encoded", len(texts) - len(missing), len(missing))
        return np.vstack([found[k] for k in keys]).astype(np.float32)

    def similarity(self, embeddings1: Any, embeddings2: Any) -> Any:
        query_embeddings = np.asarray(embeddings1, dtype=np.float32)
        document_embeddings = np.asarray(embeddings2, dtype=np.float32)
        scores = self.encoder.similarity(query_embeddings, document_embeddings)
        if self.hubness_bank is not None and self.hubness_beta and document_embeddings.size:
            scores = scores - self.hubness_beta * self._hub_penalty_for(document_embeddings)[None, :]

        if not self.enable_rerank and not self.execution_config.enabled:
            return scores
        if len(self._query_facts) != scores.shape[0]:
            logger.warning(
                "pipeline: %d recorded queries for %d score rows; skipping rerank",
                len(self._query_facts),
                scores.shape[0],
            )
            return scores
        if len(self._document_texts) != scores.shape[1]:
            logger.warning(
                "pipeline: %d recorded documents for %d score columns; skipping rerank",
                len(self._document_texts),
                scores.shape[1],
            )
            return scores

        if self.enable_rerank:
            scores = self._reranker_for_current_corpus().rescore(scores, self._query_facts)
        if self.execution_config.enabled:
            scores = self._apply_execution(scores)
        return scores

    # -- internals ---------------------------------------------------------

    def _hub_penalty_for(self, documents: np.ndarray) -> np.ndarray:
        """Per-document hub score, cached for the corpus it was computed on."""
        key = (documents.shape, float(documents[:, 0].sum()), float(documents[-1].sum()))
        if self._hub_penalty is None or self._hub_penalty_key != key:
            bank = self.hubness_bank
            penalty = np.empty(documents.shape[0], dtype=np.float32)
            k = min(self.hubness_k, bank.shape[0])
            for start in range(0, documents.shape[0], 2048):
                sims = documents[start : start + 2048] @ bank.T
                penalty[start : start + 2048] = np.partition(sims, -k, axis=1)[:, -k:].mean(axis=1)
            self._hub_penalty, self._hub_penalty_key = penalty, key
            logger.info("pipeline: hubness penalty for %d documents (bank %d, k %d, beta %.2f)",
                        documents.shape[0], bank.shape[0], k, self.hubness_beta)
        return self._hub_penalty

    def _apply_execution(self, scores: np.ndarray) -> np.ndarray:
        """Third pass: promote candidates that reproduce a statement's examples.

        Only the top ``max_candidates`` of each row are executed, and a
        verified candidate is lifted above everything else in that row.
        Nothing is ever demoted.
        """
        # A corpus (or an MTEB corpus chunk) smaller than the execution depth
        # used to crash argpartition; clamp to what exists.
        top_n = min(max(self.execution_config.max_candidates, self.execution_config.deepen_to),
                    scores.shape[1])
        fired = executed = verified = 0
        out = scores.copy()
        started = time.time()
        if top_n <= 0:
            self.execution_audit = {"queries_with_examples": 0, "candidates_executed": 0,
                                    "candidates_verified": 0, "seconds": 0.0}
            return out

        # Rows are independent; running them concurrently changes nothing but
        # the wall clock, because each row's result is written back to its own
        # slot.  Within a row the pass stays sequential (workers=1), so the
        # thread count is bounded by ``workers`` rather than squared.
        row_config = replace(self.execution_config, workers=1)

        def run_row(row: int):
            query_text = self._query_texts[row]
            if not has_executable_examples(query_text, self.execution_config):
                return row, None, None, None
            order = np.argpartition(-out[row], top_n - 1)[:top_n]
            order = order[np.argsort(-out[row][order], kind="stable")]
            # A verified candidate must outrank every document in the row, not
            # just the ones it was compared against.
            bonus = float(out[row].max() - out[row][order].min()) + 1.0
            adjusted, audit = apply_execution_rerank(
                query_text,
                [self._document_texts[i] for i in order],
                [float(v) for v in out[row][order]],
                replace(row_config, verified_bonus=bonus),
            )
            return row, order, adjusted, audit

        rows = range(scores.shape[0])
        workers = max(1, int(self.execution_config.workers))
        if workers == 1:
            results = map(run_row, rows)
        else:
            from concurrent.futures import ThreadPoolExecutor

            pool = ThreadPoolExecutor(max_workers=workers)
            results = pool.map(run_row, rows)
        try:
            for row, order, adjusted, audit in results:
                if order is None:
                    continue
                out[row][order] = adjusted
                fired += 1
                executed += int(audit["candidates_executed"])
                verified += int(audit["candidates_verified"])
                if fired % 250 == 0:
                    logger.info("pipeline: executed %d queries (%.0fs)", fired, time.time() - started)
        finally:
            if workers > 1:
                pool.shutdown(wait=True)

        self.execution_audit = {
            "queries_with_examples": fired,
            "candidates_executed": executed,
            "candidates_verified": verified,
            "seconds": round(time.time() - started, 1),
        }
        logger.info("pipeline: execution pass %s", self.execution_audit)
        return out

    def _reranker_for_current_corpus(self) -> Reranker:
        if self._corpus_stats is None:
            logger.info("pipeline: analyzing %d snippets", len(self._document_texts))
            self._corpus_stats = CorpusStatistics.build(self._document_texts)
            if self.hub_profile_path is not None and self.hub_profile_path.exists():
                hubness = load_hub_profile(self.hub_profile_path, self._document_texts)
                matched = int((hubness != 0).sum())
                logger.info(
                    "pipeline: hub profile matched %d/%d documents by content",
                    matched, len(self._document_texts),
                )
                self._corpus_stats.hubness = hubness
            self._reranker = None
        if self._reranker is None:
            extractor = FeatureExtractor(
                self._corpus_stats,
                self.reranker_config.family_compat,
                length_prior_slope=self.reranker_config.length_prior_slope,
                length_prior_intercept=self.reranker_config.length_prior_intercept,
            )
            self._reranker = Reranker(self.reranker_config, extractor)
        return self._reranker

    # -- offline use -------------------------------------------------------

    def rescore_matrix(
        self,
        scores: np.ndarray,
        query_texts: Sequence[str],
        document_texts: Sequence[str],
    ) -> np.ndarray:
        """Apply the second pass to a precomputed score matrix.

        Used by the offline harness, where dense vectors are cached and only
        the reranker changes between runs.
        """
        self._query_texts = list(query_texts)
        self._query_facts = [analyze_query(t) for t in query_texts]
        self._document_texts = list(document_texts)
        self._corpus_stats = None
        if not self.enable_rerank:
            return scores
        return self._reranker_for_current_corpus().rescore(scores, self._query_facts)
