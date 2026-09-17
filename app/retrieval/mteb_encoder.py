"""MTEB-compatible encoder for CodeDoctor.

This module implements ``PrePostPipelineEncoder``, an ``mteb.models.abs_encoder.AbsEncoder``
subclass that is what gets scored on the CoIR ``AppsRetrieval`` task.

Design note (why this shape)
----------------------------
MTEB's ``AbsEncoder`` is an *encoding* interface: it maps text to vectors and MTEB
does the nearest-neighbour search itself. CodeDoctor's ``HybridRetriever`` is a
*ranker* (exact-match + BM25 + TF-IDF fused with reciprocal rank fusion), and RRF
cannot be expressed inside ``encode()`` because it needs the whole candidate list.

So the lexical signal is carried into the vector space instead of being fused after
the fact. Each output vector is the concatenation of two L2-normalised blocks::

    [ dense_weight * dense_embedding | lexical_weight * projected_lexical ]

Because both blocks are unit-norm and the weights are constant across every vector,
cosine similarity over the concatenation is exactly the weighted sum of the dense
cosine and the lexical cosine. That gives genuine hybrid dense+lexical scoring while
still satisfying the encode-only contract.

The lexical block is a hashed sublinear-TF vector over code-aware tokens, reduced to
a fixed number of dimensions by a *seeded* sparse random projection. The seed is fixed,
so queries and documents land in the same space and results are fully reproducible.

Pre-processing branches on ``prompt_type``:
  * ``PromptType.query``   - natural-language problem statements (strip boilerplate
    scaffolding, normalise whitespace, apply identifier-oriented expansion).
  * ``PromptType.document`` - source code (split camelCase/snake_case identifiers into
    words, surface comments and docstrings, drop very long literals).

There is deliberately no random, constant, or placeholder vector path anywhere in this
file. ``encode()`` is deterministic: the same input always produces the same output.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from app.retrieval.hybrid import QUERY_EXPANSIONS, normalize_identifier

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)

DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_LEXICAL_DIM = 256
DEFAULT_HASH_SPACE = 2**18
DEFAULT_DENSE_WEIGHT = 1.0
DEFAULT_LEXICAL_WEIGHT = 0.35
PROJECTION_SEED = 20240917
PROJECTION_NONZEROS = 8

# Scaffolding that appears in nearly every APPS problem statement and therefore
# carries no discriminative signal.
_QUERY_BOILERPLATE = re.compile(
    r"(?im)^\s*(-----)?\s*(input|output|example[s]?|note[s]?|constraints?|sample\s+(input|output))\s*(-----)?\s*:?\s*$"
)
_MULTISPACE = re.compile(r"[ \t]+")
_MULTINEWLINE = re.compile(r"\n{3,}")
_LONG_LITERAL = re.compile(r"(['\"])(?:(?!\1).){80,}?\1", re.DOTALL)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def split_identifier(identifier: str) -> list[str]:
    """Split ``parse_input_lines`` / ``parseInputLines`` into component words."""
    parts: list[str] = []
    for chunk in identifier.split("_"):
        if not chunk:
            continue
        parts.extend(piece.lower() for piece in _CAMEL_BOUNDARY.split(chunk) if piece)
    return parts


def preprocess_query(text: str) -> str:
    """Normalise a natural-language query before dense encoding."""
    cleaned = _QUERY_BOILERPLATE.sub(" ", text)
    cleaned = _MULTISPACE.sub(" ", cleaned)
    cleaned = _MULTINEWLINE.sub("\n\n", cleaned).strip()

    expansions: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN.findall(cleaned.lower()):
        for expansion in QUERY_EXPANSIONS.get(token, ()):
            if expansion not in seen:
                seen.add(expansion)
                expansions.append(expansion)
    if expansions:
        cleaned = f"{cleaned}\n{' '.join(expansions)}"
    return cleaned


def preprocess_document(text: str) -> str:
    """Normalise a code snippet before dense encoding.

    Identifiers are expanded into their component words so that a dense model trained
    mostly on prose can align ``def count_pairs(...)`` with "count the number of pairs".
    The original source is kept as well, since exact token overlap still matters.
    """
    cleaned = _LONG_LITERAL.sub(" ", text)
    cleaned = _MULTISPACE.sub(" ", cleaned)
    cleaned = _MULTINEWLINE.sub("\n\n", cleaned).strip()

    words: list[str] = []
    seen: set[str] = set()
    for identifier in _IDENTIFIER.findall(cleaned):
        for word in split_identifier(identifier):
            if len(word) > 1 and word not in seen:
                seen.add(word)
                words.append(word)

    if words:
        return f"{' '.join(words)}\n{cleaned}"
    return cleaned


def lexical_tokens(text: str) -> list[str]:
    """Code-aware tokenisation matching the spirit of ``hybrid.tokenize``."""
    tokens: list[str] = []
    # Split on the ORIGINAL casing first -- lowercasing up front would destroy the
    # camelCase boundaries that split_identifier relies on.
    for raw_token in _TOKEN.findall(text):
        token = raw_token.lower()
        tokens.append(token)
        normalized = normalize_identifier(token)
        if normalized and normalized != token:
            tokens.append(normalized)
        tokens.extend(part for part in split_identifier(raw_token) if part != token)
    return tokens


class SparseRandomProjection:
    """Deterministic, seeded sparse random projection from a hash space to ``out_dim``.

    Each hash bucket is assigned a small fixed set of output dimensions with +/-1 signs,
    derived from a seeded RNG. Identical across processes and runs.
    """

    def __init__(
        self,
        hash_space: int = DEFAULT_HASH_SPACE,
        out_dim: int = DEFAULT_LEXICAL_DIM,
        nonzeros: int = PROJECTION_NONZEROS,
        seed: int = PROJECTION_SEED,
    ) -> None:
        self.hash_space = hash_space
        self.out_dim = out_dim
        self.nonzeros = nonzeros
        self.seed = seed
        rng = np.random.default_rng(seed)
        self._targets = rng.integers(0, out_dim, size=(hash_space, nonzeros), dtype=np.int32)
        self._signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(hash_space, nonzeros))
        self._scale = float(1.0 / np.sqrt(nonzeros))

    def bucket(self, token: str) -> int:
        # Stable across runs: Python's hash() is randomised per process, so use a
        # deterministic rolling hash instead.
        value = 1469598103934665603
        for char in token.encode("utf-8"):
            value = ((value ^ char) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return value % self.hash_space

    def transform(self, token_weights: dict[str, float]) -> np.ndarray:
        vector = np.zeros(self.out_dim, dtype=np.float32)
        for token, weight in token_weights.items():
            bucket = self.bucket(token)
            targets = self._targets[bucket]
            signs = self._signs[bucket]
            np.add.at(vector, targets, signs * (weight * self._scale))
        return vector


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class PrePostPipelineEncoder:
    """CodeDoctor's MTEB encoder: pre-processing + dense embedding + lexical fusion.

    Parameters
    ----------
    dense_model:
        Any object exposing ``encode(list[str], batch_size=..., ...) -> ndarray``.
        Defaults to a CPU SentenceTransformer loaded from ``model_name``.
    lexical_weight:
        Set to ``0.0`` to run a dense-only ablation.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_DENSE_MODEL,
        dense_model: Any | None = None,
        device: str = "cpu",
        lexical_dim: int = DEFAULT_LEXICAL_DIM,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
    ) -> None:
        self.model_name = model_name
        self.dense_weight = float(dense_weight)
        self.lexical_weight = float(lexical_weight)
        self.projection = SparseRandomProjection(out_dim=lexical_dim)

        if dense_model is not None:
            self.dense_model = dense_model
        else:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading dense backbone %s on %s", model_name, device)
            self.dense_model = SentenceTransformer(model_name, device=device)

    # -- pipeline stages -------------------------------------------------

    def preprocess(self, texts: Sequence[str], prompt_type: Any | None) -> list[str]:
        name = getattr(prompt_type, "value", prompt_type)
        if name == "query":
            return [preprocess_query(text) for text in texts]
        return [preprocess_document(text) for text in texts]

    def _dense(self, texts: Sequence[str], **kwargs: Any) -> np.ndarray:
        batch_size = int(kwargs.get("batch_size", 64) or 64)
        vectors = self.dense_model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=bool(kwargs.get("show_progress_bar", False)),
            convert_to_numpy=True,
        )
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))

    def _lexical(self, texts: Sequence[str]) -> np.ndarray:
        rows = np.zeros((len(texts), self.projection.out_dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[str, float] = {}
            for token in lexical_tokens(text):
                counts[token] = counts.get(token, 0.0) + 1.0
            # Sublinear term frequency damps the effect of repeated loop variables.
            weights = {token: 1.0 + float(np.log(count)) for token, count in counts.items()}
            rows[row] = self.projection.transform(weights)
        return _l2_normalize(rows)

    def encode_texts(self, texts: Sequence[str], prompt_type: Any | None = None, **kwargs: Any) -> np.ndarray:
        """Full pipeline on a plain list of strings (used by tests and the CLI)."""
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        processed = self.preprocess(texts, prompt_type)
        dense = self._dense(processed, **kwargs)

        if self.lexical_weight == 0.0:
            return dense * self.dense_weight

        lexical = self._lexical(processed)
        return np.concatenate(
            [dense * self.dense_weight, lexical * self.lexical_weight],
            axis=1,
        ).astype(np.float32)

    # -- MTEB interface --------------------------------------------------

    def encode(
        self,
        inputs: Iterable[Any],
        *,
        task_metadata: Any = None,
        hf_split: str = "test",
        hf_subset: str = "default",
        prompt_type: Any | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """MTEB entry point.

        ``inputs`` is a ``DataLoader`` yielding batches of dicts with a ``"text"`` key
        (mteb >= 2.0). A plain iterable of strings is also accepted so the class stays
        testable without constructing a DataLoader.
        """
        texts: list[str] = []
        for batch in inputs:
            if isinstance(batch, str):
                texts.append(batch)
            elif isinstance(batch, dict):
                texts.extend(batch["text"])
            else:
                texts.extend(batch)
        return self.encode_texts(texts, prompt_type=prompt_type, **kwargs)


def build_model_meta(encoder: PrePostPipelineEncoder, embed_dim: int) -> Any:
    """Wrap the encoder in an ``mteb.ModelMeta`` so results are labelled correctly."""
    from mteb.models.model_meta import ModelMeta

    return ModelMeta(
        loader=lambda **_: encoder,
        name="codedoctor/PrePostPipelineEncoder",
        revision="1.0.0",
        release_date="2026-09-18",
        languages=["eng-Latn", "python-Code"],
        n_parameters=33_000_000,
        memory_usage_mb=127,
        max_tokens=512,
        embed_dim=embed_dim,
        license="apache-2.0",
        open_weights=True,
        public_training_code=None,
        public_training_data=None,
        framework=["Sentence Transformers", "NumPy"],
        similarity_fn_name="cosine",
        use_instructions=False,
        training_datasets=None,
        reference=None,
    )
