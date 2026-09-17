"""Optional CPU-first embedding retriever using fastembed + hnswlib.

This module provides semantic vector search as a complement to the
lexical BM25/TF-IDF retrieval in hybrid.py. It uses:
- fastembed (ONNX Runtime) for encoding (no PyTorch dependency)
- hnswlib for approximate nearest neighbor search (incremental updates)

Install with: pip install -r requirements-embeddings.txt
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

_FASTEMBED_AVAILABLE = False
_HNSWLIB_AVAILABLE = False

try:
    from fastembed import TextEmbedding

    _FASTEMBED_AVAILABLE = True
except ImportError:
    pass

try:
    import hnswlib
    import numpy as np

    _HNSWLIB_AVAILABLE = True
except ImportError:
    pass


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIM = 384
MAX_ELEMENTS = 100_000


class EmbeddingRetriever:
    """CPU-first semantic code search using vector embeddings.

    Uses BAAI/bge-small-en-v1.5 (384 dimensions, 33M params) via ONNX Runtime
    for fast CPU inference. Supports incremental index updates via hnswlib's
    HNSW graph structure.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        dim: int = DEFAULT_DIM,
        max_elements: int = MAX_ELEMENTS,
    ) -> None:
        if not _FASTEMBED_AVAILABLE:
            raise ImportError("fastembed is required for embedding retrieval. Install with: pip install fastembed")
        if not _HNSWLIB_AVAILABLE:
            raise ImportError("hnswlib is required for embedding retrieval. Install with: pip install hnswlib")

        self.model_name = model_name
        self.dim = dim
        self.max_elements = max_elements

        logger.info("Loading embedding model: %s", model_name)
        self._model = TextEmbedding(model_name=model_name)

        # Initialize HNSW index
        self._index = hnswlib.Index(space="cosine", dim=dim)
        self._index.init_index(
            max_elements=max_elements,
            ef_construction=200,
            M=16,
        )
        self._index.set_ef(50)  # query time accuracy

        # ID mapping: hnswlib integer ID -> document metadata
        self._id_map: dict[int, dict[str, Any]] = {}
        self._next_id: int = 0

    def add_documents(
        self,
        documents: Sequence[dict[str, Any]],
    ) -> int:
        """Add documents to the embedding index.

        Each document dict should have at minimum:
        - 'content': str — the text to embed
        - 'document_id': str — unique identifier

        Additional fields (file_path, name, type, start_line, end_line)
        are stored as metadata for retrieval results.

        Returns the number of documents added.
        """
        if not documents:
            return 0

        texts = [doc.get("content", "") for doc in documents]
        embeddings = list(self._model.embed(texts))
        vectors = np.array(embeddings, dtype=np.float32)

        # Resize index if needed
        current_max = self._index.get_max_elements()
        needed = self._next_id + len(documents)
        if needed > current_max:
            self._index.resize_index(max(needed * 2, current_max * 2))

        ids = list(range(self._next_id, self._next_id + len(documents)))
        self._index.add_items(vectors, ids)

        for i, doc in zip(ids, documents, strict=False):
            self._id_map[i] = doc

        self._next_id += len(documents)
        logger.info("Added %d documents to embedding index (total: %d)", len(documents), self._next_id)
        return len(documents)

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for semantically similar documents.

        Returns a list of document dicts with an added 'embedding_score' field
        (cosine similarity, higher is better).
        """
        if self._next_id == 0:
            return []

        query_embedding = list(self._model.embed([query]))
        query_vector = np.array(query_embedding, dtype=np.float32)

        k = min(limit, self._next_id)
        labels, distances = self._index.knn_query(query_vector, k=k)

        results = []
        for label, distance in zip(labels[0], distances[0], strict=False):
            doc = self._id_map.get(int(label), {})
            result = dict(doc)
            # hnswlib cosine distance = 1 - cosine_similarity
            result["embedding_score"] = float(1.0 - distance)
            results.append(result)

        return results

    def save(self, path: str | Path) -> None:
        """Save the HNSW index and metadata to disk."""
        import json

        path = Path(path)
        index_path = path.with_suffix(".hnsw")
        meta_path = path.with_suffix(".meta.json")

        self._index.save_index(str(index_path))

        metadata = {
            "model_name": self.model_name,
            "dim": self.dim,
            "next_id": self._next_id,
            "id_map": {str(k): v for k, v in self._id_map.items()},
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info("Saved embedding index to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingRetriever:
        """Load a previously saved embedding index."""
        import json

        path = Path(path)
        index_path = path.with_suffix(".hnsw")
        meta_path = path.with_suffix(".meta.json")

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        instance = cls(
            model_name=metadata["model_name"],
            dim=metadata["dim"],
        )
        instance._index.load_index(str(index_path))
        instance._next_id = metadata["next_id"]
        instance._id_map = {int(k): v for k, v in metadata["id_map"].items()}

        logger.info("Loaded embedding index from %s (%d documents)", path, instance._next_id)
        return instance

    @property
    def document_count(self) -> int:
        """Number of documents in the index."""
        return self._next_id


def is_available() -> bool:
    """Check if embedding dependencies are installed."""
    return _FASTEMBED_AVAILABLE and _HNSWLIB_AVAILABLE
