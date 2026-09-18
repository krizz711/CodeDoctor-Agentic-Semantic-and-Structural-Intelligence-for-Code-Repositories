from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence, Union
import numpy as np
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "retrieval.yaml"
FROZEN_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "frozen.yaml"


import os
import torch


def _select_device(preferred: str | None = None) -> str:
    """Select compute device. Defaults to GPU if present only when preferred='auto', else CPU."""
    if preferred and preferred != "auto":
        return preferred
    if preferred == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


_LOADED_MODELS: dict[tuple[str, str], Any] = {}


class CodeEncoder:
    """
    Prefix-aware code encoder wrapper.
    Enforces separate query-mode and document-mode instruction prefixes.
    Ensures deterministic, reproducible embeddings.
    Defaults to CPU for verified offline/service parity.
    """

    def __init__(
        self,
        model_name: str | None = None,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        similarity_metric: str = "cosine",
        device: str = "cpu",
        config_path: Path | None = None,
        max_seq_length: int | None = None,
        query_max_seq_length: int | None = None,
        document_max_seq_length: int | None = None,
    ) -> None:
        if config_path is not None:
            self.config_path = config_path
        elif FROZEN_CONFIG_PATH.exists():
            self.config_path = FROZEN_CONFIG_PATH
        else:
            self.config_path = DEFAULT_CONFIG_PATH
        config = self._load_config()

        self.model_name = (
            model_name
            or config.get("production_model")
            or config.get("model_name", "codefuse-ai/F2LLM-v2-0.6B")
        )
        self.query_prefix = query_prefix if query_prefix is not None else config.get(
            "query_prefix", "Instruct: Given a coding task, retrieve relevant code snippet\nQuery: "
        )
        self.document_prefix = document_prefix if document_prefix is not None else config.get("document_prefix", "")
        self.max_seq_length = max_seq_length or config.get("max_seq_length", 512)
        self.query_max_seq_length = (
            query_max_seq_length
            or config.get("query_max_seq_length")
            or 1024
        )
        self.document_max_seq_length = (
            document_max_seq_length
            or config.get("document_max_seq_length")
            or self.max_seq_length
        )
        self.similarity_metric = similarity_metric or config.get("similarity_metric", "cosine")
        self.device = device or config.get("device", "cpu")
        self.normalize_embeddings = config.get("normalize_embeddings", True)

        self._model = None

    def _load_config(self) -> dict:
        if self.config_path.exists():
            try:
                content = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
                return content.get("encoder", {})
            except Exception:
                pass
        return {}

    @property
    def model(self):
        if self._model is None:
            cache_key = (self.model_name, self.device)
            if cache_key in _LOADED_MODELS:
                self._model = _LOADED_MODELS[cache_key]
            else:
                from sentence_transformers import SentenceTransformer

                torch.manual_seed(42)
                model_kwargs: dict[str, Any] = {}
                if self.device == "cpu":
                    model_kwargs["torch_dtype"] = torch.float32
                elif self.device == "cuda":
                    model_kwargs["torch_dtype"] = torch.float16

                m = SentenceTransformer(self.model_name, device=self.device, model_kwargs=model_kwargs)
                m.eval()
                if hasattr(m, "max_seq_length"):
                    m.max_seq_length = self.max_seq_length
                if len(m) > 0 and hasattr(m[0], "max_seq_length"):
                    m[0].max_seq_length = self.max_seq_length
                if hasattr(m, "tokenizer") and hasattr(m.tokenizer, "model_max_length"):
                    m.tokenizer.model_max_length = self.max_seq_length
                _LOADED_MODELS[cache_key] = m
                self._model = m
        return self._model

    def get_prompt_text(self, text: str, mode: str = "document") -> str:
        if mode == "query":
            return f"{self.query_prefix}{text}"
        return f"{self.document_prefix}{text}"

    def inspect_encoding_input(self, text: str, mode: str = "document") -> dict[str, Any]:
        """Inspection helper returning the exact prompt, prefix, token length, and cap for verification."""
        prefix = self.query_prefix if mode == "query" else self.document_prefix
        prompt_text = self.get_prompt_text(text, mode=mode)
        cap = self.query_max_seq_length if mode == "query" else self.document_max_seq_length
        toks = []
        if hasattr(self.model, "tokenizer"):
            toks = self.model.tokenizer.encode(prompt_text, truncation=False)
        return {
            "mode": mode,
            "prefix": prefix,
            "has_prefix": bool(prefix),
            "raw_text_sample": text[:120],
            "final_input_sample": prompt_text[:160],
            "token_count": len(toks),
            "max_seq_length_cap": cap,
            "is_truncated": len(toks) > cap,
        }

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        mode: str = "document",
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Encodes text(s) into numpy float32 array.
        mode: 'query' or 'document' (applies respective prefix and sequence length cap).
        """
        single = isinstance(texts, str)
        text_list = [texts] if single else list(texts)

        # Apply appropriate sequence length cap per mode
        target_seq_len = self.query_max_seq_length if mode == "query" else self.document_max_seq_length
        if hasattr(self.model, "max_seq_length"):
            self.model.max_seq_length = target_seq_len
        if len(self.model) > 0 and hasattr(self.model[0], "max_seq_length"):
            self.model[0].max_seq_length = target_seq_len
        if hasattr(self.model, "tokenizer") and hasattr(self.model.tokenizer, "model_max_length"):
            self.model.tokenizer.model_max_length = target_seq_len

        # Apply appropriate instruction prefix
        prefixed = [self.get_prompt_text(t, mode=mode) for t in text_list]

        # sentence-transformers encode
        torch.manual_seed(42)
        orig_threads = torch.get_num_threads()
        try:
            if self.device == "cpu":
                if single or len(text_list) <= 16:
                    torch.set_num_threads(1)
                else:
                    torch.set_num_threads(os.cpu_count() or 1)
            with torch.inference_mode():
                embeddings = self.model.encode(
                    prefixed,
                    batch_size=batch_size,
                    show_progress_bar=show_progress_bar,
                    normalize_embeddings=self.normalize_embeddings,
                    convert_to_numpy=True,
                )
        finally:
            if self.device == "cpu":
                torch.set_num_threads(orig_threads)

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if single:
            return embeddings[0]
        return embeddings

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """
        Compute similarity score between embeddings.
        Asserts metric matches model configuration.
        """
        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)

        if self.similarity_metric == "cosine":
            # If already unit-normalized, dot product equals cosine similarity
            if not self.normalize_embeddings:
                norm1 = np.linalg.norm(emb1, axis=1, keepdims=True)
                norm2 = np.linalg.norm(emb2, axis=1, keepdims=True)
                norm1[norm1 == 0] = 1e-12
                norm2[norm2 == 0] = 1e-12
                emb1 = emb1 / norm1
                emb2 = emb2 / norm2
            return np.dot(emb1, emb2.T)
        elif self.similarity_metric == "dot":
            return np.dot(emb1, emb2.T)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")


# Global shared instance for parity between harness and service
_GLOBAL_ENCODER: CodeEncoder | None = None


def get_shared_encoder() -> CodeEncoder:
    global _GLOBAL_ENCODER
    if _GLOBAL_ENCODER is None:
        _GLOBAL_ENCODER = CodeEncoder(device="cpu")
    return _GLOBAL_ENCODER
