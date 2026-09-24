from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
import torch
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "retrieval.yaml"
FROZEN_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "frozen.yaml"

#: The published F2LLM-v2 retrieval recipe, reproduced verbatim from
#: ``mteb/models/model_implementations/codefuse_models.py``.  These four facts
#: are the difference between the leaderboard score and a silently degraded
#: one, so they are constants in the source, not defaults scattered in YAML:
#:
#:   * queries carry an instruction, documents carry none;
#:   * the instruction is the task-specific one, not the generic QA prompt;
#:   * ``add_eos_token=True`` -- pooling is *last-token*, so a missing EOS
#:     pools an arbitrary interior token;
#:   * ``max_seq_length=8192`` -- and for the same reason, truncation is not a
#:     graceful degradation here, it is a wrong vector.
F2LLM_INSTRUCTION_TEMPLATE = "Instruct: {instruction}\nQuery: "
APPS_INSTRUCTION = "Retrieve the most relevant code snippet for the given query."
APPS_QUERY_PREFIX = F2LLM_INSTRUCTION_TEMPLATE.format(instruction=APPS_INSTRUCTION)
PUBLISHED_MAX_SEQ_LENGTH = 8192

#: Attention memory scales as ``batch * seq^2``, not ``batch * seq``.  A batch
#: size tuned on 100-token snippets therefore asks for *tens of gigabytes* on
#: an 8192-token one: encoding this corpus with a fixed batch size died twice
#: on CUDA OOM (32 GiB, then 12 GiB) before the budget was made quadratic.
#:
#: The unit is "padded attention cells": ``max_len_in_batch ** 2 * batch``.
#: 6.7e7 admits one 8192-token sequence alone (~2 GiB of fp16 attention across
#: 16 heads) and 64 sequences of 1024 tokens.
DEFAULT_ATTENTION_BUDGET = 67_000_000
DEFAULT_MAX_BATCH = 64


def _select_device(preferred: str | None = None) -> str:
    """Select compute device. Defaults to CPU unless 'auto' finds a GPU."""
    if preferred and preferred != "auto":
        return preferred
    if preferred == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def select_dtype(device: str) -> str:
    """Pick a numerically sane *and* fast dtype for the device.

    Turing GPUs (T4, compute capability 7.5) have no native bfloat16: torch
    emulates it and the encode slows by roughly an order of magnitude — the
    first GPU run of this corpus took over an hour before the dtype was fixed.
    Ampere and newer get bfloat16, which is what the model was trained in.
    CPU stays float32, where half precision is slower, not faster.
    """
    if device != "cuda" or not torch.cuda.is_available():
        return "float32"
    major, _ = torch.cuda.get_device_capability(0)
    return "bfloat16" if major >= 8 else "float16"


_LOADED_MODELS: dict[tuple[str, str, str], Any] = {}


def _keep_sdpa_off_the_math_kernel() -> bool:
    """Stop grouped-query attention from forcing SDPA onto the math kernel.

    F2LLM-v2 (Qwen3) uses grouped-query attention: 16 query heads, 8 key/value
    heads.  For an *unpadded* batch -- which is exactly what ``plan_batches``
    produces for a long document, isolated in a batch of one -- transformers
    hands the unequal head counts to SDPA with ``enable_gqa=True``.  Only the
    flash kernel accepts that; the memory-efficient kernel rejects it ("both
    fused kernels require query, key and value to have the same num_heads").
    Torch builds without flash attention (the Windows wheels) then fall back to
    the math kernel, which materialises a 16 x L x L score matrix: 4 GiB for one
    8k-token document, which is what killed the encode on a 4 GB GPU twice.

    Expanding key/value heads first (``repeat_kv``, what transformers already
    does whenever a padding mask is present) is the same attention, computed by
    the memory-efficient kernel.  Applied only when this build has no flash
    attention; where flash exists (the Linux T4 the scored run used) nothing
    changes.  Returns whether the patch was applied.
    """
    backends = getattr(torch.backends, "cuda", None)
    has_flash = getattr(backends, "is_flash_attention_available", lambda: True)()
    if has_flash:
        return False
    try:
        from transformers.integrations import sdpa_attention
    except ImportError:  # pragma: no cover - transformers without the integration
        return False
    if getattr(sdpa_attention.use_gqa_in_sdpa, "_codedoctor_patched", False):
        return True

    def _never_gqa(attention_mask, key):  # noqa: ARG001 - signature mirrors transformers
        return False

    _never_gqa._codedoctor_patched = True
    sdpa_attention.use_gqa_in_sdpa = _never_gqa
    return True


def plan_batches(
    lengths: Sequence[int],
    *,
    attention_budget: int = DEFAULT_ATTENTION_BUDGET,
    max_batch: int = DEFAULT_MAX_BATCH,
) -> list[list[int]]:
    """Group indices into length-sorted batches under a quadratic memory budget.

    A batch costs ``max_len_in_batch ** 2 * batch_size`` attention cells, so
    that is what is budgeted.  Returns batches of *original* indices; sorting
    by length also removes padding waste, which is where most of the speed-up
    comes from on a corpus that averages 225 tokens and reaches 289k.

    A single input that exceeds the budget on its own still gets a batch of
    one rather than being dropped.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for index in order:
        candidate_max = max(current_max, lengths[index])
        cost = candidate_max * candidate_max * (len(current) + 1)
        if current and (cost > attention_budget or len(current) >= max_batch):
            batches.append(current)
            current, current_max = [index], lengths[index]
        else:
            current.append(index)
            current_max = candidate_max
    if current:
        batches.append(current)
    return batches


class CodeEncoder:
    """Prefix-aware code encoder wrapping the frozen production model.

    Query mode applies the task instruction; document mode applies none.  Both
    modes share one sequence-length cap, because the encoder input *is* the
    cache key and a mode-dependent cap would make the same text hash to two
    different vectors.
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
        revision: str | None = None,
        torch_dtype: str | None = None,
        tier: str = "interactive",
    ) -> None:
        if config_path is not None:
            self.config_path = config_path
        elif FROZEN_CONFIG_PATH.exists():
            self.config_path = FROZEN_CONFIG_PATH
        else:
            self.config_path = DEFAULT_CONFIG_PATH
        if tier not in ("accuracy", "interactive"):
            raise ValueError(f"tier must be 'accuracy' or 'interactive', got {tier!r}")
        self.tier = tier
        config = self._load_config()

        self.model_name = (
            model_name
            or config.get("production_model")
            or config.get("model_name", "codefuse-ai/F2LLM-v2-0.6B")
        )
        self.revision = revision if revision is not None else config.get("revision")
        self.query_prefix = (
            query_prefix if query_prefix is not None else config.get("query_prefix", APPS_QUERY_PREFIX)
        )
        self.document_prefix = (
            document_prefix if document_prefix is not None else config.get("document_prefix", "")
        )

        cap = max_seq_length or config.get("max_seq_length", PUBLISHED_MAX_SEQ_LENGTH)
        # Per-mode caps are accepted for backwards compatibility but are
        # collapsed to one value: see the class docstring.
        self.max_seq_length = int(cap)
        self.query_max_seq_length = int(query_max_seq_length or config.get("query_max_seq_length") or cap)
        self.document_max_seq_length = int(
            document_max_seq_length or config.get("document_max_seq_length") or cap
        )

        self.similarity_metric = similarity_metric or config.get("similarity_metric", "cosine")
        self.device = device or config.get("device", "cpu")
        self.normalize_embeddings = bool(config.get("normalize_embeddings", True))
        self.add_eos_token = bool(config.get("add_eos_token", True))
        self.torch_dtype = (
            torch_dtype or config.get("torch_dtype") or select_dtype(self.device)
        )
        self.attention_budget = int(config.get("attention_budget", DEFAULT_ATTENTION_BUDGET))
        self.max_batch = int(config.get("max_batch", DEFAULT_MAX_BATCH))

        self._model = None

    def _load_config(self) -> dict:
        """Merge the base ``encoder`` block with the requested tier's overrides.

        ``accuracy`` is the submission encoder; ``interactive`` is the smaller
        one the live service and the hands-on demo run.  Both share every other
        recipe value, so a tier can only change *which* model, never *how* it
        is prompted or truncated.
        """
        if not self.config_path.exists():
            return {}
        try:
            content = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
        base = dict(content.get("encoder", {}) or {})
        if self.tier == "interactive":
            base.update(content.get("interactive_encoder", {}) or {})
        return base

    @property
    def model(self):
        if self._model is None:
            cache_key = (self.model_name, self.device, self.torch_dtype)
            cached = _LOADED_MODELS.get(cache_key)
            if cached is not None:
                self._model = cached
                return self._model

            from sentence_transformers import SentenceTransformer

            if str(self.device).startswith("cuda"):
                _keep_sdpa_off_the_math_kernel()
            torch.manual_seed(42)
            model_kwargs: dict[str, Any] = {
                "torch_dtype": getattr(torch, self.torch_dtype),
                "attn_implementation": "sdpa",
            }
            load_kwargs: dict[str, Any] = {
                "device": self.device,
                "model_kwargs": model_kwargs,
                "tokenizer_kwargs": {
                    "add_eos_token": self.add_eos_token,
                    "model_max_length": self.max_seq_length,
                },
            }
            if self.revision:
                load_kwargs["revision"] = self.revision

            model = SentenceTransformer(self.model_name, **load_kwargs)
            model.max_seq_length = self.max_seq_length
            model.eval()
            self._verify_recipe_applied(model)
            _LOADED_MODELS[cache_key] = model
            self._model = model
        return self._model

    def _verify_recipe_applied(self, model: Any) -> None:
        """Assert the loaded model really got the recipe it was asked for.

        ``tokenizer_kwargs`` is deprecated in sentence-transformers 6 in favour
        of ``processor_kwargs``.  If a future release stops honouring it, the
        EOS token silently stops being appended — and because pooling is
        last-token, every embedding quietly becomes wrong while the run still
        completes and reports a plausible-looking score.  That is the exact
        failure this project already paid ~3 NDCG points for once, so it is
        checked rather than trusted.
        """
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:  # pragma: no cover - non-transformer backends
            return

        if int(getattr(model, "max_seq_length", 0)) != self.max_seq_length:
            raise RuntimeError(
                f"max_seq_length is {model.max_seq_length}, expected {self.max_seq_length}. "
                "Truncation under last-token pooling produces wrong vectors, not degraded ones."
            )

        if not self.add_eos_token:
            return
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None:  # pragma: no cover - tokenizer without an EOS
            return
        probe = tokenizer.encode("def f(): return 1", add_special_tokens=True)
        if not probe or probe[-1] != eos_id:
            # Try to repair before giving up: some loaders accept the flag only
            # as a tokenizer attribute.
            setattr(tokenizer, "add_eos_token", True)
            probe = tokenizer.encode("def f(): return 1", add_special_tokens=True)
        if not probe or probe[-1] != eos_id:
            raise RuntimeError(
                f"{self.model_name} was loaded without an appended EOS token "
                f"(last id {probe[-1] if probe else None}, expected {eos_id}). "
                "Pooling is last-token, so every embedding would be taken from an "
                "arbitrary interior token. Refusing to encode."
            )

    # -- prompt handling ---------------------------------------------------

    def get_prompt_text(self, text: str, mode: str = "document") -> str:
        """The exact string handed to the tokenizer. This is the cache key."""
        if mode == "query":
            return f"{self.query_prefix}{text}"
        return f"{self.document_prefix}{text}"

    # -- encoding ----------------------------------------------------------

    def _token_lengths(self, texts: Sequence[str], mode: str) -> list[int]:
        tokenizer = self.model.tokenizer
        prefix = self.query_prefix if mode == "query" else self.document_prefix
        prefix_len = len(tokenizer.encode(prefix, add_special_tokens=False)) if prefix else 0
        cap = self.max_seq_length
        return [
            min(cap, prefix_len + len(tokenizer.encode(t, add_special_tokens=False)) + 1)
            for t in texts
        ]

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        mode: str = "document",
        batch_size: int | None = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode text(s) into a float32 array.

        ``mode`` selects the prefix; batching is length-aware so a single long
        outlier cannot blow up memory for the whole batch.  ``batch_size`` is
        accepted for API compatibility and, when given, caps the batch count.
        """
        single = isinstance(texts, str)
        text_list = [texts] if single else list(texts)
        if not text_list:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)

        prompt = self.query_prefix if mode == "query" else self.document_prefix
        max_batch = min(self.max_batch, batch_size) if batch_size else self.max_batch
        batches = plan_batches(
            self._token_lengths(text_list, mode),
            attention_budget=self.attention_budget,
            max_batch=max_batch,
        )

        out: list[np.ndarray | None] = [None] * len(text_list)
        original_threads = torch.get_num_threads()
        progress_every = max(1, len(batches) // 20) if show_progress_bar else 0
        started = time.time()
        done_texts = 0
        try:
            if self.device == "cpu":
                torch.set_num_threads(os.cpu_count() or 1)
            torch.manual_seed(42)
            with torch.inference_mode():
                for batch_number, batch in enumerate(batches, 1):
                    done_texts += len(batch)
                    if progress_every and (batch_number % progress_every == 0 or batch_number == len(batches)):
                        peak = ""
                        if str(self.device).startswith("cuda") and torch.cuda.is_available():
                            peak = f", peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
                        print(
                            f"    batch {batch_number}/{len(batches)}: {done_texts}/{len(text_list)} texts, "
                            f"{time.time() - started:.0f}s{peak}",
                            flush=True,
                        )
                    embeddings = self.model.encode(
                        [text_list[i] for i in batch],
                        prompt=prompt or None,
                        batch_size=len(batch),
                        show_progress_bar=False,
                        normalize_embeddings=self.normalize_embeddings,
                        convert_to_numpy=True,
                    )
                    for slot, vector in zip(batch, embeddings):
                        out[slot] = vector
        finally:
            if self.device == "cpu":
                torch.set_num_threads(original_threads)

        stacked = np.asarray(np.vstack(out), dtype=np.float32)
        return stacked[0] if single else stacked

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """Similarity under the configured metric."""
        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)

        if self.similarity_metric == "cosine":
            if not self.normalize_embeddings:
                norm1 = np.linalg.norm(emb1, axis=1, keepdims=True)
                norm2 = np.linalg.norm(emb2, axis=1, keepdims=True)
                norm1[norm1 == 0] = 1e-12
                norm2[norm2 == 0] = 1e-12
                emb1 = emb1 / norm1
                emb2 = emb2 / norm2
            return np.dot(emb1, emb2.T)
        if self.similarity_metric == "dot":
            return np.dot(emb1, emb2.T)
        raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")


_GLOBAL_ENCODER: CodeEncoder | None = None


def get_shared_encoder(tier: str = "interactive") -> CodeEncoder:
    """The process-wide encoder for the live service.

    Defaults to the interactive tier: the developer UI answers queries in a
    browser, so it runs the smaller model.  The submission path asks for
    ``tier="accuracy"`` explicitly.
    """
    global _GLOBAL_ENCODER
    if _GLOBAL_ENCODER is None or _GLOBAL_ENCODER.tier != tier:
        _GLOBAL_ENCODER = CodeEncoder(device="cpu", tier=tier)
    return _GLOBAL_ENCODER
