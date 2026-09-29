"""Experimental, CPU-only llama.cpp query tower for C4-E1.

The Hugging Face tokenizer is kept in the application. Passing token IDs to
llama-server avoids a second, potentially different tokenisation recipe.
F32 and Q8_0 parity still require the preregistered real-model controls.
"""

from __future__ import annotations

import atexit
import json
import math
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


class LlamaQueryEncoder:
    """Own a localhost llama-server and return ordered query embeddings."""

    DIMENSION = 2048
    EOS_ID = 151645

    def __init__(
        self,
        *,
        binary: str | Path,
        model: str | Path,
        model_name: str,
        revision: str,
        query_prefix: str,
        max_seq_length: int = 8192,
        threads: int = 2,
        startup_timeout: float = 120.0,
        request_timeout: float = 120.0,
        log_path: str | Path | None = None,
    ) -> None:
        self.binary = Path(binary).expanduser().resolve(strict=True)
        self.model = Path(model).expanduser().resolve(strict=True)
        self.model_name = model_name
        self.revision = revision
        self.query_prefix = query_prefix
        self.max_seq_length = int(max_seq_length)
        self.threads = int(threads)
        self.startup_timeout = float(startup_timeout)
        self.request_timeout = float(request_timeout)
        if self.max_seq_length < 2 or self.threads < 1:
            raise ValueError("max_seq_length must be >=2 and threads must be >=1")
        if not all(math.isfinite(x) and x > 0 for x in (self.startup_timeout, self.request_timeout)):
            raise ValueError("timeouts must be positive and finite")

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            use_fast=True,
            trust_remote_code=False,
            add_eos_token=True,
            model_max_length=self.max_seq_length,
        )
        if self.tokenizer.eos_token_id != self.EOS_ID:
            raise RuntimeError(f"expected EOS {self.EOS_ID}, got {self.tokenizer.eos_token_id}")
        probe = self.token_ids("def f(): return 1")
        if not probe or probe[-1] != self.EOS_ID or probe.count(self.EOS_ID) != 1:
            raise RuntimeError("HF tokenizer must append exactly one terminal EOS")

        self.log_path = Path(log_path).expanduser().resolve() if log_path else None
        self._log_file = None
        self._process: subprocess.Popen | None = None
        self._port: int | None = None
        self._slot_directory = None
        atexit.register(self.close)

    def token_ids(self, text: str) -> list[int]:
        """Exact HF query recipe; the stage-1 control compares it with Torch."""
        if not isinstance(text, str):
            raise TypeError("query text must be a string")
        ids = self.tokenizer.encode(
            # Sentence Transformers 5.0's Transformer.tokenize strips the
            # complete prompted string before invoking this same tokenizer.
            (self.query_prefix + text).strip(),
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_length,
        )
        if not ids or ids[-1] != self.EOS_ID:
            raise RuntimeError("query did not end in EOS")
        return ids

    def _request(self, path: str, payload: dict | None = None, *, timeout: float) -> object:
        if self._port is None:
            raise RuntimeError("llama-server has not started")
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"http://127.0.0.1:{self._port}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, ValueError) as error:
            raise RuntimeError(f"llama-server {path} failed: {error}") from error

    def start(self) -> None:
        if self._process is not None:
            if self._process.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self._process.returncode}")
            return
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self._port = int(sock.getsockname()[1])
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self.log_path.open("x", encoding="utf-8")
        output = self._log_file if self._log_file is not None else subprocess.DEVNULL
        self._slot_directory = tempfile.TemporaryDirectory(prefix="codedoctor-slots-")
        command = [
            str(self.binary), "--model", str(self.model), "--host", "127.0.0.1",
            "--port", str(self._port), "--embedding", "--pooling", "last",
            # b10269 requires token count strictly below context size when
            # last-token pooling can split a prompt across microbatches.
            "--ctx-size", str(self.max_seq_length + 1), "--threads", str(self.threads),
            "--threads-batch", str(self.threads), "--parallel", "1",
            "--no-cont-batching", "--n-gpu-layers", "0",
            "--cache-ram", "0", "--slot-save-path", self._slot_directory.name,
        ]
        try:
            self._process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        except OSError:
            self.close()
            raise
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                self.close()
                raise RuntimeError("llama-server exited during startup; inspect its log")
            try:
                if self._request("/health", timeout=min(1.0, self.request_timeout)) == {"status": "ok"}:
                    return
            except RuntimeError:
                pass
            time.sleep(0.2)
        self.close()
        raise TimeoutError("llama-server startup timed out")

    def encode(
        self,
        texts: str | Sequence[str],
        mode: str = "query",
        batch_size: int | None = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        if mode != "query":
            raise ValueError("LlamaQueryEncoder only encodes queries")
        single = isinstance(texts, str)
        values = [texts] if single else list(texts)
        if not values:
            return np.empty((0, self.DIMENSION), dtype=np.float32)
        self.start()
        rows = []
        # One request at a time makes query ordering and process memory explicit.
        for text in values:
            ids = self.token_ids(text)
            cleared = self._request("/slots/0?action=erase", {}, timeout=self.request_timeout)
            if not isinstance(cleared, dict) or cleared.get("id_slot") != 0:
                raise RuntimeError("llama-server did not confirm prefix-cache erasure")
            response = self._request(
                "/v1/embeddings", {"input": ids, "encoding_format": "float"},
                timeout=self.request_timeout,
            )
            if not isinstance(response, dict) or not isinstance(response.get("data"), list):
                raise RuntimeError("unexpected llama-server embedding response")
            data = response["data"]
            if len(data) != 1 or not isinstance(data[0], dict) or data[0].get("index") != 0:
                raise RuntimeError("unexpected llama-server embedding index")
            if response.get("usage", {}).get("prompt_tokens") != len(ids):
                raise RuntimeError("llama-server token count differs from the HF token IDs")
            embedding = data[0].get("embedding")
            vector = np.asarray(embedding, dtype=np.float32)
            if vector.shape != (self.DIMENSION,) or not np.isfinite(vector).all():
                raise RuntimeError(f"invalid llama-server embedding shape/values: {vector.shape}")
            norm = float(np.linalg.norm(vector))
            if not math.isfinite(norm) or norm <= 0:
                raise RuntimeError("llama-server returned a zero or invalid embedding")
            rows.append(vector / norm)
        result = np.vstack(rows).astype(np.float32)
        return result[0] if single else result

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        if self._slot_directory is not None:
            self._slot_directory.cleanup()
            self._slot_directory = None

    def __enter__(self) -> LlamaQueryEncoder:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
