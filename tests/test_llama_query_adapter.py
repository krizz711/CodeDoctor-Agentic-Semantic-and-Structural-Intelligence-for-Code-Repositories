"""C4 adapter boundaries; model fidelity remains a real-model experiment."""

import numpy as np
import pytest

from app.pipeline.encoder import PrePostPipelineEncoder
from app.pipeline.reranker import RerankerConfig
from app.retrieval.llama_query import LlamaQueryEncoder


class OriginalEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, mode="document"):
        self.calls.append((mode, list(texts)))
        return np.ones((len(texts), 2048), dtype=np.float32)

    def similarity(self, a, b):
        return a @ b.T


class QueryStub:
    DIMENSION = 2048

    def __init__(self):
        self.calls = []

    def encode(self, texts, mode="query"):
        self.calls.append((mode, list(texts)))
        return np.full((len(texts), self.DIMENSION), 0.5, dtype=np.float32)


def test_pipeline_routes_only_queries_to_experimental_encoder():
    original = OriginalEncoder()
    query = QueryStub()
    pipeline = PrePostPipelineEncoder(original, RerankerConfig(), query_encoder=query)
    assert pipeline.encode([{"text": ["second", "first"]}], prompt_type="query").shape == (2, 2048)
    assert pipeline.encode([{"text": ["doc"]}], prompt_type="document").shape == (1, 2048)
    assert pipeline.encode([{"text": []}], prompt_type="query").shape == (0, 2048)
    assert query.calls == [("query", ["second", "first"])]
    assert original.calls == [("document", ["doc"])]


def fake_adapter(response):
    adapter = object.__new__(LlamaQueryEncoder)
    adapter.start = lambda: None
    adapter.token_ids = lambda text: [len(text), LlamaQueryEncoder.EOS_ID]
    adapter._request = lambda path, payload, timeout: {"id_slot": 0} if "erase" in path else response
    adapter.request_timeout = 1
    return adapter


def embedding_response(vector):
    return {"data": [{"index": 0, "embedding": vector}], "usage": {"prompt_tokens": 2}}


def test_adapter_preserves_order_and_normalises():
    rows = []
    adapter = fake_adapter(None)
    def request(path, payload, timeout):
        rows.append((path, payload))
        if "erase" in path:
            return {"id_slot": 0}
        return embedding_response([2.0] + [0.0] * 2047)
    adapter._request = request
    result = adapter.encode(["a", "longer"])
    assert result.shape == (2, 2048)
    assert [path for path, _ in rows] == ["/slots/0?action=erase", "/v1/embeddings"] * 2
    assert [payload["input"] for path, payload in rows if path == "/v1/embeddings"] == [[1, adapter.EOS_ID], [6, adapter.EOS_ID]]
    assert np.array_equal(result[:, 0], [1.0, 1.0])
    assert adapter.encode([]).shape == (0, 2048)


@pytest.mark.parametrize("response", [[], embedding_response([float("nan")] * 2048), embedding_response([0.0] * 2048), embedding_response([1.0])])
def test_adapter_fails_closed_on_bad_embedding(response):
    with pytest.raises(RuntimeError):
        fake_adapter(response).encode(["query"])


def test_token_count_mismatch_fails():
    response = embedding_response([1.0] * 2048)
    response["usage"]["prompt_tokens"] = 3
    with pytest.raises(RuntimeError, match="token count"):
        fake_adapter(response).encode("query")


def test_cache_clear_failure_is_visible():
    adapter = fake_adapter(None)
    adapter._request = lambda *args, **kwargs: {}
    with pytest.raises(RuntimeError, match="cache erasure"):
        adapter.encode("query")


def test_prompt_preprocessing_matches_sentence_transformers_strip():
    adapter = object.__new__(LlamaQueryEncoder)
    adapter.query_prefix = "Instruct: test\nQuery: "
    adapter.max_seq_length = 8192
    seen = []
    class Tokenizer:
        def encode(self, text, **kwargs):
            seen.append(text)
            return [1, adapter.EOS_ID]
    adapter.tokenizer = Tokenizer()
    adapter.token_ids("  content\n\t")
    assert seen == ["Instruct: test\nQuery:   content"]


def test_close_kills_unresponsive_child():
    import subprocess
    from unittest.mock import Mock

    adapter = object.__new__(LlamaQueryEncoder)
    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("server", 5), 0]
    adapter._process = process
    adapter._log_file = None
    adapter._slot_directory = None
    adapter.close()
    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert adapter._process is None
