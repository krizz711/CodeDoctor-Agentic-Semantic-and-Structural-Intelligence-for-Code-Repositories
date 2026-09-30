"""E2 (experiments/C3_cycle): an OpenVINO query model never stands in for the torch model."""

from __future__ import annotations

import pytest

from app.retrieval import encode
from app.retrieval.encode import CodeEncoder


def test_torch_is_the_default_backend():
    enc = CodeEncoder(model_name="m", revision="r", device="cpu", torch_dtype="float32")
    assert enc.backend == "torch" and enc.ov_path is None


def test_openvino_needs_a_saved_model_and_a_known_backend():
    with pytest.raises(ValueError):
        CodeEncoder(model_name="m", device="cpu", backend="openvino")
    with pytest.raises(ValueError):
        CodeEncoder(model_name="m", device="cpu", backend="onnx", ov_path="x")


def test_loaded_models_are_keyed_on_backend(monkeypatch):
    torch_model, ov_model = object(), object()
    monkeypatch.setattr(encode, "_LOADED_MODELS", {
        ("m", "cpu", "float32", "torch", None): torch_model,
        ("m", "cpu", "float32", "openvino", "ir"): ov_model,
    })
    as_torch = CodeEncoder(model_name="m", device="cpu", torch_dtype="float32")
    as_ov = CodeEncoder(model_name="m", device="cpu", torch_dtype="float32", backend="openvino", ov_path="ir")
    assert as_torch.model is torch_model
    assert as_ov.model is ov_model
