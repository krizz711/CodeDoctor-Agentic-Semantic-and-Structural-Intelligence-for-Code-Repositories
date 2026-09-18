from __future__ import annotations

import torch
import pytest
from app.retrieval.encode import CodeEncoder, get_shared_encoder, _select_device


@pytest.fixture(autouse=True)
def clean_encoder_state():
    yield
    import app.retrieval.encode as enc
    enc._GLOBAL_ENCODER = None
    enc._LOADED_MODELS.clear()


def test_service_encoder_defaults_to_cpu(monkeypatch):
    """
    Section 3.4 Acceptance Test:
    Even if CUDA is available on the machine, the service path (get_shared_encoder)
    must always construct a CPU encoder, ensuring cold-start and Docker reproducibility
    on CPU-only jury machines.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    encoder = get_shared_encoder()
    assert encoder.device == "cpu"
    param_device = next(encoder.model.parameters()).device.type
    assert param_device == "cpu", f"Expected parameter device 'cpu', got {param_device}"


def test_code_encoder_explicit_cpu_default(monkeypatch):
    """CodeEncoder() with no arguments must default to CPU even if CUDA exists."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    enc = CodeEncoder()
    assert enc.device == "cpu"


def test_device_selection_helper(monkeypatch):
    """_select_device only chooses cuda when explicitly requested as 'auto' or 'cuda'."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _select_device(None) == "cpu"
    assert _select_device("cpu") == "cpu"
    assert _select_device("cuda") == "cuda"
    assert _select_device("auto") == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _select_device("auto") == "cpu"
