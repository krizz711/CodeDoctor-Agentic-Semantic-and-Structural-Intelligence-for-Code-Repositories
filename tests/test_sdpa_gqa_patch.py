"""The GQA/SDPA workaround applies only where it is needed, and changes nothing else."""

from __future__ import annotations

import pytest
import torch

sdpa_attention = pytest.importorskip("transformers.integrations.sdpa_attention")

from app.retrieval.encode import _keep_sdpa_off_the_math_kernel  # noqa: E402


@pytest.fixture
def restore_transformers(monkeypatch):
    # Registering the original with monkeypatch guarantees it is put back.
    monkeypatch.setattr(sdpa_attention, "use_gqa_in_sdpa", sdpa_attention.use_gqa_in_sdpa)
    return monkeypatch


def test_builds_with_flash_are_left_alone(restore_transformers):
    restore_transformers.setattr(torch.backends.cuda, "is_flash_attention_available", lambda: True)
    original = sdpa_attention.use_gqa_in_sdpa
    assert _keep_sdpa_off_the_math_kernel() is False
    assert sdpa_attention.use_gqa_in_sdpa is original


def test_builds_without_flash_expand_heads_instead(restore_transformers):
    restore_transformers.setattr(torch.backends.cuda, "is_flash_attention_available", lambda: False)
    assert _keep_sdpa_off_the_math_kernel() is True
    key = torch.zeros(1, 8, 4, 16)
    # no mask is exactly the case that used to route to enable_gqa -> math kernel
    assert sdpa_attention.use_gqa_in_sdpa(None, key) is False
    assert _keep_sdpa_off_the_math_kernel() is True  # idempotent


def test_expanded_heads_compute_the_same_attention():
    torch.manual_seed(0)
    q = torch.randn(1, 16, 32, 8)
    k = torch.randn(1, 8, 32, 8)
    v = torch.randn(1, 8, 32, 8)
    grouped = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    expanded = torch.nn.functional.scaled_dot_product_attention(
        q, sdpa_attention.repeat_kv(k, 2), sdpa_attention.repeat_kv(v, 2), is_causal=True
    )
    assert torch.allclose(grouped, expanded, atol=1e-6)
