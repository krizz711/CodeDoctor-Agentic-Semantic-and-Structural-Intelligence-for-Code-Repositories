"""Every MTEB evaluation path must run the model, not look up its score.

``mteb.evaluate`` defaults to ``cache=ResultCache(~/.cache/mteb)`` with
``overwrite_strategy="only-missing"``. Our encoder advertises the *upstream*
model meta -- ``codefuse-ai/F2LLM-v2-*`` at a pinned revision -- so on a machine
that has pulled MTEB's public results repository the evaluator matches a
published result and returns it without a single forward pass.

It is a silent failure. The run finishes in ~11s instead of hours, writes a
well-formed result JSON, and reports the BASE ENCODER's score
(0.34186 on CodeTransOceanDL, 0.93692 on AppsRetrieval) as though it were the
pipeline's. Nothing in the output distinguishes it from a real run.

These tests read the source rather than importing it, because ``mteb`` is an
optional evaluation dependency that the default test environment does not have.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
EVAL_PATHS = [SCRIPTS / "evaluate_mteb.py", SCRIPTS / "blind_eval.py"]


def _evaluate_call_kwargs(source: str) -> list[dict[str, ast.expr]]:
    """Every ``mteb.evaluate(...)`` call's keyword arguments, by name."""
    calls = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "evaluate":
            value = func.value
            if isinstance(value, ast.Name) and value.id == "mteb":
                calls.append({kw.arg: kw.value for kw in node.keywords if kw.arg})
    return calls


def _resolve(name: str, source: str, kwargs: dict[str, ast.expr]) -> ast.expr | None:
    """Follow one level of ``**some_dict`` indirection to find ``name``."""
    if name in kwargs:
        return kwargs[name]
    # evaluate_mteb.py builds an `evaluate_kwargs` dict and splats it.
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == name:
                    return value
    return None


@pytest.mark.parametrize("path", EVAL_PATHS, ids=lambda p: p.name)
def test_result_cache_is_disabled(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    calls = _evaluate_call_kwargs(source)
    assert calls, f"{path.name} does not call mteb.evaluate"

    for kwargs in calls:
        cache = _resolve("cache", source, kwargs)
        assert cache is not None, (
            f"{path.name} calls mteb.evaluate without passing `cache`. The default "
            "ResultCache will serve the base encoder's published score."
        )
        assert isinstance(cache, ast.Constant) and cache.value is None, (
            f"{path.name} must pass cache=None to mteb.evaluate"
        )


@pytest.mark.parametrize("path", EVAL_PATHS, ids=lambda p: p.name)
def test_overwrite_strategy_is_always(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    for kwargs in _evaluate_call_kwargs(source):
        strategy = _resolve("overwrite_strategy", source, kwargs)
        assert strategy is not None, f"{path.name} does not pin overwrite_strategy"
        assert isinstance(strategy, ast.Constant) and strategy.value == "always", (
            f"{path.name} must pass overwrite_strategy='always'; "
            "the default 'only-missing' skips tasks that already have a cached result"
        )
