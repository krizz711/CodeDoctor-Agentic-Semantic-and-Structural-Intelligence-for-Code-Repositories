"""Failure modes that used to be silent on the scored path.

Found by an independent review and confirmed in code (experiments/STRESS_TEST.md):
a malformed or missing config silently switched execution off; a sandbox that
could not start a process silently turned the pipeline into dense-only; the
default worker count was the logical-core count; and outcome stores could not
say how close a verdict was to its time limit.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

import app.pipeline.execution_rerank as er
from app.pipeline.encoder import PrePostPipelineEncoder
from app.pipeline.execution_rerank import ExecutionConfig, execute_candidate, spawn_failures
from app.pipeline.query_analysis import analyze_query
from app.pipeline.reranker import RerankerConfig
from scripts.evaluate_mteb import get_pipeline_config, resolve_workers

STATEMENT = "Print the sum.\n\n-----Example-----\nInput\n2 3\n\nOutput\n5\n"


class TestConfigFailsLoud:
    def test_a_malformed_config_raises(self, tmp_path):
        path = tmp_path / "broken.yaml"
        path.write_text("pipeline: [unclosed\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid YAML"):
            get_pipeline_config(path)

    def test_a_missing_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_pipeline_config(tmp_path / "missing.yaml")


def test_default_workers_are_physical_cores():
    psutil = pytest.importorskip("psutil")
    physical = psutil.cpu_count(logical=False)
    if not physical:
        pytest.skip("psutil cannot report physical cores on this machine")
    assert resolve_workers(0) == physical
    assert resolve_workers(3) == 3


def _refuse(*args, **kwargs):
    raise OSError("simulated: cannot create process")


class TestSpawnFailuresAreVisible:
    def test_counted_and_warned(self, monkeypatch):
        monkeypatch.setattr(er.subprocess, "Popen", _refuse)
        before = spawn_failures()
        with pytest.warns(RuntimeWarning, match="could not start"):
            outcome = execute_candidate("print(5)", "", ExecutionConfig(enabled=True))
        assert outcome.status == "spawn_failed"
        assert spawn_failures() == before + 1

    def test_the_pipeline_audit_reports_them(self, monkeypatch):
        monkeypatch.setattr(er.subprocess, "Popen", _refuse)
        config = ExecutionConfig(enabled=True, max_candidates=2, timeout_seconds=5.0)
        pipeline = PrePostPipelineEncoder(object(), RerankerConfig(), execution_config=config)
        pipeline._query_texts = [STATEMENT]
        pipeline._query_facts = [analyze_query(STATEMENT)]
        pipeline._document_texts = ["print(5)", "print(6)"]
        with pytest.warns(RuntimeWarning):
            pipeline._apply_execution(np.array([[0.9, 0.8]], dtype=np.float32))
        assert pipeline.execution_audit["spawn_failures"] == 2


def test_real_runs_record_their_duration():
    outcome = execute_candidate("print(1)", "", ExecutionConfig(enabled=True, timeout_seconds=5.0))
    assert outcome.status == "completed"
    assert isinstance(outcome.seconds, float) and 0.0 < outcome.seconds < 5.0


def test_outcome_stores_keep_the_duration():
    from scripts.record_outcomes import _stdin_record

    outcome = execute_candidate("print(input())", "7\n", ExecutionConfig(enabled=True, timeout_seconds=5.0),
                                track_input=True)
    assert _stdin_record(outcome)["secs"] == outcome.seconds


def test_subprocess_module_is_the_one_patched():
    # The tests above patch ``er.subprocess.Popen``; make sure that is the
    # module execute_candidate actually uses, or they would prove nothing.
    assert er.subprocess is subprocess
