"""E1 (experiments/C3_cycle): the blank-line fallback in the verifier and the replay."""

from __future__ import annotations

from app.pipeline.execution_rerank import (
    ExecutionConfig,
    WorkedExample,
    collapse_blank_lines,
    has_internal_blank_line,
    verify_candidate,
)
from scripts.replay_outcomes import Policy, candidate_tier

READS_EVERY_LINE = "n = int(input())\nprint(sum(int(input()) for _ in range(n)))\n"
# Reads two lines and ignores the rest: right answer, wrong reason.
IGNORES_THE_REST = "a = int(input())\nb = int(input())\nprint(a + b)\n"


def config(**kw) -> ExecutionConfig:
    return ExecutionConfig(enabled=True, timeout_seconds=5.0, **kw)


def test_detector_and_collapse():
    assert has_internal_blank_line("3\n\n1\n\n2\n")
    assert has_internal_blank_line("1\n   \n2")
    assert not has_internal_blank_line("3\n1\n2\n")
    assert not has_internal_blank_line("\n3\n1\n\n")  # leading/trailing blank lines only
    assert collapse_blank_lines("3\n\n1\n  \n2\n\n") == "3\n1\n2\n"


def test_fallback_recovers_a_program_that_crashes_on_rendered_blank_lines():
    example = WorkedExample(stdin="2\n\n4\n\n5\n", expected_stdout="9\n")
    assert not verify_candidate(READS_EVERY_LINE, [example], config())
    assert verify_candidate(READS_EVERY_LINE, [example], config(blank_line_fallback=True))


def test_fallback_needs_the_whole_input_read():
    example = WorkedExample(stdin="1\n\n2\n\n5\n", expected_stdout="3\n")
    # collapsed input "1\n2\n5\n" prints 3 but never reads the 5
    assert not verify_candidate(IGNORES_THE_REST, [example], config(blank_line_fallback=True))


def test_fallback_is_off_by_default_and_never_removes_a_raw_match():
    assert ExecutionConfig().blank_line_fallback is False
    example = WorkedExample(stdin="2\n4\n5\n", expected_stdout="9\n")
    assert verify_candidate(READS_EVERY_LINE, [example], config(blank_line_fallback=True))


def _entry(fallback_read: int | None, tracked: bool = True) -> tuple[dict, dict]:
    entry = {
        "qid": "q1", "family": "competitive_stdin", "legacy_examples": 1,
        "rules": {"any_answer": False, "tolerance": None, "case_insensitive": False, "modulus": None},
        "stdin_examples": [{"stdin": "2\n\n4\n\n5\n", "expected": "9"}], "call_examples": [],
        "distinct_expected": 0, "fallback_inputs": {"0": "2\n4\n5\n"},
    }
    record = {
        "j": 0, "doc": "d1", "function_style": False,
        "stdin_runs": [{"status": "completed", "rc": 1, "exc": "ValueError", "out": "", "tracked": True,
                        "read": 2, "eof": False, "detail": None}],
        "stdin_runs_fallback": {"0": {"status": "completed", "rc": 0, "exc": None, "out": "9",
                                      "tracked": tracked, "read": fallback_read, "eof": False, "detail": None}},
    }
    return entry, record


def test_replay_uses_the_fallback_only_when_asked_and_fully_read():
    on, off = Policy("fb", blank_fallback=True), Policy("raw")
    entry, record = _entry(fallback_read=6)
    assert candidate_tier(record, entry, off) == "none"
    assert candidate_tier(record, entry, on) == "strong"
    entry, record = _entry(fallback_read=3)            # read "2\n4" only
    assert candidate_tier(record, entry, on) == "none"
    entry, record = _entry(fallback_read=None, tracked=False)
    assert candidate_tier(record, entry, on) == "strong"  # untracked: accepted and counted
