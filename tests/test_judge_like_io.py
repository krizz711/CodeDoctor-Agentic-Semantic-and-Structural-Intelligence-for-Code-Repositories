"""E1 arm B (experiments/C3_cycle): run candidates the way a judge does.

Judges redirect a file to stdin and run CPython with ``site``; the sandbox fed a
pipe under ``-S``.  Two idioms broke: ``os.read(0, os.fstat(0).st_size)``
(a pipe reports size 0) and bare ``exit()`` (``site`` defines it).  With
``judge_like_io`` both behave as under plain CPython, and everything else is
unchanged.  These tests run on every CI platform.
"""

from __future__ import annotations

import sys

import pytest

from app.pipeline.execution_rerank import ExecutionConfig, WorkedExample, execute_candidate, verify_candidate

FAST_IO = "import os\ndata = os.read(0, os.fstat(0).st_size)\nprint(sum(map(int, data.split())))\n"
BARE_EXIT = "n = int(input())\nprint(n * 2)\nexit()\n"
EXIT_MESSAGE = "print(1)\nexit('bye')\n"
FUTURE = "from __future__ import annotations\nprint(int(input()) + 1)\n"


def config(**kw) -> ExecutionConfig:
    return ExecutionConfig(enabled=True, timeout_seconds=10.0, **kw)


def test_fast_io_idiom_reads_the_real_input_from_a_file():
    example = WorkedExample(stdin="1 2 3\n", expected_stdout="6\n")
    assert verify_candidate(FAST_IO, [example], config(judge_like_io=True))


@pytest.mark.skipif(sys.platform == "darwin", reason=(
    "macOS fstat on a pipe reports the bytes already buffered (CI, 2026-09-26), so the idiom "
    "can pass through a pipe there; Linux and Windows report 0"))
def test_a_pipe_hides_the_input_size_from_the_fast_io_idiom():
    example = WorkedExample(stdin="1 2 3\n", expected_stdout="6\n")
    assert not verify_candidate(FAST_IO, [example], config())


def test_bare_exit_exits_cleanly_like_cpython():
    example = WorkedExample(stdin="4\n", expected_stdout="8\n")
    assert not verify_candidate(BARE_EXIT, [example], config())
    assert verify_candidate(BARE_EXIT, [example], config(judge_like_io=True))


def test_exit_with_a_message_still_fails_like_cpython():
    outcome = execute_candidate(EXIT_MESSAGE, "", config(judge_like_io=True))
    assert outcome.returncode == 1 and outcome.stdout.strip() == "1"
    assert "bye" in outcome.stderr_tail


def test_future_imports_and_input_tracking_still_work():
    example = WorkedExample(stdin="41\n", expected_stdout="42\n")
    assert verify_candidate(FUTURE, [example], config(judge_like_io=True))
    tracked = execute_candidate(FUTURE, "41\n", config(judge_like_io=True), track_input=True)
    assert tracked.ok and tracked.stdin_tracked and tracked.stdin_chars_read >= 2


def test_off_by_default():
    assert ExecutionConfig().judge_like_io is False
