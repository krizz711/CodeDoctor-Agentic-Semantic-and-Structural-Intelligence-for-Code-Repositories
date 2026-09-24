"""The sandbox runner: limits applied in the child, never fatal; bytes, not locale text.

These pin the fixes for three defects in the ``preexec_fn`` sandbox:

* on macOS ``setrlimit(RLIMIT_AS)`` is refused, the refusal surfaced as an
  uncaught ``SubprocessError`` and aborted the whole evaluation;
* ``preexec_fn`` is unsafe in a multi-threaded parent (torch);
* on Windows the parent encoded stdin with the locale codec (cp1252), so any
  worked example containing a non-ASCII character failed as "spawn failed".
"""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

from app.pipeline import execution_rerank as er
from app.pipeline.execution_rerank import (
    ExecutionConfig,
    WorkedExample,
    compare_output,
    execute_candidate,
    run_candidate,
)

CONFIG = ExecutionConfig(enabled=True, timeout_seconds=5.0)


def _runner_namespace() -> dict:
    """The runner's helpers, loaded without executing its ``_main()``."""
    source = er._RUNNER_SOURCE.replace("\n_main()\n", "\n")
    namespace: dict = {}
    exec(compile(source, "_runner.py", "exec"), namespace)
    return namespace


class TestLimitsAreNeverFatal:
    def test_a_refused_limit_is_skipped(self, monkeypatch):
        calls = []

        def refuse(which, value):
            calls.append(which)
            raise ValueError("current limit exceeds maximum limit")  # what macOS says

        fake = types.SimpleNamespace(RLIMIT_AS=9, RLIMIT_NPROC=7, setrlimit=refuse)
        monkeypatch.setitem(sys.modules, "resource", fake)
        _runner_namespace()["_apply_limits"](512)  # must not raise
        assert calls == [9, 7], "every limit is still attempted"

    def test_a_platform_without_resource_is_fine(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "resource", None)  # import raises ImportError
        _runner_namespace()["_apply_limits"](512)

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="limits are enforced on Linux")
    def test_limits_are_in_force_inside_the_child(self):
        code = (
            "import resource\n"
            "print(resource.getrlimit(resource.RLIMIT_AS)[0])\n"
            "print(resource.getrlimit(resource.RLIMIT_NPROC)[0])\n"
        )
        outcome = execute_candidate(code, "", CONFIG)
        assert outcome.ok
        assert outcome.stdout.split() == [str(512 * 1024 * 1024), "64"]

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS is enforced on Linux")
    def test_a_memory_hog_fails_cleanly(self):
        matched, detail = run_candidate(
            "x = bytearray(2 * 1024 * 1024 * 1024)\nprint(1)", WorkedExample("", "1"), CONFIG
        )
        assert matched is False
        assert "exit" in detail


class TestFailuresAreReportedNotRaised:
    def test_subprocess_error_at_spawn_is_caught(self, monkeypatch):
        def explode(*args, **kwargs):
            raise subprocess.SubprocessError("Exception occurred in preexec_fn.")

        monkeypatch.setattr(er, "_run_capped", explode)
        matched, detail = run_candidate("print(1)", WorkedExample("", "1"), CONFIG)
        assert matched is False
        assert detail.startswith("spawn failed")

    def test_exception_class_is_recorded(self):
        matched, detail = run_candidate("int('x')", WorkedExample("", "1"), CONFIG)
        assert matched is False
        assert detail == "exit 1 (ValueError)"

    def test_compile_time_errors_are_classified(self):
        # 'return' outside function is raised while compiling, not parsing.
        outcome = execute_candidate("return 1", "", CONFIG)
        assert outcome.returncode == 1 and outcome.exception == "SyntaxError"

    def test_sys_exit_keeps_its_code(self):
        outcome = execute_candidate("import sys\nprint(5)\nsys.exit(4)", "", CONFIG)
        assert outcome.returncode == 4
        assert outcome.stdout.strip() == "5"

    def test_an_output_flood_is_cut_off(self):
        config = ExecutionConfig(enabled=True, timeout_seconds=5.0, max_output_bytes=10_000)
        outcome = execute_candidate("while True:\n    print('y' * 100)", "", config)
        assert outcome.status == "output_cap"
        assert len(outcome.stdout.encode()) <= 10_000
        assert run_candidate("while True:\n    print('y' * 100)", WorkedExample("", "y"), config) == (
            False,
            "output cap exceeded",
        )

    def test_unread_stdin_does_not_hang(self):
        big = "1\n" * 200_000  # larger than a pipe buffer, never read
        outcome = execute_candidate("print(7)", big, CONFIG)
        assert outcome.ok and outcome.stdout.strip() == "7"


class TestFaithfulExecution:
    def test_non_ascii_round_trips(self):
        matched, detail = run_candidate(
            "print(input())", WorkedExample("héllo ✓\n", "héllo ✓"), CONFIG
        )
        assert (matched, detail) == (True, "match")

    def test_candidate_runs_as_main(self):
        code = "if __name__ == '__main__':\n    import __main__\n    print(__main__.__file__.endswith('candidate.py'))"
        assert execute_candidate(code, "", CONFIG).stdout.strip() == "True"

    def test_input_is_passed_as_exact_bytes(self):
        code = "import sys\nprint(repr(sys.stdin.buffer.read()))"
        assert execute_candidate(code, "1 2\n3\n", CONFIG).stdout.strip() == repr(b"1 2\n3\n")


class TestInputTracking:
    def test_off_by_default(self):
        outcome = execute_candidate("print(input())", "5\n", CONFIG)
        assert outcome.stdin_tracked is False and outcome.stdin_chars_read is None

    def test_partial_read_is_visible(self):
        outcome = execute_candidate("print(int(input()))", "5\n6 7\n", CONFIG, track_input=True)
        assert outcome.stdin_tracked and outcome.stdin_chars_read == 2
        assert outcome.stdin_eof_hit is False

    def test_reading_past_the_end_is_visible(self):
        outcome = execute_candidate("a = input()\nb = input()", "5\n", CONFIG, track_input=True)
        assert outcome.exception == "EOFError"
        assert outcome.stdin_eof_hit is True

    def test_full_read_and_iteration_are_counted(self):
        text = "1 2\n3\n"
        read_all = execute_candidate("import sys\nsys.stdin.read()", text, CONFIG, track_input=True)
        assert read_all.stdin_chars_read == len(text)
        iterate = execute_candidate("import sys\nfor l in sys.stdin: pass", text, CONFIG, track_input=True)
        assert iterate.stdin_chars_read == len(text) and iterate.stdin_eof_hit is True

    def test_byte_level_reads_are_reported_untracked(self):
        for code in ("print(open(0).read())", "import sys\nprint(sys.stdin.buffer.read())"):
            outcome = execute_candidate(code, "1\n", CONFIG, track_input=True)
            assert outcome.ok
            assert outcome.stdin_tracked is False and outcome.stdin_chars_read is None

    def test_tracking_does_not_change_the_output(self):
        code = "import sys\nn = int(input())\nprint(sum(int(x) for x in sys.stdin.readline().split()) * n)"
        plain = execute_candidate(code, "2\n1 2 3\n", CONFIG)
        tracked = execute_candidate(code, "2\n1 2 3\n", CONFIG, track_input=True)
        assert plain.stdout == tracked.stdout and plain.returncode == tracked.returncode


class TestCompareOutput:
    def test_exact_then_numeric(self):
        # Line ends and blank lines are ignored; spacing *inside* a line is not.
        assert compare_output("  1 2  \n\n", "1 2", CONFIG) == "match"
        assert compare_output("1 2\n", "1  2", CONFIG) is None
        assert compare_output("1.0000000001", "1.0", CONFIG) is None
        numeric = ExecutionConfig(enabled=True, numeric_tolerance=1e-6)
        assert compare_output("1.0000000001", "1.0", numeric) == "numeric match"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_CPU is enforced on Linux")
class TestCpuTimeMode:
    """Wall-clock verdicts depend on machine load; CPU-time verdicts much less."""

    def test_a_busy_loop_is_stopped_by_cpu_time(self):
        config = ExecutionConfig(enabled=True, timeout_seconds=1.0, timeout_mode="cpu", wall_backstop_factor=4.0)
        assert execute_candidate("while True:\n    pass", "", config).status == "timeout"

    def test_a_sleeper_is_stopped_by_the_wall_backstop(self):
        config = ExecutionConfig(enabled=True, timeout_seconds=1.0, timeout_mode="cpu", wall_backstop_factor=2.0)
        assert execute_candidate("import time\ntime.sleep(10)", "", config).status == "timeout"

    def test_normal_programs_are_unaffected(self):
        config = ExecutionConfig(enabled=True, timeout_seconds=2.0, timeout_mode="cpu")
        assert run_candidate("print(int(input()) * 2)", WorkedExample("21\n", "42"), config) == (True, "match")


def test_cpu_mode_falls_back_to_wall_clock_off_posix(monkeypatch):
    monkeypatch.setattr(er.os, "name", "nt")
    config = ExecutionConfig(enabled=True, timeout_seconds=1.0, timeout_mode="cpu")
    assert execute_candidate("while True:\n    pass", "", config).status == "timeout"
