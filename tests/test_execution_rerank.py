"""Execution-based verification: parsing, sandboxing, and the default-off gate."""

from __future__ import annotations

import pytest

from app.pipeline.execution_rerank import (
    ExecutionConfig,
    WorkedExample,
    apply_execution_rerank,
    parse_worked_examples,
    run_candidate,
    verify_candidate,
)


CODEFORCES = """Polycarp has n numbers.

-----Example-----
Input
2
3
1 2 3
2
4 5

Output
6
9
"""

CODECHEF = """Find the last two digits of 5^n.

-----Sample Input-----
2
3

-----Sample Output-----
25
25

The answer is always 25 because five raised to any power above one ends in 25.
"""

KATA = """Write a function that returns the number of squares. ```count(x)```"""


class TestParsing:
    def test_codeforces_example_block(self):
        examples = parse_worked_examples(CODEFORCES)
        assert len(examples) == 1
        assert examples[0].stdin.startswith("2\n3\n1 2 3")
        assert examples[0].expected_stdout == "6\n9"

    def test_codechef_sample_sections(self):
        examples = parse_worked_examples(CODECHEF)
        assert len(examples) == 1
        assert examples[0].stdin == "2\n3\n"
        # the trailing explanatory sentence must not become expected output
        assert examples[0].expected_stdout == "25\n25"

    def test_trailing_note_divider_is_not_expected_output(self):
        text = CODEFORCES + "\n-----Note-----\nIn the first case the sum is six.\n"
        assert parse_worked_examples(text)[0].expected_stdout == "6\n9"

    def test_kata_statement_yields_nothing(self):
        assert parse_worked_examples(KATA) == []

    def test_empty_statement_yields_nothing(self):
        assert parse_worked_examples("") == []

    def test_limit_is_respected(self):
        text = CODEFORCES + "\n-----Example-----\nInput\n1\n\nOutput\n1\n"
        assert len(parse_worked_examples(text, limit=1)) == 1


class TestSandbox:
    config = ExecutionConfig(enabled=True, timeout_seconds=5.0)

    def test_correct_solution_matches(self):
        matched, detail = run_candidate(
            "x = int(input())\nprint(x * 2)", WorkedExample("21\n", "42"), self.config
        )
        assert matched is True
        assert detail == "match"

    def test_wrong_output_does_not_match(self):
        matched, detail = run_candidate(
            "print(0)", WorkedExample("21\n", "42"), self.config
        )
        assert matched is False
        assert detail == "output mismatch"

    def test_infinite_loop_is_killed_by_the_timeout(self):
        matched, detail = run_candidate(
            "while True:\n    pass", WorkedExample("1\n", "1"), ExecutionConfig(enabled=True, timeout_seconds=1.0)
        )
        assert matched is False
        assert detail == "timeout"

    def test_syntax_error_is_caught_not_raised(self):
        matched, _ = run_candidate("def broken(:\n  pass", WorkedExample("1\n", "1"), self.config)
        assert matched is False

    def test_crash_is_caught_not_raised(self):
        matched, detail = run_candidate(
            "import sys\nsys.exit(3)", WorkedExample("1\n", "1"), self.config
        )
        assert matched is False
        assert "exit" in detail

    def test_oversized_snippet_is_refused_without_running(self):
        matched, detail = run_candidate(
            "print(1)\n" * 10_000,
            WorkedExample("1\n", "1"),
            ExecutionConfig(enabled=True, max_code_chars=100),
        )
        assert matched is False
        assert "max_code_chars" in detail

    def test_whitespace_differences_are_tolerated(self):
        matched, _ = run_candidate(
            "print(1)\nprint(2)", WorkedExample("", "1  \n\n  2\n"), self.config
        )
        assert matched is True

    def test_verification_requires_every_example(self):
        examples = [WorkedExample("1\n", "1"), WorkedExample("2\n", "99")]
        assert verify_candidate("print(input())", examples, self.config) is False

    def test_verification_with_no_examples_is_false_not_true(self):
        assert verify_candidate("print(1)", [], self.config) is False


class TestGate:
    def test_disabled_is_the_default_and_touches_nothing(self):
        assert ExecutionConfig().enabled is False
        scores, audit = apply_execution_rerank(CODEFORCES, ["print(1)"], [0.9], ExecutionConfig())
        assert scores == [0.9]
        assert audit["enabled"] is False
        assert audit["candidates_executed"] == 0

    def test_statement_without_examples_short_circuits(self):
        scores, audit = apply_execution_rerank(
            KATA, ["print(1)"], [0.9], ExecutionConfig(enabled=True)
        )
        assert scores == [0.9]
        assert audit["examples_found"] == 0
        assert audit["candidates_executed"] == 0

    def test_verified_candidate_is_promoted(self):
        config = ExecutionConfig(enabled=True, timeout_seconds=5.0, verified_bonus=10.0)
        candidates = [
            "print(0)",
            "n = int(input())\n"
            "for _ in range(n):\n"
            "    input()\n"
            "    print(sum(map(int, input().split())))",
        ]
        scores, audit = apply_execution_rerank(CODEFORCES, candidates, [0.9, 0.1], config)
        assert audit["candidates_verified"] == 1
        assert scores[1] > scores[0], "the verified candidate must overtake the dense winner"

    def test_failure_never_demotes(self):
        config = ExecutionConfig(enabled=True, timeout_seconds=5.0)
        assert config.failure_penalty == 0.0
        scores, _ = apply_execution_rerank(CODEFORCES, ["print(0)"], [0.9], config)
        assert scores == [pytest.approx(0.9)]
