"""Execution-based verification: parsing, sandboxing, and the default-off gate."""

from __future__ import annotations

from dataclasses import replace

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


#: Reproduces the CODEFORCES worked example (sum each of two lists).
SOLVER = (
    "n = int(input())\n"
    "for _ in range(n):\n"
    "    input()\n"
    "    print(sum(map(int, input().split())))"
)


class TestEarlyExit:
    """Stopping at the first verified candidate cannot change the winner.

    Every verified candidate receives the same bonus, so verified candidates
    keep their dense order among themselves and the first to verify is the one
    that ends up on top.  These tests pin that property, because it is the
    entire justification for not executing the rest.
    """

    config = ExecutionConfig(enabled=True, timeout_seconds=5.0, verified_bonus=10.0)

    def test_off_by_default(self):
        assert ExecutionConfig().stop_after_first_match is False

    def test_stops_executing_after_the_first_match(self):
        config = replace(self.config, stop_after_first_match=True)
        # The solver is first, so nothing after it should run.
        candidates = [SOLVER, "print(0)", "print(0)", "print(0)"]
        _, audit = apply_execution_rerank(
            CODEFORCES, candidates, [0.9, 0.8, 0.7, 0.6], config
        )
        assert audit["candidates_executed"] == 1
        assert audit["candidates_verified"] == 1
        assert audit["stopped_early"] is True

    def test_runs_everything_when_nothing_verifies(self):
        config = replace(self.config, stop_after_first_match=True)
        candidates = ["print(0)", "print(1)", "print(2)"]
        _, audit = apply_execution_rerank(
            CODEFORCES, candidates, [0.9, 0.8, 0.7], config
        )
        assert audit["candidates_executed"] == 3
        assert audit["stopped_early"] is False

    def test_the_winner_is_identical_either_way(self):
        """The property the optimisation rests on."""
        candidates = ["print(0)", SOLVER, "print(0)", SOLVER]
        scores = [0.9, 0.8, 0.7, 0.6]
        full, _ = apply_execution_rerank(CODEFORCES, candidates, scores, self.config)
        early, _ = apply_execution_rerank(
            CODEFORCES, candidates, scores, replace(self.config, stop_after_first_match=True)
        )
        assert full.index(max(full)) == early.index(max(early)) == 1

    def test_early_exit_lifts_fewer_documents(self):
        """Why it is not a pure no-op: it promotes one, not many."""
        candidates = ["print(0)", SOLVER, SOLVER, "print(0)"]
        scores = [0.9, 0.8, 0.7, 0.6]
        full, _ = apply_execution_rerank(CODEFORCES, candidates, scores, self.config)
        early, _ = apply_execution_rerank(
            CODEFORCES, candidates, scores, replace(self.config, stop_after_first_match=True)
        )
        assert sum(s > 1.0 for s in full) == 2
        assert sum(s > 1.0 for s in early) == 1


class TestDenseGate:
    """Refusing to execute candidates far below the top of the pool."""

    config = ExecutionConfig(enabled=True, timeout_seconds=5.0, verified_bonus=10.0)

    def test_off_by_default(self):
        assert ExecutionConfig().max_dense_gap is None

    def test_candidates_beyond_the_gap_are_not_executed(self):
        config = replace(self.config, max_dense_gap=0.10)
        candidates = ["print(0)"] * 4
        _, audit = apply_execution_rerank(
            CODEFORCES, candidates, [0.90, 0.85, 0.60, 0.55], config
        )
        assert audit["candidates_executed"] == 2
        assert audit["candidates_skipped_by_gate"] == 2

    def test_the_top_candidate_is_never_gated_out(self):
        config = replace(self.config, max_dense_gap=0.0)
        _, audit = apply_execution_rerank(
            CODEFORCES, ["print(0)", "print(0)"], [0.9, 0.5], config
        )
        assert audit["candidates_executed"] == 1

    def test_a_wide_gap_executes_everything(self):
        config = replace(self.config, max_dense_gap=10.0)
        candidates = ["print(0)"] * 3
        _, audit = apply_execution_rerank(CODEFORCES, candidates, [0.9, 0.5, 0.1], config)
        assert audit["candidates_executed"] == 3
        assert audit["candidates_skipped_by_gate"] == 0

    def test_gate_blocks_a_distant_false_positive(self):
        """A lookalike far down the pool no longer gets the chance to verify."""
        candidates = ["print(0)", SOLVER]
        scores = [0.90, 0.40]
        ungated, _ = apply_execution_rerank(CODEFORCES, candidates, scores, self.config)
        gated, _ = apply_execution_rerank(
            CODEFORCES, candidates, scores, replace(self.config, max_dense_gap=0.10)
        )
        assert ungated[1] > ungated[0], "without the gate the distant candidate wins"
        assert gated == [pytest.approx(0.90), pytest.approx(0.40)]

    def test_gate_and_early_exit_compose(self):
        config = replace(self.config, max_dense_gap=0.10, stop_after_first_match=True)
        candidates = [SOLVER, "print(0)", SOLVER]
        _, audit = apply_execution_rerank(
            CODEFORCES, candidates, [0.90, 0.85, 0.40], config
        )
        assert audit["candidates_executed"] == 1
        assert audit["stopped_early"] is True


class TestPromotionGate:
    """Run the candidate, record the verification, withhold the bonus.

    Distinct from ``max_dense_gap``: that one never runs the distant
    candidate, so nothing is learned about it.  This one runs it, counts it,
    and only refuses to let it displace the answer.
    """

    config = ExecutionConfig(enabled=True, timeout_seconds=5.0, verified_bonus=10.0)

    def test_off_by_default(self):
        assert ExecutionConfig().max_promotion_gap is None

    def test_default_behaviour_is_byte_identical(self):
        """Point 8, first clause: the shipped path must not move."""
        candidates = ["print(0)", SOLVER]
        scores = [0.90, 0.40]
        before, audit_before = apply_execution_rerank(
            CODEFORCES, candidates, scores, self.config
        )
        after, audit_after = apply_execution_rerank(
            CODEFORCES, candidates, scores, replace(self.config, max_promotion_gap=None)
        )
        assert before == after
        assert audit_before["candidates_verified"] == audit_after["candidates_verified"]
        assert audit_after["promotions_blocked"] == 0

    def test_verified_candidate_outside_the_margin_is_not_promoted(self):
        config = replace(self.config, max_promotion_gap=0.05)
        scores, audit = apply_execution_rerank(
            CODEFORCES, ["print(0)", SOLVER], [0.90, 0.40], config
        )
        assert audit["candidates_verified"] == 1, "it still ran and still verified"
        assert audit["promotions_blocked"] == 1
        assert audit["candidates_promoted"] == 0
        assert scores == [pytest.approx(0.90), pytest.approx(0.40)]

    def test_candidate_inside_the_margin_behaves_exactly_as_before(self):
        candidates = ["print(0)", SOLVER]
        scores_in = [0.90, 0.88]
        ungated, _ = apply_execution_rerank(CODEFORCES, candidates, scores_in, self.config)
        gated, audit = apply_execution_rerank(
            CODEFORCES, candidates, scores_in, replace(self.config, max_promotion_gap=0.05)
        )
        assert gated == ungated
        assert audit["promotions_blocked"] == 0
        assert audit["candidates_promoted"] == 1

    def test_it_still_executes_what_it_refuses_to_promote(self):
        """The difference from max_dense_gap, pinned."""
        promo = replace(self.config, max_promotion_gap=0.05)
        exec_gate = replace(self.config, max_dense_gap=0.05)
        candidates = ["print(0)", SOLVER, SOLVER]
        scores = [0.90, 0.40, 0.30]
        _, promo_audit = apply_execution_rerank(CODEFORCES, candidates, scores, promo)
        _, exec_audit = apply_execution_rerank(CODEFORCES, candidates, scores, exec_gate)
        assert promo_audit["candidates_executed"] == 3
        assert promo_audit["candidates_verified"] == 2
        assert exec_audit["candidates_executed"] == 1
        assert exec_audit["candidates_verified"] == 0

    def test_the_top_candidate_can_always_be_promoted(self):
        config = replace(self.config, max_promotion_gap=0.0)
        scores, audit = apply_execution_rerank(
            CODEFORCES, [SOLVER, SOLVER], [0.9, 0.5], config
        )
        assert audit["candidates_promoted"] == 1
        assert audit["promotions_blocked"] == 1
        assert scores[0] > scores[1]

    def test_blocked_promotion_does_not_trigger_early_exit(self):
        config = replace(
            self.config, max_promotion_gap=0.05, stop_after_first_match=True
        )
        # First verifier is out of margin; the in-margin one must still be reached.
        candidates = ["print(0)", SOLVER]
        scores, audit = apply_execution_rerank(
            CODEFORCES, candidates, [0.90, 0.40], config
        )
        assert audit["candidates_executed"] == 2
        assert audit["stopped_early"] is False
