"""The oracle's own matching rules, on hand-built cases.

The parser is scored against external labels by ``scripts/parser_oracle.py``;
these tests pin the matcher so its verdicts do not depend on the parser's
author getting the comparison right by eye.
"""

from __future__ import annotations

from scripts.parser_oracle import (
    PRIMARY,
    Row,
    as_text,
    call_matches,
    derange,
    half_of,
    is_visible,
    norm_lines,
    ratio_ci,
    score_call,
    score_stdin,
)

STATEMENT = """Print the sum of two numbers.

-----Input-----
Two integers a and b.

-----Output-----
Their sum.

-----Examples-----
Input
2 3

Output
5

Input
10 20
Output
30
"""


class TestVisibility:
    def test_line_rendered_example_is_visible(self):
        assert is_visible(norm_lines(STATEMENT), "2 3", "5")

    def test_trailing_spaces_and_blank_lines_do_not_matter(self):
        assert is_visible(norm_lines(STATEMENT), "10 20\n\n", "30   ")

    def test_output_must_follow_input(self):
        assert not is_visible(norm_lines("Output\n5\nInput\n2 3\n"), "2 3", "5")

    def test_inline_example_is_not_visible(self):
        assert not is_visible(norm_lines("For example, Input: 2 3 gives 5."), "2 3", "5")

    def test_empty_input_or_output_is_not_visible(self):
        assert not is_visible(norm_lines(STATEMENT), "", "5")
        assert not is_visible(norm_lines(STATEMENT), "2 3", "")

    def test_a_later_occurrence_of_the_input_is_found(self):
        lines = norm_lines("7\nnote\n7\nOutput\n49")
        assert is_visible(lines, "7", "49")


def test_intra_line_whitespace_is_kept_as_the_verifier_keeps_it():
    # _normalize_output strips line ends only; "2   3" != "2 3".  The oracle
    # inherits the verifier's rule instead of inventing a looser one.
    assert not is_visible(norm_lines("Input\n2   3\nOutput\n5"), "2 3", "5")


def test_list_valued_tests_are_joined_by_lines():
    assert as_text(["3", "1 2 3", ""]) == "3\n1 2 3\n"
    assert as_text("5\n") == "5\n"


class TestCallMatching:
    def test_plain_and_wrapped_outputs(self):
        assert call_matches((3, 4), 7, [3, 4], 7)
        assert call_matches((3, 4), 7, [3, 4], [7])

    def test_wrong_arguments_or_result(self):
        assert not call_matches((3, 5), 7, [3, 4], [7])
        assert not call_matches((3, 4), 8, [3, 4], [7])

    def test_bool_never_equals_int(self):
        assert not call_matches((1,), True, [1], [1])

    def test_float_formatting_tolerated(self):
        assert call_matches((1,), 0.30000000000000004, [1], [0.3])


def test_halves_are_deterministic_and_balanced():
    ids = [f"q{i}" for i in range(2000)]
    halves = [half_of(i) for i in ids]
    assert halves == [half_of(i) for i in ids]
    assert 900 < halves.count("CONF") < 1100


class TestScoring:
    def test_stdin_row(self):
        tests = [("2 3", "5"), ("10 20", "30"), ("1 1", "2")]  # third one is hidden
        row = score_stdin(Row("O2", "p", "g", "CONF", "stdin"), STATEMENT, tests, PRIMARY)
        assert row.extracted == 2 and row.confirmed == 2
        assert row.visible == 2 and row.visible_capped == 2 and row.found == 2
        assert row.exact_set == 1 and not row.unconfirmed and not row.missed

    def test_a_wrong_extraction_is_unconfirmed(self):
        tests = [("2 3", "6")]  # the labels disagree with the statement
        row = score_stdin(Row("O2", "p", "g", "CONF", "stdin"), STATEMENT, tests, PRIMARY)
        assert row.confirmed == 0 and len(row.unconfirmed) == 2

    def test_token_level_diagnostic_forgives_spacing_only(self):
        statement = STATEMENT.replace("Input\n2 3\n", "Input\n2   3\n")
        row = score_stdin(Row("O2", "p", "g", "CONF", "stdin"), statement, [("2 3", "5"), ("10 20", "30")], PRIMARY)
        assert row.confirmed == 1 and row.confirmed_tokens == 2

    def test_call_row(self):
        statement = "Write `add(a, b)`.\n\nExamples:\n\n    add(1, 2) == 3\n    add(2, 2) == 4\n"
        row = score_call(Row("O1", "p", "kata_function", "DIAG", "call"), statement,
                         [([1, 2], [3]), ([2, 2], [4])], PRIMARY)
        assert row.extracted >= 1 and row.confirmed == row.extracted

    def test_ratio_ci(self):
        rows = [Row("O2", str(i), "g", "CONF", "stdin", extracted=2, confirmed=2 if i % 10 else 1) for i in range(50)]
        ci = ratio_ci(rows, "confirmed", "extracted")
        assert ci["num"] == 95 and ci["den"] == 100 and ci["value"] == 0.95
        assert ci["ci95"][0] <= 0.95 <= ci["ci95"][1]
        assert ratio_ci([], "confirmed", "extracted")["value"] is None


def test_derangement_moves_every_test_set_and_keeps_modes():
    problems = [{"id": str(i), "statement": f"s{i}", "io": {"inputs": [str(i)], "outputs": [str(i)]}}
                for i in range(6)]
    problems += [{"id": f"c{i}", "statement": f"c{i}", "io": {"fn_name": "f", "inputs": [[i]], "outputs": [[i]]}}
                 for i in range(4)]
    shuffled = derange(problems)
    assert all(s["statement"] == p["statement"] for s, p in zip(shuffled, problems))
    assert all(s["io"] is not p["io"] for s, p in zip(shuffled, problems))
    assert all(("fn_name" in s["io"]) == ("fn_name" in p["io"]) for s, p in zip(shuffled, problems))
    assert derange(problems) == shuffled  # seeded
