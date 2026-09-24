"""Statement parser v2 and call-style verification.

The statements below are paraphrases of real APPS train-split layouts
(LeetCode ``Input: a = ..`` blocks, Codewars ``f(x) = y`` / ``x ==> y`` /
``n, k = 1, 2  return 3`` / ``name = value .. output = ..``), not test-split
text.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.pipeline.execution_rerank import (
    ExecutionConfig,
    apply_execution_rerank,
    call_verification_promotable,
    is_function_style,
    parse_worked_examples,
    verify_call_candidate,
)
from app.pipeline.statement import (
    analyze_statement,
    parse_call_examples,
    parse_extra_stdin_examples,
    parse_judge_rules,
    parse_leading_literal,
    values_equal,
)

LEETCODE = """Given two numbers, hour and minutes. Return the smaller angle.
Example 1:

Input: hour = 12, minutes = 30
Output: 165

Example 2:

Input: hour = 3, minutes = 30
Output: 75

Constraints:
1 <= hour <= 12"""

KATA_CALL = """Find the maximum product taking one element from each sub-array.
```
Examples:
solve( [[1, 2],[3, 4]] ) = 8. The max product is given by 2 * 4
solve( [[10,-15],[-1,-3]] ) = 45, given by (-15) * (-3)
solve( [[1,-1],[2,3],[10,-100]] ) = 300, given by (-1) * 3 * (-100)
```
Good luck!"""

KATA_ARROW = """Move all vowels to the end.
## Examples
```python
"day"    ==>  "dya"
"apple"  ==>  "pplae"
```"""

KATA_RETURN = """## Examples
```python
n, k = 100, 3  return  10, #because 10*100 becomes 1000
n, k = 36, 2   return   1, #because 36 is already a square
```"""

KATA_BLOCK = """Shift the vowels.
Example:

text = "This is a test!"
n = 1
output = "Thes is i tast!"

text = "This is a test!"
n = 3
output = "This as e tist!"
"""

CODEFORCES = """Print the sum.

-----Example-----
Input
2 3

Output
5
"""


class TestJudgeRules:
    def test_all_rules(self):
        rules = parse_judge_rules(
            "If there are multiple answers, print any of them. Your answer is correct if its "
            "absolute or relative error does not exceed $10^{-9}$. You may print each letter "
            "in any case. Print the answer modulo $10^9+7$."
        )
        assert rules.any_answer and rules.case_insensitive
        assert rules.tolerance == pytest.approx(1e-9)
        assert rules.modulus == 1_000_000_007

    def test_nothing_stated_means_nothing_assumed(self):
        rules = parse_judge_rules("Print the sum of a and b.")
        assert rules == type(rules)()

    def test_other_tolerance_spellings(self):
        assert parse_judge_rules("absolute error at most 1e-4").tolerance == pytest.approx(1e-4)
        assert parse_judge_rules("relative error of 0.000001").tolerance == pytest.approx(1e-6)
        assert parse_judge_rules("the absolute error is small").tolerance == pytest.approx(1e-6)


class TestLiterals:
    def test_trailing_prose_is_cut(self):
        assert parse_leading_literal("8. The max product is given by 2 * 4") == (True, 8)
        assert parse_leading_literal("45, given by (-15) * (-3)") == (True, 45)
        assert parse_leading_literal("10, #because") == (True, 10)

    def test_json_words(self):
        assert parse_leading_literal("[true, false, null]") == (True, [True, False, None])

    def test_non_literals_are_refused(self):
        assert parse_leading_literal("__import__('os').system('x')")[0] is False
        assert parse_leading_literal("the answer")[0] is False


class TestCallExamples:
    def test_leetcode_named_arguments(self):
        examples = parse_call_examples(LEETCODE)
        assert [(e.kwargs, e.expected) for e in examples] == [
            ((("hour", 12), ("minutes", 30)), 165),
            ((("hour", 3), ("minutes", 30)), 75),
        ]

    def test_codewars_call_with_trailing_prose(self):
        examples = parse_call_examples(KATA_CALL)
        assert [(e.callee, e.args, e.expected) for e in examples] == [
            ("solve", ([[1, 2], [3, 4]],), 8),
            ("solve", ([[10, -15], [-1, -3]],), 45),
            ("solve", ([[1, -1], [2, 3], [10, -100]],), 300),
        ]

    def test_value_arrow(self):
        examples = parse_call_examples(KATA_ARROW)
        assert [(e.args, e.expected) for e in examples] == [(("day",), "dya"), (("apple",), "pplae")]

    def test_assign_return(self):
        examples = parse_call_examples(KATA_RETURN)
        assert [(e.kwargs, e.expected) for e in examples] == [
            ((("n", 100), ("k", 3)), 10),
            ((("n", 36), ("k", 2)), 1),
        ]

    def test_assign_block(self):
        examples = parse_call_examples(KATA_BLOCK)
        assert examples[0].kwargs == (("text", "This is a test!"), ("n", 1))
        assert examples[0].expected == "Thes is i tast!"

    def test_doctest(self):
        examples = parse_call_examples(">>> add(1, 2)\n3\n>>> add(-1, 1)\n0\n")
        assert [(e.callee, e.args, e.expected) for e in examples] == [("add", (1, 2), 3), ("add", (-1, 1), 0)]

    def test_math_prose_is_not_an_example(self):
        assert parse_call_examples("Let gcd(a, b) = d be the greatest common divisor.") == []


class TestStatementFacts:
    def test_original_parser_wins_and_is_unchanged(self):
        facts = analyze_statement(CODEFORCES)
        assert facts.mode == "stdin" and facts.source == "original"
        assert [(e.stdin, e.expected_stdout) for e in facts.stdin_examples] == [
            (e.stdin, e.expected_stdout) for e in parse_worked_examples(CODEFORCES)
        ]

    def test_call_statement(self):
        facts = analyze_statement(KATA_CALL)
        assert facts.mode == "call" and facts.distinct_expected == 3

    def test_extra_stdin_layout(self):
        text = "Sum.\n\nSample Input 1\n2 3\n\nSample Output 1\n5\n\nSample Input 2\n1 1\n\nSample Output 2\n2\n"
        assert parse_worked_examples(text) == []
        extra = parse_extra_stdin_examples(text)
        assert [(e.stdin, e.expected_stdout) for e in extra] == [("2 3\n", "5"), ("1 1\n", "2")]

    def test_note_heading_no_longer_leaks_into_output(self):
        text = "Examples\nInput\n3\n1 2 3\nOutput\n6\nInput\n1\n5\nOutput\n5\nNote\nIn the first sample the sum is six.\n"
        assert [e.expected_stdout for e in parse_worked_examples(text)] == ["6", "5"]


class TestValuesEqual:
    def test_sequences_ignore_list_tuple(self):
        assert values_equal((1, 2), [1, 2])

    def test_bool_never_equals_a_number(self):
        assert not values_equal(True, 1)
        assert not values_equal(0, False)

    def test_tolerance_only_for_floats(self):
        assert values_equal(0.30000000000000004, 0.3, tolerance=1e-9)
        assert not values_equal(3, 4, tolerance=1.0)

    def test_case_insensitive_only_when_allowed(self):
        assert not values_equal("Yes", "YES")
        assert values_equal("Yes", "YES", case_insensitive=True)


CONFIG = ExecutionConfig(enabled=True, timeout_seconds=5.0, max_candidates=5, call_style=True)


class TestCallHarness:
    def test_function_style_detection(self):
        assert is_function_style("def solve(a):\n    return a")
        assert is_function_style("class Solution:\n    def f(self, x):\n        return x")
        assert not is_function_style("n = int(input())\nprint(n)")
        assert not is_function_style("def main():\n    print(input())\nmain()")

    def test_free_function_verifies(self):
        facts = analyze_statement(KATA_CALL)
        code = (
            "def solve(arr):\n"
            "    p = q = 1\n"
            "    for k in arr:\n"
            "        x, y = max(k), min(k)\n"
            "        c = (p*x, q*x, p*y, q*y)\n"
            "        p, q = max(c), min(c)\n"
            "    return p\n"
        )
        result = verify_call_candidate(code, facts, CONFIG)
        assert result.verified and result.total == 3
        assert call_verification_promotable(result, facts, CONFIG)

    def test_wrong_function_fails(self):
        facts = analyze_statement(KATA_CALL)
        result = verify_call_candidate("def solve(arr):\n    return 0", facts, CONFIG)
        assert not result.verified

    def test_leetcode_method_with_named_arguments(self):
        facts = analyze_statement(LEETCODE)
        code = (
            "class Solution:\n"
            "    def angleClock(self, hour: int, minutes: int) -> float:\n"
            "        a = abs(30 * (hour % 12) + 0.5 * minutes - 6 * minutes)\n"
            "        return min(a, 360 - a)\n"
        )
        assert verify_call_candidate(code, facts, CONFIG).verified

    def test_parameter_names_that_differ_fall_back_to_positional(self):
        facts = analyze_statement(KATA_RETURN)
        code = (
            "def mul_power(num, power):\n"
            "    from collections import Counter\n"
            "    f, m, d = Counter(), 1, 2\n"
            "    while num > 1:\n"
            "        while num % d == 0:\n"
            "            f[d] += 1; num //= d\n"
            "        d += 1\n"
            "    for p, e in f.items():\n"
            "        m *= p ** ((-e) % power)\n"
            "    return m\n"
        )
        assert verify_call_candidate(code, facts, CONFIG).verified

    def test_helpers_do_not_confuse_resolution(self):
        facts = analyze_statement(KATA_ARROW)
        code = (
            "def is_vowel(c):\n    return c in 'aeiou'\n\n"
            "def move_vowels(s):\n    return ''.join(sorted(s, key=is_vowel))\n"
        )
        # two one-argument functions: the entry point is the last one defined
        assert verify_call_candidate(code, facts, CONFIG).verified

    def test_main_guard_does_not_run(self):
        facts = analyze_statement(KATA_ARROW)
        code = (
            "def move_vowels(s):\n    return ''.join(sorted(s, key=lambda c: c in 'aeiou'))\n"
            "if __name__ == '__main__':\n    raise SystemExit(input())\n"
        )
        assert verify_call_candidate(code, facts, CONFIG).verified

    def test_crash_and_infinite_loop_fail_cleanly(self):
        facts = analyze_statement(KATA_ARROW)
        assert not verify_call_candidate("def f(s):\n    return 1/0", facts, CONFIG).verified
        slow = replace(CONFIG, timeout_seconds=1.0)
        assert not verify_call_candidate("def f(s):\n    while True: pass", facts, slow).verified

    def test_low_entropy_guard(self):
        text = "Return true if n is even.\n```\nis_even(2) == True\nis_even(4) == True\n```"
        facts = analyze_statement(text)
        assert facts.mode == "call" and facts.distinct_expected == 1
        result = verify_call_candidate("def is_even(n):\n    return True", facts, CONFIG)
        assert result.verified
        assert not call_verification_promotable(result, facts, CONFIG)


class TestRerankIntegration:
    def test_flags_off_is_the_old_behaviour(self):
        code = "def solve(arr):\n    return 8 if arr == [[1, 2], [3, 4]] else 45 if arr[0][0] == 10 else 300"
        scores, audit = apply_execution_rerank(KATA_CALL, [code], [0.5], replace(CONFIG, call_style=False))
        assert scores == [0.5]
        assert audit.get("reason") == "statement has no machine-readable worked examples"

    def test_call_style_promotes_a_function(self):
        good = "def move_vowels(s):\n    return ''.join(sorted(s, key=lambda c: c in 'aeiou'))"
        bad = "def move_vowels(s):\n    return s"
        scores, audit = apply_execution_rerank(
            KATA_ARROW, [bad, good], [0.9, 0.8], replace(CONFIG, verified_bonus=1.0)
        )
        assert scores[1] > scores[0]
        assert audit["call_verified"] == 1

    def test_stdin_candidates_are_judged_exactly_as_before(self):
        stdin_ok = "a, b = map(int, input().split())\nprint(a + b)"
        old = apply_execution_rerank(CODEFORCES, [stdin_ok], [0.1], replace(CONFIG, call_style=False))
        new = apply_execution_rerank(CODEFORCES, [stdin_ok], [0.1], CONFIG)
        assert old[0] == new[0]


class TestGuardsFromValidation:
    """Regressions found by the train-split validation run."""

    EXPLANATION = """Sort by value times index.
Example:
```
Input: 23, 2, 3, 4, 5
Product of value and index:
23 => 23 * 1 = 23  -> Output-Pos 4
 2 =>  2 * 2 = 4   -> Output-Pos 1
Output: 2, 3, 4, 23, 5
```"""

    def test_an_expression_is_not_a_leading_literal(self):
        assert parse_leading_literal("23 * 1 = 23  -> Output-Pos 4")[0] is False

    def test_identity_examples_never_promote(self):
        facts = analyze_statement(self.EXPLANATION)
        identity = "def shuffle_it(A, *T):\n    return A"
        result = verify_call_candidate(identity, facts, CONFIG)
        assert not call_verification_promotable(result, facts, CONFIG)

    def test_leetcode_typing_names_are_available(self):
        facts = analyze_statement(LEETCODE)
        code = (
            "class Solution:\n"
            "    def angleClock(self, hour: int, minutes: int) -> Optional[float]:\n"
            "        seen: List[int] = []\n"
            "        a = abs(30 * (hour % 12) + 0.5 * minutes - 6 * minutes)\n"
            "        return min(a, 360 - a)\n"
        )
        assert verify_call_candidate(code, facts, CONFIG).verified

    def test_prelude_never_overrides_the_candidate(self):
        facts = analyze_statement(KATA_ARROW)
        code = (
            "def deque(s):\n    return 'shadowed'\n\n"
            "def move_vowels(s):\n    return deque(s) and ''.join(sorted(s, key=lambda c: c in 'aeiou'))\n"
        )
        assert verify_call_candidate(code, facts, CONFIG).verified


class TestCodeforcesRenderedStatements:
    """Found on CodeContests (held-out): bare Input/Output *specification*
    sections precede the Example block, and 40 of 217 statements had their
    specification prose parsed as example input."""

    STATEMENT = """You are given n numbers. Print their sum.

Input

The first line contains a single integer n (1 ≤ n ≤ 100).

The second line contains n integers.

Output

Print one integer — the sum.

Examples

Input


3
1 2 3


Output


6


Input


1
5


Output


5

Note

In the first example the sum is 1 + 2 + 3 = 6.
"""

    def test_examples_come_from_after_the_example_heading(self):
        examples = parse_worked_examples(self.STATEMENT)
        assert [(e.stdin, e.expected_stdout) for e in examples] == [("3\n1 2 3\n", "6"), ("1\n5\n", "5")]

    def test_a_solution_now_verifies(self):
        from app.pipeline.execution_rerank import verify_candidate

        code = "input()\nprint(sum(map(int, input().split())))"
        assert verify_candidate(code, parse_worked_examples(self.STATEMENT), CONFIG)
