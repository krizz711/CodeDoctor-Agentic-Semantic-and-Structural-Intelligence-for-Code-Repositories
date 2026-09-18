"""Query and snippet categorization (Theme 01 axes 1-4)."""

from __future__ import annotations

import pytest

from app.pipeline.code_analysis import CodeFamily, analyze_code
from app.pipeline.query_analysis import QueryFamily, analyze_query, preprocess_query


COMPETITIVE = """Given a number n. Find the last two digits of 5 ^ n.

-----Input:-----
- N the power in which you need to raise number 5.

-----Output:-----
Last two digits of 5^n.

-----Constraints-----
- $2 \\leq N \\leq 2.1018$

-----Sample Input:-----
2

-----Sample Output:-----
25
"""

KATA = """Your task is to write a function which counts the number of squares
contained in an ASCII art picture. Your function should return 2 for this input.

```count_squares(lines)```
"""

LEETCODE = """Given an integer array nums, return the maximum subarray sum.

Example 1:
Input: nums = [1,2,3]
Output: 6

Constraints:
1 <= nums.length <= 10^5
"""


class TestQueryFamily:
    def test_competitive_statement_is_stdin_shaped(self):
        facts = analyze_query(COMPETITIVE)
        assert facts.family is QueryFamily.COMPETITIVE_STDIN
        assert facts.is_verified()
        assert facts.confidence > 0.5

    def test_kata_statement_is_function_shaped(self):
        facts = analyze_query(KATA)
        assert facts.family is QueryFamily.KATA_FUNCTION
        assert "count_squares" in facts.identifiers

    def test_leetcode_statement_is_method_shaped(self):
        facts = analyze_query(LEETCODE)
        assert facts.family is QueryFamily.LEETCODE_METHOD

    def test_featureless_text_reports_unknown_rather_than_guessing(self):
        facts = analyze_query("do the thing with the stuff")
        assert facts.family is QueryFamily.UNKNOWN
        assert not facts.is_verified()
        assert facts.confidence == 0.0

    def test_empty_query_does_not_raise(self):
        facts = analyze_query("")
        assert facts.family is QueryFamily.UNKNOWN
        assert facts.char_length == 0

    def test_analysis_is_deterministic(self):
        first, second = analyze_query(COMPETITIVE), analyze_query(COMPETITIVE)
        assert first == second

    def test_constraint_literals_are_extracted(self):
        facts = analyze_query(COMPETITIVE)
        assert "25" in facts.numbers


class TestQueryPreprocessing:
    def test_note_section_is_stripped(self):
        text = COMPETITIVE + "\n-----Note-----\nThis note explains nothing useful.\n"
        cleaned = preprocess_query(text)
        assert "-----Note-----" not in cleaned
        assert "Sample Output" in cleaned

    def test_stripping_never_empties_the_query(self):
        text = "\n-----Note-----\nonly a note\n"
        assert preprocess_query(text) == text


class TestCodeFamily:
    def test_stdin_script(self):
        facts = analyze_code("n = int(input())\nprint(n * 2)")
        assert facts.family is CodeFamily.STDIN_SCRIPT
        assert facts.reads_stdin and facts.writes_stdout
        assert facts.evidence() == "VERIFIED"

    def test_bare_print_is_still_submission_shaped(self):
        assert analyze_code("print(25)").family is CodeFamily.STDIN_SCRIPT

    def test_class_solution(self):
        facts = analyze_code("class Solution:\n    def maxSub(self, nums):\n        return max(nums)")
        assert facts.family is CodeFamily.CLASS_SOLUTION
        assert {"Solution", "maxSub"} <= facts.defined_names

    def test_function_library(self):
        facts = analyze_code("def count_squares(lines):\n    return len(lines)")
        assert facts.family is CodeFamily.FUNCTION_LIBRARY
        assert "count_squares" in facts.defined_names

    def test_unparsable_snippet_is_labelled_candidate_not_verified(self):
        facts = analyze_code("def broken(:\n    pass")
        assert facts.parsed is False
        assert facts.evidence() == "CANDIDATE"
        # the regex fallback still recovers the name, but does not claim a parse
        assert "broken" in facts.defined_names

    def test_python2_snippet_falls_back_without_raising(self):
        facts = analyze_code("print 'hello'\nx = raw_input()")
        assert facts.parsed is False
        assert facts.reads_stdin is True

    def test_imports_and_literals(self):
        facts = analyze_code("import math\nfrom functools import reduce\nMOD = 1000000007")
        assert {"math", "functools"} <= facts.imports
        assert "1000000007" in facts.numbers

    def test_empty_snippet_is_unknown(self):
        assert analyze_code("").family is CodeFamily.UNKNOWN

    @pytest.mark.parametrize("text", ["", "   ", "\n\n", "#" * 5000])
    def test_degenerate_inputs_do_not_raise(self, text):
        analyze_code(text)
        analyze_query(text)
