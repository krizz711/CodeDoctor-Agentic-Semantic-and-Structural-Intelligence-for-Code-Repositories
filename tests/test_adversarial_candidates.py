"""V4 adversarial controls: programs that should never count as verified.

A verifier is only worth its promotion if cheap programs cannot pass it.  Each
case here is a way a wrong candidate could get lucky: a constant verdict, an
echo of the input, a constant or identity function.  With at least two
examples whose expected outputs differ, none of them may verify -- and the
low-entropy guard must refuse the cases where the examples themselves cannot
tell them apart.
"""

from __future__ import annotations

import pytest

from app.pipeline.execution_rerank import (
    ExecutionConfig,
    WorkedExample,
    call_verification_promotable,
    verify_call_candidate,
    verify_candidate,
)
from app.pipeline.statement import analyze_statement

CONFIG = ExecutionConfig(enabled=True, timeout_seconds=5.0, call_style=True, numeric_tolerance=1e-6)
YES_NO = [WorkedExample("4\n", "YES"), WorkedExample("7\n", "NO")]
NUMBERS = [WorkedExample("2 3\n", "5"), WorkedExample("10 4\n", "14")]


@pytest.mark.parametrize(
    "code",
    [
        'print("YES")',
        'print("NO")',
        "print(input())",
        "import sys\nprint(sys.stdin.read())",
        "print(0)",
        "pass",
    ],
)
def test_trivial_stdin_programs_do_not_verify(code):
    assert verify_candidate(code, YES_NO, CONFIG) is False
    assert verify_candidate(code, NUMBERS, CONFIG) is False


def test_a_real_solution_does_verify():
    assert verify_candidate("n = int(input())\nprint('YES' if n % 2 == 0 else 'NO')", YES_NO, CONFIG)


CALL_STATEMENT = "```\nadd(1, 2) == 3\nadd(10, -4) == 6\nadd(0, 0) == 0\n```"


@pytest.mark.parametrize(
    "code",
    [
        "def add(a, b):\n    return 3",
        "def add(a, b):\n    return a",
        "def add(a, b):\n    return None",
        "def add(*args):\n    return args",
    ],
)
def test_trivial_functions_do_not_verify(code):
    facts = analyze_statement(CALL_STATEMENT)
    result = verify_call_candidate(code, facts, CONFIG)
    assert not call_verification_promotable(result, facts, CONFIG)


def test_the_real_function_is_promoted():
    facts = analyze_statement(CALL_STATEMENT)
    result = verify_call_candidate("def add(a, b):\n    return a + b", facts, CONFIG)
    assert call_verification_promotable(result, facts, CONFIG)


def test_examples_that_cannot_discriminate_never_promote():
    # every expected value identical: a constant function is indistinguishable
    facts = analyze_statement("```\nis_ok(1) == True\nis_ok(2) == True\n```")
    result = verify_call_candidate("def is_ok(x):\n    return True", facts, CONFIG)
    assert result.verified
    assert not call_verification_promotable(result, facts, CONFIG)
