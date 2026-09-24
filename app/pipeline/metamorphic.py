"""Metamorphic sample amplification: new tests with known answers, from the samples.

Worked examples verify the *problem* a program solves, but not whether it
solves it correctly: on held-out CodeContests, incorrect submissions to the
same problem reproduced the public examples 43% of the time against 51% for
correct ones -- contestants test on the samples before submitting.  More tests
are needed, and at query time there is no oracle and (per the FAQ) no LLM.

Multi-test-case statements carry an oracle anyway.  When the input is ``t``
followed by ``t`` independent cases and the output is the cases' outputs in
order, then for the *same* cases listed twice (``2t`` on the first line, the
case lines repeated) the correct output is the sample output repeated.  No
grammar, no generator, no model: the relation holds for every correct program
of such a statement, and it breaks the classic near-miss bugs -- state leaking
between test cases (a global not reset, a list not cleared), reading ``t`` and
ignoring it, reading the wrong number of lines.

Applied only when the statement says it has test cases *and* the sample's first
line is a single integer ``t`` with at least one case line after it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.pipeline.execution_rerank import WorkedExample

_TEST_CASES = re.compile(
    r"number of test ?cases|the first line (?:of (?:the )?input )?contains (?:a single |one )?integer \$?t\$?"
    r"|\bt\s*\(?\$?\s*1\s*\\?le|test ?cases? follow|each test ?case",
    re.I,
)


@dataclass(frozen=True)
class MetamorphicExample:
    """A derived test and the relation that produced it."""

    example: WorkedExample
    relation: str


def is_multitest(statement: str) -> bool:
    return bool(_TEST_CASES.search(statement))


def duplicate_cases(example: WorkedExample, *, copies: int = 2) -> MetamorphicExample | None:
    """``t`` cases -> the same cases ``copies`` times, output repeated.

    Returns ``None`` when the sample does not look like ``t`` then cases.
    """
    lines = example.stdin.strip("\n").split("\n")
    if len(lines) < 2:
        return None
    first = lines[0].strip()
    if not re.fullmatch(r"\d{1,5}", first):
        return None
    t = int(first)
    if t < 1:
        return None
    body = [line.rstrip() for line in lines[1:]]
    while body and not body[-1].strip():
        body.pop()
    if not body:
        return None
    expected = example.expected_stdout.strip("\n")
    if not expected.strip():
        return None
    stdin = "\n".join([str(t * copies)] + body * copies) + "\n"
    expected_out = "\n".join([expected] * copies)
    return MetamorphicExample(WorkedExample(stdin, expected_out), f"duplicate-cases x{copies}")


def derive(statement: str, examples: list[WorkedExample], *, copies: int = 2) -> list[MetamorphicExample]:
    """Every metamorphic test the statement's samples support (possibly none)."""
    if not is_multitest(statement):
        return []
    derived = []
    for example in examples:
        test = duplicate_cases(example, copies=copies)
        if test is not None:
            derived.append(test)
    return derived
