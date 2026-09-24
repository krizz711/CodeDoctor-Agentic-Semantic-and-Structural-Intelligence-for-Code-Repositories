"""Statement parser v2: judging rules and call-style worked examples.

``execution_rerank.parse_worked_examples`` reads one thing -- stdin/stdout
examples in the layouts the APPS *test* split uses -- and that is why execution
verification covers 97.6% of test statements but only 32% of train statements:
57.6% of train queries are Codewars or LeetCode problems whose examples are
*function calls*, not stdin transcripts.  A dataset "representative of APPS"
may look like either.

This module adds, as pure text analysis (no model, no I/O):

* :func:`parse_judge_rules` -- what the statement says counts as a correct
  answer: several valid answers, a float tolerance, case-insensitive verdicts,
  an answer modulus.  These decide *how* an output may be compared.
* :func:`parse_call_examples` -- ``(args, expected return)`` pairs from the
  call-style layouts: LeetCode ``Input: a = ... / Output: ...``, Codewars
  ``f(x) == y`` / ``=> y`` / ``-> y`` / ``returns y``, ``x ==> y``,
  ``a, b = 1, 2  return  3``, ``name = value ... output = value`` and doctest
  ``>>> f(x)``.
* :func:`parse_extra_stdin_examples` -- stdin layouts the original parser does
  not read (``Sample Input 1`` without dashes, CodeContests ``Examples`` blocks
  with blank-line separators, inline ``Input:`` lines).
* :func:`analyze_statement` -- all of the above plus the execution *mode*.

**Compatibility guarantee.**  :func:`analyze_statement` first runs the original
``parse_worked_examples``; whenever that finds anything, its examples are used
unchanged and nothing new is tried.  A statement that verified before therefore
verifies identically now; the new layouts can only add coverage where there was
none.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from typing import Any

from app.pipeline.execution_rerank import WorkedExample, parse_worked_examples

# --------------------------------------------------------------------------
# Judging rules
# --------------------------------------------------------------------------

_ANY_ANSWER = re.compile(
    r"(multiple|several|many|more than one)\s+(possible\s+|correct\s+|valid\s+|optimal\s+|suitable\s+)?"
    r"(answers|solutions|ways to answer)"
    r"|(print|output|find|return)\s+any(\s+(of\s+them|one|valid|correct|such))?\b"
    r"|any\s+(valid|correct|possible|such|suitable)\s+(answer|solution|permutation|sequence|arrangement|string)"
    r"|if there are (many|multiple|several)",
    re.I,
)
_TOLERANCE_PHRASE = re.compile(r"(absolute|relative)\s+(or\s+(absolute|relative)\s+)?error", re.I)
_TOLERANCE_VALUE = re.compile(
    r"10\s*\^\s*\{?\s*-\s*(\d+)\s*\}?"  # 10^{-6}
    r"|\b1\s*[eE]\s*-\s*(\d+)"  # 1e-6
    r"|\b0\.(0*)1\b"  # 0.000001
)
_CASE_INSENSITIVE = re.compile(
    r"\bin any case\b|\bany case\b|case[- ]insensitive|each letter in any case"
    r"|\b(upper|lower)\s*(case)?\s+(or|and)\s+(upper|lower)\s*case"
    r"|\b(uppercase|lowercase)\s+(or|and)\s+(uppercase|lowercase)",
    re.I,
)
_MODULUS = re.compile(
    r"modul[oe]\s*\$?\s*(10\s*\^\s*\{?\s*9\s*\}?\s*\+\s*7|1\s*0{3}\s*0{3}\s*0{3}\s*7|998\s*244\s*353|10\s*\^\s*\{?\s*9\s*\}?\s*\+\s*9)",
    re.I,
)


@dataclass(frozen=True)
class JudgeRules:
    """What the statement itself says a correct answer looks like."""

    #: Several different outputs are accepted ("print any of them").
    any_answer: bool = False
    #: Absolute-or-relative float tolerance stated by the problem, if any.
    tolerance: float | None = None
    #: Verdict words may be printed in any letter case.
    case_insensitive: bool = False
    #: Answers are reported modulo this number.
    modulus: int | None = None


def parse_judge_rules(text: str) -> JudgeRules:
    tolerance: float | None = None
    phrase = _TOLERANCE_PHRASE.search(text)
    if phrase:
        window = text[phrase.start() : phrase.end() + 160]
        value = _TOLERANCE_VALUE.search(window)
        if value is None:
            tolerance = 1e-6  # the phrase without a readable bound: the CP default
        elif value.group(1) or value.group(2):
            tolerance = 10.0 ** -int(value.group(1) or value.group(2))
        else:
            tolerance = 10.0 ** -(len(value.group(3)) + 1)
    modulus: int | None = None
    mod = _MODULUS.search(text)
    if mod:
        digits = re.sub(r"\s", "", mod.group(1))
        if "998" in digits:
            modulus = 998_244_353
        elif digits.endswith("+9"):
            modulus = 1_000_000_009
        else:
            modulus = 1_000_000_007
    return JudgeRules(
        any_answer=bool(_ANY_ANSWER.search(text)),
        tolerance=tolerance,
        case_insensitive=bool(_CASE_INSENSITIVE.search(text)),
        modulus=modulus,
    )


# --------------------------------------------------------------------------
# Literal values as statements write them
# --------------------------------------------------------------------------

_JS_WORDS = {"true": "True", "false": "False", "null": "None", "nil": "None", "None": "None"}
_JS_WORD_RE = re.compile(r"\b(true|false|null|nil)\b")


def parse_literal(raw: str) -> tuple[bool, Any]:
    """Parse a value as written in a statement: Python, JSON-ish, or Ruby-ish.

    Returns ``(ok, value)``.  Only literals are ever produced (``ast``), never
    evaluated code.
    """
    text = raw.strip().rstrip(";").strip()
    if text.endswith(",") and not text.startswith("("):
        # "return  10, #because ..." -- a list separator, not a 1-tuple.
        text = text.rstrip(",").strip()
    if not text:
        return False, None
    for candidate in (text, _JS_WORD_RE.sub(lambda m: _JS_WORDS[m.group(1)], text)):
        try:
            value = ast.literal_eval(candidate)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        if isinstance(value, tuple) and len(value) == 1 and not text.startswith("("):
            # "10, # because ..." is a separator plus a comment, not a 1-tuple.
            value = value[0]
        return True, value
    return False, None


#: Where a value written in prose may end: before a comment, a sentence, or
#: a trailing explanation ("= 8. The max product is ...").
_CUT_POINTS = re.compile(r"\s+#|\s+//|\s+--(?!>)|\.\s+[A-Z(]|,\s+(?:because|since|as|given)\b|\s+\(|\s{2,}\S")


def parse_leading_literal(raw: str, *, max_chars: int = 400) -> tuple[bool, Any]:
    """The longest leading literal of ``raw`` -- values are often followed by prose."""
    text = raw.strip()[:max_chars]
    ok, value = parse_literal(text)
    if ok:
        return ok, value
    for cut in _CUT_POINTS.finditer(text):
        ok, value = parse_literal(text[: cut.start()])
        if ok:
            return ok, value
    # Last resort: shrink token by token from the right -- but never cut an
    # expression: "23 * 1 = 23" does not *contain* the value 23.
    parts = text.split()
    for end in range(len(parts) - 1, 0, -1):
        if _OPERATOR_TOKEN.match(parts[end]):
            continue
        ok, value = parse_literal(" ".join(parts[:end]))
        if ok:
            return ok, value
    return False, None


_OPERATOR_TOKEN = re.compile(r"^(?:[-+*/%^=<>&|]|\*\*|//(?!\S)|==|!=|<=|>=)")


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split on ``sep`` outside brackets and quotes."""
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    current: list[str] = []
    escape = False
    for char in text:
        if quote:
            current.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == sep and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    parts.append("".join(current))
    return [p.strip() for p in parts]


# --------------------------------------------------------------------------
# Call-style examples
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CallExample:
    """One ``callee(*args, **kwargs) -> expected`` pair from a statement."""

    args: tuple = ()
    kwargs: tuple[tuple[str, Any], ...] = ()
    expected: Any = None
    #: Function name the statement used, when it named one.
    callee: str | None = None
    #: The layout that produced it, for audits.
    layout: str = ""

    def arity(self) -> int:
        return len(self.args) + len(self.kwargs)

    def expected_repr(self) -> str:
        return repr(self.expected)


_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"

# LeetCode: "Input: nums = [2,7,11,15], target = 9" then "Output: [0,1]"
_LEETCODE_IO = re.compile(
    r"Input\s*:?\s*(?P<input>[^\n]+(?:\n(?!\s*(?:Output|Explanation|Example|Note|Constraints)\b)[^\n]+)*)"
    r"\s*\n\s*Output\s*:?\s*(?P<output>[^\n]+)",
    re.I,
)
# f(args) <op> expected, where <op> is ==, =, =>, ->, -->, ==>, returns, should return, # =>
_CALL_ARROW = re.compile(
    rf"(?P<name>{_IDENT})\s*\((?P<args>[^\n]*?)\)\s*"
    r"(?P<op>==>|-->|==|=>|->|=(?!=)|#\s*=>|#\s*->|//\s*=>|\bshould\s+(?:return|equal|be)\b|\breturns?\b|\bis\b)\s*"
    r"(?P<rhs>[^\n]+)"
)
# "day" ==> "dya"   /   [1, 2] --> 3   (no function name)
_VALUE_ARROW = re.compile(r"^\s*(?P<lhs>[^\n=<>-][^\n]*?)\s*(?P<op>==>|-->|=>|->)\s*(?P<rhs>[^\n]+)$", re.M)
# n, k = 100, 3  return  10
_ASSIGN_RETURN = re.compile(
    rf"^\s*(?P<names>{_IDENT}(?:\s*,\s*{_IDENT})*)\s*=\s*(?P<values>[^\n]+?)\s+returns?\s+(?P<rhs>[^\n]+)$",
    re.M | re.I,
)
# >>> f(x)\n expected
_DOCTEST = re.compile(rf">>>\s*(?P<name>{_IDENT})\s*\((?P<args>[^\n]*)\)\s*\n\s*(?P<rhs>[^\n>][^\n]*)")
# text = "This is a test!"\n n = 1\n output = "..."
_ASSIGN_BLOCK = re.compile(
    rf"(?P<block>(?:^[ \t]*{_IDENT}[ \t]*=[ \t]*[^\n]+\n)+?)^[ \t]*(?:output|result|expected|returns?|answer)[ \t]*[=:][ \t]*(?P<rhs>[^\n]+)",
    re.M | re.I,
)

#: Names that look like calls in prose but are never the function under test.
_NOT_CALLEES = frozenset(
    {"print", "input", "len", "range", "int", "str", "list", "max", "min", "sum", "sorted",
     "abs", "O", "f", "g", "gcd", "lcm", "mod", "log", "sqrt", "pow", "e"}
)


def _parse_args(raw: str) -> tuple[bool, tuple, tuple[tuple[str, Any], ...]]:
    """``"1, [2, 3], k=4"`` -> positional and keyword literals."""
    raw = raw.strip()
    if not raw:
        return True, (), ()
    positional: list[Any] = []
    keywords: list[tuple[str, Any]] = []
    for part in _split_top_level(raw):
        if not part:
            return False, (), ()
        keyword = re.match(rf"^({_IDENT})\s*=(?!=)\s*(.+)$", part, re.S)
        if keyword and not part.lstrip().startswith(("'", '"')):
            ok, value = parse_literal(keyword.group(2))
            if not ok:
                return False, (), ()
            keywords.append((keyword.group(1), value))
            continue
        if keywords:  # positional after keyword: not a call we can reproduce
            return False, (), ()
        ok, value = parse_literal(part)
        if not ok:
            return False, (), ()
        positional.append(value)
    return True, tuple(positional), tuple(keywords)


def _leetcode_examples(text: str) -> list[CallExample]:
    examples: list[CallExample] = []
    for match in _LEETCODE_IO.finditer(text):
        raw_input = " ".join(match.group("input").split())
        ok_out, expected = parse_leading_literal(match.group("output"))
        if not ok_out:
            continue
        # "nums = [2,7], target = 9"  or  "[1,2,3]"  or  "\"abc\""
        if re.match(rf"^{_IDENT}\s*=(?!=)", raw_input):
            ok, args, kwargs = _parse_args(raw_input)
            if ok and kwargs and not args:
                examples.append(CallExample((), kwargs, expected, None, "leetcode"))
            continue
        ok, args, kwargs = _parse_args(raw_input)
        if ok and (args or kwargs):
            examples.append(CallExample(args, kwargs, expected, None, "leetcode-positional"))
    return examples


def _arrow_call_examples(text: str) -> list[CallExample]:
    examples: list[CallExample] = []
    for match in _CALL_ARROW.finditer(text):
        name = match.group("name")
        if name in _NOT_CALLEES or name.lower() in {"input", "output", "example", "examples"}:
            continue
        ok, args, kwargs = _parse_args(match.group("args"))
        if not ok:
            continue
        ok_rhs, expected = parse_leading_literal(match.group("rhs"))
        if not ok_rhs:
            continue
        examples.append(CallExample(args, kwargs, expected, name, "call-arrow"))
    for match in _DOCTEST.finditer(text):
        ok, args, kwargs = _parse_args(match.group("args"))
        ok_rhs, expected = parse_leading_literal(match.group("rhs"))
        if ok and ok_rhs:
            examples.append(CallExample(args, kwargs, expected, match.group("name"), "doctest"))
    return examples


def _value_arrow_examples(text: str) -> list[CallExample]:
    examples: list[CallExample] = []
    for match in _VALUE_ARROW.finditer(text):
        lhs = match.group("lhs").strip().rstrip(":")
        if not lhs or re.search(r"[A-Za-z]{4,}\s+[A-Za-z]{4,}", lhs) and not lhs.startswith(("'", '"', "[", "(", "{")):
            continue  # prose, not a value
        ok, args, kwargs = _parse_args(lhs)
        if not ok or kwargs or not args:
            continue
        ok_rhs, expected = parse_leading_literal(match.group("rhs"))
        if ok_rhs:
            examples.append(CallExample(args, (), expected, None, "value-arrow"))
    for match in _ASSIGN_RETURN.finditer(text):
        names = [n.strip() for n in match.group("names").split(",")]
        ok, args, kwargs = _parse_args(match.group("values"))
        ok_rhs, expected = parse_leading_literal(match.group("rhs"))
        if ok and not kwargs and len(args) == len(names) and ok_rhs:
            examples.append(CallExample((), tuple(zip(names, args)), expected, None, "assign-return"))
    for match in _ASSIGN_BLOCK.finditer(text):
        pairs: list[tuple[str, Any]] = []
        for line in match.group("block").strip().splitlines():
            name, _, raw = line.partition("=")
            ok, value = parse_literal(raw)
            if not ok:
                pairs = []
                break
            pairs.append((name.strip(), value))
        ok_rhs, expected = parse_leading_literal(match.group("rhs"))
        if pairs and ok_rhs:
            examples.append(CallExample((), tuple(pairs), expected, None, "assign-block"))
    return examples


def parse_call_examples(text: str, *, limit: int = 5) -> list[CallExample]:
    """Call-style ``(args, expected)`` pairs, most explicit layouts first.

    Duplicates (same args and expected) are dropped; at most ``limit`` kept.
    """
    found: list[CallExample] = []
    seen: set[str] = set()
    for layout in (_leetcode_examples, _arrow_call_examples, _value_arrow_examples):
        for example in layout(text):
            key = repr((example.args, example.kwargs, example.expected))
            if key in seen:
                continue
            seen.add(key)
            found.append(example)
            if len(found) >= limit:
                return found
    return found


# --------------------------------------------------------------------------
# Additional stdin layouts (only tried when the original parser finds nothing)
# --------------------------------------------------------------------------

_SAMPLE_PAIR = re.compile(
    r"(?:^|\n)[ \t]*(?:Sample|Example)[ \t]*Input[ \t]*(?P<n>\d*)[ \t]*:?[ \t]*\n(?P<input>.*?)"
    r"\n[ \t]*(?:Sample|Example)[ \t]*Output[ \t]*(?P=n)[ \t]*:?[ \t]*\n(?P<output>.*?)"
    r"(?=\n[ \t]*(?:Sample|Example)[ \t]*Input|\n[ \t]*(?:Note|Explanation)\b|\Z)",
    re.I | re.S,
)
_EXAMPLES_BLOCK = re.compile(
    r"(?:^|\n)[ \t]*Examples?[ \t]*:?[ \t]*\n(?P<body>.*?)(?=\n[ \t]*(?:Note|Explanation)s?\b[ \t]*:?[ \t]*\n|\Z)",
    re.I | re.S,
)
_BLANK_IO = re.compile(
    r"(?:^|\n)[ \t]*Input[ \t]*:?[ \t]*\n+(?P<input>.*?)\n[ \t]*Output[ \t]*:?[ \t]*\n+(?P<output>.*?)"
    r"(?=\n[ \t]*Input[ \t]*:?[ \t]*\n|\Z)",
    re.I | re.S,
)


def _clean_block(block: str) -> str:
    lines = [line.rstrip() for line in block.strip("\n").splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip("\n")


def _cut_output(block: str) -> str:
    from app.pipeline.execution_rerank import _cut_at_prose

    return _cut_at_prose(block)


def parse_extra_stdin_examples(text: str, *, limit: int = 2) -> list[WorkedExample]:
    examples: list[WorkedExample] = []
    for match in _SAMPLE_PAIR.finditer(text):
        stdin, stdout = _clean_block(match.group("input")), _cut_output(match.group("output"))
        if stdin and stdout and len(stdin) < 4000:
            examples.append(WorkedExample(stdin + "\n", stdout))
            if len(examples) >= limit:
                return examples
    if examples:
        return examples
    block = _EXAMPLES_BLOCK.search(text)
    if block:
        for match in _BLANK_IO.finditer(block.group("body")):
            stdin, stdout = _clean_block(match.group("input")), _cut_output(match.group("output"))
            if stdin and stdout and len(stdin) < 4000:
                examples.append(WorkedExample(stdin + "\n", stdout))
                if len(examples) >= limit:
                    return examples
    return examples


# --------------------------------------------------------------------------
# Putting it together
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StatementFacts:
    """How a statement can be checked by execution, and how to judge it."""

    #: ``stdin``, ``call`` or ``none``.
    mode: str
    rules: JudgeRules = field(default_factory=JudgeRules)
    stdin_examples: tuple[WorkedExample, ...] = ()
    call_examples: tuple[CallExample, ...] = ()
    #: Which parser produced the examples: ``original``, ``extra-stdin``, ``call``.
    source: str = "none"
    #: Distinct expected values across the call examples: a low-entropy guard.
    distinct_expected: int = 0


_CALL_CUES = re.compile(
    rf"\bInput\s*:\s*{_IDENT}\s*=|```|\bdef\s+{_IDENT}\s*\(|\bclass\s+Solution\b|{_IDENT}\s*\([^)\n]*\)\s*(==|=>|->|returns?)",
    re.I,
)


def analyze_statement(text: str, *, max_examples: int = 2, max_call_examples: int = 5) -> StatementFacts:
    """Execution mode, examples and judging rules for one statement."""
    if not isinstance(text, str):
        text = str(text)
    rules = parse_judge_rules(text)

    original = parse_worked_examples(text, limit=max_examples)
    if original:
        return StatementFacts("stdin", rules, tuple(original), (), "original")

    calls: list[CallExample] = []
    if _CALL_CUES.search(text):
        calls = parse_call_examples(text, limit=max_call_examples)
    if calls:
        distinct = len({example.expected_repr() for example in calls})
        return StatementFacts("call", rules, (), tuple(calls), "call", distinct)

    extra = parse_extra_stdin_examples(text, limit=max_examples)
    if extra:
        return StatementFacts("stdin", rules, tuple(extra), (), "extra-stdin")

    calls = parse_call_examples(text, limit=max_call_examples)
    if calls:
        distinct = len({example.expected_repr() for example in calls})
        return StatementFacts("call", rules, (), tuple(calls), "call", distinct)
    return StatementFacts("none", rules)


def values_equal(got: Any, expected: Any, *, tolerance: float | None = None, case_insensitive: bool = False) -> bool:
    """Structural equality for call-style results.

    Sequences compare element-wise regardless of list/tuple; numbers compare
    with ``tolerance`` when given (else exactly, with int/float interchangeable);
    ``bool`` never equals a number, because ``True == 1`` would let a predicate
    "verify" against a count.
    """
    if isinstance(expected, bool) or isinstance(got, bool):
        return isinstance(expected, bool) and isinstance(got, bool) and got == expected
    if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        if tolerance is not None and (isinstance(expected, float) or isinstance(got, float)):
            if math.isnan(got) or math.isnan(expected):
                return False
            return math.isclose(got, expected, rel_tol=tolerance, abs_tol=tolerance)
        return got == expected
    if isinstance(expected, str) and isinstance(got, str):
        return got.lower() == expected.lower() if case_insensitive else got == expected
    if isinstance(expected, (list, tuple)) and isinstance(got, (list, tuple)):
        return len(got) == len(expected) and all(
            values_equal(g, e, tolerance=tolerance, case_insensitive=case_insensitive)
            for g, e in zip(got, expected)
        )
    if isinstance(expected, dict) and isinstance(got, dict):
        return got.keys() == expected.keys() and all(
            values_equal(got[k], expected[k], tolerance=tolerance, case_insensitive=case_insensitive)
            for k in expected
        )
    if isinstance(expected, (set, frozenset)) and isinstance(got, (set, frozenset, list, tuple)):
        return set(got) == set(expected)
    return got == expected
