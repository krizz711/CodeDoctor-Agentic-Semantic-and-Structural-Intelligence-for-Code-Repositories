"""Query categorization and pre-processing (Theme 01 improvement axes 1 and 2).

A natural-language code-retrieval query is not a bag of words: competitive
programming statements, LeetCode prompts and Codewars katas each imply a
*different shape of answer*.  Detecting that shape before retrieval lets the
post-processing pass apply the right prior over candidate snippets.

Everything here is pure text analysis: no model, no network, no I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class QueryFamily(str, Enum):
    """The answer *shape* a query implies."""

    #: Codeforces / CodeChef / AtCoder style: read stdin, print to stdout.
    COMPETITIVE_STDIN = "competitive_stdin"
    #: LeetCode style: implement a method on ``class Solution``.
    LEETCODE_METHOD = "leetcode_method"
    #: Codewars / kata style: implement a free-standing function.
    KATA_FUNCTION = "kata_function"
    #: No decisive surface evidence.
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------
# Surface cues.  Each is a (regex, weight) pair; weights are relative evidence
# strengths calibrated on the *train* split only (see scripts/tune_reranker.py).
# --------------------------------------------------------------------------

_STDIN_CUES: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"-{3,}\s*Input", re.I), 3.0),
    (re.compile(r"-{3,}\s*Output", re.I), 3.0),
    (re.compile(r"-{3,}\s*Constraints", re.I), 2.0),
    (re.compile(r"-{3,}\s*(Sample|Example)\s*(Input|Output)", re.I), 3.0),
    (re.compile(r"^\s*Input\s*$", re.M), 2.0),
    (re.compile(r"^\s*Output\s*$", re.M), 2.0),
    (re.compile(r"\bthe first line (of the input )?contains\b", re.I), 2.5),
    (re.compile(r"\bthe (next|second|following) lines?\b", re.I), 1.5),
    (re.compile(r"\bprint (a|the|out|one|two|any|\-?1|\"?yes)", re.I), 2.0),
    (re.compile(r"\$[^$\n]{1,60}\$"), 1.0),  # inline LaTeX, very common in CF
    (re.compile(r"\b(test cases?|testcases?)\b", re.I), 1.0),
    (re.compile(r"\bmodulo\s+(10\^\{?9\}?\s*\+\s*7|1000000007)", re.I), 1.0),
)

_LEETCODE_CUES: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"^\s*Example\s*\d+\s*:", re.M), 3.0),
    (re.compile(r"^\s*Constraints\s*:", re.M), 3.0),
    (re.compile(r"\bInput\s*:\s*\w+\s*=", re.I), 3.0),
    (re.compile(r"\bOutput\s*:\s*", re.I), 1.0),
    (re.compile(r"\bExplanation\s*:", re.I), 1.0),
    (re.compile(r"\bgiven an? (integer|string|binary|sorted|0-indexed)", re.I), 1.5),
    (re.compile(r"\bnums\b|\bnums1\b|\bnums2\b"), 2.0),
    (re.compile(r"\breturn the (number|maximum|minimum|length|total|sum|list|array)", re.I), 1.5),
    (re.compile(r"\b10\^\d+\s*<=|\b1 <= \w+\.length"), 2.0),
)

_KATA_CUES: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"```"), 2.5),
    (re.compile(r"\bwrite a (function|method|program)\b", re.I), 3.0),
    (re.compile(r"\byour (function|method|task) (should|must|is to)\b", re.I), 2.5),
    (re.compile(r"\b(create|implement|complete) (a|the) function\b", re.I), 2.5),
    (re.compile(r"\bkata\b", re.I), 3.0),
    (re.compile(r"\bshould return\b", re.I), 2.0),
    (re.compile(r"\bfor example[,:]?\s*```", re.I | re.S), 1.5),
    (re.compile(r"\bHappy coding\b|\bGood luck\b|\benjoy!", re.I), 2.0),
)

_FAMILY_CUES: dict[QueryFamily, tuple[tuple[re.Pattern[str], float], ...]] = {
    QueryFamily.COMPETITIVE_STDIN: _STDIN_CUES,
    QueryFamily.LEETCODE_METHOD: _LEETCODE_CUES,
    QueryFamily.KATA_FUNCTION: _KATA_CUES,
}

#: Minimum absolute evidence before a family is asserted at all.
_MIN_EVIDENCE = 2.0
#: Minimum lead over the runner-up before a family is asserted.
_MIN_MARGIN = 1.0


# --------------------------------------------------------------------------
# Identifier / literal extraction
# --------------------------------------------------------------------------

#: Identifiers written inside markdown code spans or fences.
_CODE_SPAN = re.compile(r"```+([^`]{1,400})```+|`([^`\n]{1,200})`", re.S)
#: A call-like mention, e.g. ``solve(a, b)`` or ``count_squares(lines)``.
_CALL_MENTION = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,40})\s*\(")
#: snake_case or camelCase words that look like code rather than prose.
_CODEY_WORD = re.compile(r"\b(?:[a-z][a-z0-9]*(?:_[a-z0-9]+)+|[a-z]+[A-Z][A-Za-z0-9]*)\b")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_STRING_LIT = re.compile(r"[\"']([^\"'\n]{2,40})[\"']")

#: Prose words that look like identifiers but carry no retrieval signal.
_IDENT_STOPWORDS = frozenset(
    {
        "the", "and", "for", "you", "your", "can", "this", "that", "with", "not",
        "are", "will", "all", "any", "one", "two", "int", "str", "input", "output",
        "print", "return", "function", "example", "examples", "note", "notes",
        "constraints", "explanation", "sample", "test", "tests", "case", "cases",
        "first", "second", "last", "line", "lines", "number", "numbers", "integer",
        "integers", "string", "strings", "array", "arrays", "list", "lists",
    }
)

#: Sections that are pure boilerplate for retrieval purposes.
_BOILERPLATE_SECTIONS = re.compile(
    r"\n-{3,}[ \t]*(?:Notes?|Explanation)[ \t]*:?-*[ \t]*\n.*?(?=\n-{3,}|\Z)",
    re.I | re.S,
)
_TRAILING_CHATTER = re.compile(
    r"\n\s*(Happy coding!?|Good luck!?|Enjoy!?|Please rate.*|If you liked.*)\s*$",
    re.I,
)


@dataclass(frozen=True)
class QueryFacts:
    """Everything the post-processing pass needs to know about one query."""

    family: QueryFamily
    family_scores: dict[str, float]
    #: Confidence in ``family`` in [0, 1]; 0.0 means UNKNOWN.
    confidence: float
    #: Identifiers the statement explicitly names (code spans, call mentions).
    identifiers: frozenset[str] = field(default_factory=frozenset)
    #: Numeric literals that plausibly appear in a solution (limits, answers).
    numbers: frozenset[str] = field(default_factory=frozenset)
    #: Quoted string literals, e.g. ``"YES"`` / ``"Impossible"``.
    strings: frozenset[str] = field(default_factory=frozenset)
    #: Statement with boilerplate sections removed (encoder input candidate).
    cleaned_text: str = ""
    char_length: int = 0

    def is_verified(self) -> bool:
        """``True`` only when a family was asserted on decisive evidence.

        ``UNKNOWN`` is reported honestly rather than guessed: an unlabelled
        query contributes no family prior downstream.
        """
        return self.family is not QueryFamily.UNKNOWN


def _score_families(text: str) -> dict[QueryFamily, float]:
    scores: dict[QueryFamily, float] = {}
    for family, cues in _FAMILY_CUES.items():
        total = 0.0
        for pattern, weight in cues:
            if pattern.search(text):
                total += weight
        scores[family] = total
    return scores


def _extract_identifiers(text: str) -> frozenset[str]:
    found: set[str] = set()
    for match in _CODE_SPAN.finditer(text):
        span = match.group(1) or match.group(2) or ""
        for word in _CODEY_WORD.findall(span):
            found.add(word)
        for name in _CALL_MENTION.findall(span):
            found.add(name)
    for name in _CALL_MENTION.findall(text):
        found.add(name)
    for word in _CODEY_WORD.findall(text):
        found.add(word)
    return frozenset(
        w for w in found if len(w) >= 3 and w.lower() not in _IDENT_STOPWORDS
    )


def _extract_numbers(text: str) -> frozenset[str]:
    out: set[str] = set()
    for raw in _NUMBER.findall(text):
        # Single digits are noise; very long ones are usually constraint bounds
        # that do appear verbatim in solutions (e.g. 1000000007).
        if len(raw.lstrip("-")) >= 2:
            out.add(raw)
    return frozenset(out)


def _extract_strings(text: str) -> frozenset[str]:
    return frozenset(
        s for s in _STRING_LIT.findall(text) if s.strip() and len(s) <= 40
    )


def preprocess_query(text: str) -> str:
    """Strip retrieval-inert boilerplate from a problem statement.

    Removes trailing ``-----Note-----`` / ``Explanation`` sections and kata
    sign-offs.  The result is never empty: if stripping would remove
    everything, the original text is returned unchanged.
    """
    cleaned = _BOILERPLATE_SECTIONS.sub("\n", text)
    cleaned = _TRAILING_CHATTER.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned if len(cleaned) >= 40 else text


def analyze_query(text: str) -> QueryFacts:
    """Categorize and pre-process one natural-language query.

    Deterministic and side-effect free: the same text always yields the same
    facts, which is what lets the embedding cache key on encoder input.
    """
    if not isinstance(text, str):
        text = str(text)

    raw_scores = _score_families(text)
    ordered = sorted(raw_scores.items(), key=lambda kv: kv[1], reverse=True)
    best_family, best_score = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0.0

    if best_score < _MIN_EVIDENCE or (best_score - runner_up) < _MIN_MARGIN:
        family = QueryFamily.UNKNOWN
        confidence = 0.0
    else:
        family = best_family
        # Squash the margin into (0, 1]; 6 points of lead is full confidence.
        confidence = min(1.0, (best_score - runner_up) / 6.0)

    return QueryFacts(
        family=family,
        family_scores={f.value: s for f, s in raw_scores.items()},
        confidence=confidence,
        identifiers=_extract_identifiers(text),
        numbers=_extract_numbers(text),
        strings=_extract_strings(text),
        cleaned_text=preprocess_query(text),
        char_length=len(text),
    )
