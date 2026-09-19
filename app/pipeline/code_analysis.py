"""Code-snippet categorization and pre/post-processing (Theme 01 axes 3 and 4).

A dense encoder sees a code snippet as one undifferentiated string.  The
retrieval-relevant structure of a Python solution — whether it reads stdin,
what it defines, which literals it hard-codes — is cheap to recover with the
standard library and is exactly the evidence the post-processing pass needs.

Parsing uses :mod:`ast` first and falls back to regex when the snippet does not
parse (the APPS corpus contains Python 2 fragments and truncated solutions).
The fallback is *labelled*, never silently equated with a real parse.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache


class CodeFamily(str, Enum):
    """The answer *shape* a snippet provides."""

    #: Reads stdin and prints — the competitive-programming submission shape.
    STDIN_SCRIPT = "stdin_script"
    #: Defines ``class Solution`` — the LeetCode shape.
    CLASS_SOLUTION = "class_solution"
    #: Defines free-standing callables and never touches stdin — the kata shape.
    FUNCTION_LIBRARY = "function_library"
    #: Parses, but matches none of the above decisively.
    UNKNOWN = "unknown"


_STDIN_CALL = re.compile(r"\b(?:input|raw_input)\s*\(|\bsys\.stdin\b|\bstdin\.readline\b")
_PRINT_CALL = re.compile(r"\bprint\s*\(|\bsys\.stdout\b")
_CLASS_SOLUTION = re.compile(r"^\s*class\s+Solution\b", re.M)
_DEF = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)
_IDENT = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,40})\b")
_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])(-?\d+(?:\.\d+)?)")
_STRING_LIT = re.compile(r"[\"']([^\"'\n]{1,40})[\"']")

#: Python builtins and boilerplate that carry no discriminative signal.
_IDENT_STOPWORDS = frozenset(
    {
        "self", "len", "range", "int", "str", "list", "dict", "set", "tuple",
        "for", "while", "not", "and", "the", "return", "print", "input", "map",
        "sum", "min", "max", "abs", "sorted", "append", "join", "split", "else",
        "elif", "def", "class", "import", "from", "None", "True", "False",
        "res", "ans", "out", "tmp", "val", "idx", "cnt", "num", "arr", "lst",
    }
)


@dataclass(frozen=True)
class CodeFacts:
    """Structural facts about one code snippet."""

    family: CodeFamily
    #: ``True`` when :mod:`ast` parsed the snippet; ``False`` for regex fallback.
    parsed: bool
    #: Names bound by ``def`` / ``class`` statements at any nesting depth.
    defined_names: frozenset[str] = field(default_factory=frozenset)
    #: Top-level module names reached by ``import`` / ``from``.
    imports: frozenset[str] = field(default_factory=frozenset)
    #: Every identifier occurring in the snippet, minus stopwords.
    identifiers: frozenset[str] = field(default_factory=frozenset)
    numbers: frozenset[str] = field(default_factory=frozenset)
    strings: frozenset[str] = field(default_factory=frozenset)
    reads_stdin: bool = False
    writes_stdout: bool = False
    char_length: int = 0
    line_count: int = 0

    def evidence(self) -> str:
        """``"VERIFIED"`` for an AST parse, ``"CANDIDATE"`` for the fallback.

        Downstream consumers must not treat a regex-derived family as equal to
        a parsed one; an unparsed snippet is *unknown*, not *verified*.
        """
        return "VERIFIED" if self.parsed else "CANDIDATE"


def _ast_facts(tree: ast.AST) -> tuple[set[str], set[str], set[str], set[str]]:
    defined: set[str] = set()
    imports: set[str] = set()
    numbers: set[str] = set()
    strings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                text = repr(value)
                if len(text.lstrip("-")) >= 2:
                    numbers.add(text)
            elif isinstance(value, str) and 1 <= len(value) <= 40:
                strings.add(value)
    return defined, imports, numbers, strings


def _classify(
    *,
    reads_stdin: bool,
    writes_stdout: bool,
    has_class_solution: bool,
    defined: set[str],
) -> CodeFamily:
    if has_class_solution:
        return CodeFamily.CLASS_SOLUTION
    if reads_stdin:
        return CodeFamily.STDIN_SCRIPT
    if defined and not reads_stdin:
        return CodeFamily.FUNCTION_LIBRARY
    if writes_stdout:
        # `print(25)` with no input read is still a submission-shaped answer.
        return CodeFamily.STDIN_SCRIPT
    return CodeFamily.UNKNOWN


@lru_cache(maxsize=32768)
def analyze_code(text: str) -> CodeFacts:
    """Extract structural facts from one Python snippet.

    Cached on the snippet text: identical content always produces identical
    facts, which keeps the pipeline deterministic across index rebuilds.
    """
    if not isinstance(text, str):
        text = str(text)

    reads_stdin = bool(_STDIN_CALL.search(text))
    writes_stdout = bool(_PRINT_CALL.search(text))
    has_class_solution = bool(_CLASS_SOLUTION.search(text))

    parsed = True
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        parsed = False
        defined = set(_DEF.findall(text))
        imports = set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.M))
        numbers = {n for n in _NUMBER.findall(text) if len(n.lstrip("-")) >= 2}
        strings = {s for s in _STRING_LIT.findall(text) if s.strip()}
    else:
        defined, imports, numbers, strings = _ast_facts(tree)

    identifiers = {
        token
        for token in _IDENT.findall(text)
        if token not in _IDENT_STOPWORDS and not token.isdigit()
    }

    return CodeFacts(
        family=_classify(
            reads_stdin=reads_stdin,
            writes_stdout=writes_stdout,
            has_class_solution=has_class_solution,
            defined=defined,
        ),
        parsed=parsed,
        defined_names=frozenset(defined),
        imports=frozenset(imports),
        identifiers=frozenset(identifiers),
        numbers=frozenset(numbers),
        strings=frozenset(strings),
        reads_stdin=reads_stdin,
        writes_stdout=writes_stdout,
        char_length=len(text),
        line_count=text.count("\n") + 1,
    )
