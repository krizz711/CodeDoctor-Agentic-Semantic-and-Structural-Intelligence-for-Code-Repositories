"""Execution-based verification of candidate snippets. **Disabled by default.**

Competitive-programming statements publish worked examples.  If a statement's
sample input, fed to a candidate solution, reproduces the sample output, that
candidate is almost certainly the right answer — evidence no embedding can
supply.

Two things make this dangerous rather than merely clever:

1. It executes code that came from the dataset, not from us.  Every run is a
   fresh ``python -I -S`` subprocess in a throwaway directory, capped by a
   wall-clock timeout, given a stripped environment, and address-space capped
   where the platform supports it.  It never runs in-process.

   The limits are applied by a small runner script *inside* the child, after
   ``exec``, never through ``preexec_fn``.  ``preexec_fn`` runs between
   ``fork`` and ``exec`` in a copy of a multi-threaded parent (torch keeps a
   thread pool), which CPython documents as unsafe, and on macOS
   ``setrlimit(RLIMIT_AS)`` is rejected there -- the failure surfaced as an
   uncaught ``SubprocessError`` that aborted the whole evaluation.  In the
   runner a limit the platform refuses is skipped, not fatal.

   Input and output cross the pipe as UTF-8 bytes rather than through the
   parent's locale codec, so a Windows parent (cp1252, ``\r\n``) feeds a
   candidate exactly the bytes a Linux parent would.

   ``-S`` (no ``site``, so no third-party packages) was measured rather than
   assumed: it costs **one** verification out of 120, because only 1.2% of the
   corpus imports a third-party package at all.  Free safety is kept.
2. It is a dataset-shaped signal.  It only fires on statements that carry
   parseable examples, and it must clear the same paired-bootstrap gate as
   every other component before it ships.

The default is ``enabled=False``.  Turning it on is an explicit, recorded
decision — see ``config/frozen.yaml: pipeline.execution_rerank``.
"""

from __future__ import annotations

import ast
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass, field, replace

logger = logging.getLogger(__name__)

#: Layout A -- one ``-----Example-----`` block containing bare ``Input`` /
#: ``Output`` headings (Codeforces, as CoIR ships it).
_EXAMPLE_BLOCK = re.compile(
    r"-{3,}[ \t]*Examples?[ \t]*:?-*[ \t]*\n(.*?)(?=\n-{3,}|\Z)",
    re.I | re.S,
)
#: Note the ``[ \t]`` rather than ``\s``: the heading is followed by exactly
#: one newline, and a greedy ``\s*`` would eat it and then fail to match.
_IO_PAIR = re.compile(
    r"(?:^|\n)[ \t]*Input[ \t]*:?[ \t]*\n(.*?)\n[ \t]*Output[ \t]*:?[ \t]*\n(.*?)"
    r"(?=\n[ \t]*Input[ \t]*:?[ \t]*\n|\Z)",
    re.I | re.S,
)
#: Layout B -- separate ``-----Sample Input-----`` / ``-----Sample Output-----``
#: headings (CodeChef, AtCoder).  The output block runs straight into
#: explanatory prose, so it is cut at the first prose line.
_SAMPLE_INPUT = re.compile(
    r"-{3,}[ \t]*(?:Sample|Example)[ \t]*Input[ \t]*\d*[ \t]*:?-*[ \t]*\n(.*?)(?=\n-{3,}|\Z)", re.I | re.S
)
_SAMPLE_OUTPUT = re.compile(
    r"-{3,}[ \t]*(?:Sample|Example)[ \t]*Output[ \t]*\d*[ \t]*:?-*[ \t]*\n(.*?)(?=\n-{3,}|\Z)", re.I | re.S
)
#: A line of English rather than program output: several alphabetic words in a
#: row.  Sample outputs are numbers, tokens, or short uppercase verdicts.
_PROSE_LINE = re.compile(r"^\s*(?:[A-Za-z][A-Za-z']*\s+){3,}[A-Za-z]")


#: A bare ``Example`` / ``Examples`` heading line (no dashes, no number).
_EXAMPLES_HEADING = re.compile(r"(?:^|\n)[ \t]*Examples?[ \t]*:?[ \t]*\n", re.I)
#: A ``-----Note-----`` style divider, which ends an output block.
_DIVIDER_LINE = re.compile(r"^\s*-{3,}")
#: A bare section heading on its own line (``Note``, ``Explanation``), the way
#: Codeforces pages render once the dashes are gone.  Without this the last
#: example of a CodeContests-style statement swallowed the note that follows.
_HEADING_LINE = re.compile(
    r"^\s*(?:Notes?|Explanations?|Hints?|Constraints|Scoring|Subtasks?)\s*:?\s*$", re.I
)


def _cut_at_prose(block: str) -> str:
    """Keep leading output lines, stop at prose, a heading, or a section divider."""
    kept: list[str] = []
    for line in block.splitlines():
        if _PROSE_LINE.match(line) or _DIVIDER_LINE.match(line) or _HEADING_LINE.match(line):
            break
        kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept).strip()


@dataclass(frozen=True)
class WorkedExample:
    stdin: str
    expected_stdout: str


@dataclass
class ExecutionConfig:
    """Guard rails. Every field is a limit, never a capability."""

    enabled: bool = False
    timeout_seconds: float = 2.0
    #: Only the top-N dense candidates are ever executed.
    max_candidates: int = 5
    #: Only the first N worked examples per query are used.
    max_examples: int = 2
    #: Refuse snippets above this size rather than running them.
    max_code_chars: int = 20_000
    memory_limit_mb: int = 512
    #: Stop reading a candidate's stdout past this many bytes and kill it.
    #: Sample outputs are tiny, so a flood can only ever be a mismatch; the cap
    #: keeps a print loop from holding hundreds of MB in the parent.
    max_output_bytes: int = 1_000_000
    #: Score bonus added to a candidate that reproduced every example.
    verified_bonus: float = 0.5
    #: Candidates are never demoted on failure: a crash may be an environment
    #: difference, not a wrong answer. Evidence promotes, absence proves nothing.
    failure_penalty: float = 0.0
    #: Stop executing a query's candidates once one has verified.
    #:
    #: Every verified candidate in a row receives the same bonus, so verified
    #: candidates keep their dense order among themselves and the FIRST one to
    #: verify is the one that ends up ranked first.  Executing past it cannot
    #: change which document wins, only how many others are lifted above the
    #: unverified remainder.  Off by default because it changes the ranks below
    #: the winner, which is a measurable difference and has to clear the gate.
    stop_after_first_match: bool = False
    #: Refuse to execute a candidate scoring more than this far below the best
    #: candidate in its row.  ``None`` executes the whole top-N.
    #:
    #: Candidates arrive in descending score order, so this is a prefix cut.
    max_dense_gap: float | None = None
    #: Compare numeric output with this relative tolerance instead of as text.
    #:
    #: ``None`` keeps the exact whitespace-insensitive string rule.  When set
    #: it can only turn a mismatch into a match, so it is a pure relaxation --
    #: which means it admits false verifications as readily as it recovers
    #: real ones, and has to be measured on both counts rather than assumed.
    numeric_tolerance: float | None = None
    #: Refuse to *promote* a candidate scoring more than this far below the
    #: best candidate in its row, even when it verified.
    #:
    #: Distinct from ``max_dense_gap``, and the difference is the point.  That
    #: one is a cost lever: the candidate is never run, so nothing is learned
    #: about it.  This one still runs the candidate and still records the
    #: verification -- it only withholds the bonus.  A lookalike deep in the
    #: pool that reproduces the worked examples is therefore counted and
    #: reportable, but cannot displace the answer.
    max_promotion_gap: float | None = None
    #: Candidates verified concurrently.  Each run is its own subprocess, so
    #: threads only wait; results are applied in dense order afterwards, which
    #: keeps the outcome identical to a sequential pass.  Early exit forces a
    #: sequential pass, since running past the first match would change the
    #: audit counts.
    workers: int = 1
    #: ``wall`` (the shipped rule): kill a candidate after ``timeout_seconds``
    #: of wall clock.  ``cpu`` (POSIX only; falls back to ``wall`` elsewhere):
    #: limit *CPU time* to ``timeout_seconds`` and keep a wall-clock backstop of
    #: ``wall_backstop_factor`` times that for programs that sleep or block.
    #: Wall-clock verdicts depend on machine load -- a gold that needs 1.05 s
    #: alone timed out at 2 s under an 8-worker recording -- so a judge's re-run
    #: on a busier machine could score differently; CPU time does not.
    timeout_mode: str = "wall"
    wall_backstop_factor: float = 4.0
    #: Also verify *function-style* candidates (no stdin read; defines a
    #: function or ``class Solution``) against call-style examples --
    #: ``f(x) == y``, LeetCode ``Input: a = .. / Output: ..`` and the like.
    #: Stdin verification is untouched: a candidate that verified before still
    #: does, so this can only add verifications, never remove one.
    call_style: bool = False
    #: Promote a call-style verification only with at least this many examples
    #: and this many distinct expected values: a predicate returning ``True``
    #: or a counter returning ``0`` would otherwise "verify" by coincidence.
    call_min_examples: int = 2
    call_min_distinct: int = 2
    #: Try the stdin layouts the original parser does not read (``Sample
    #: Input 1`` without dashes, CodeContests ``Examples`` blocks) when it
    #: finds nothing.
    extended_stdin: bool = False
    #: POST-7 adaptive deepening: when nothing in the first ``max_candidates``
    #: verifies, also execute candidates ranked up to ``deepen_to`` (still
    #: subject to ``max_dense_gap``).  ``0`` = off.  With the 0.20 gate this
    #: measured +0.0033 (DEV) / +0.0026 (TUNE) NDCG@10 over call-style
    #: verification at half the executions of an ungated top-20.
    deepen_to: int = 0
    #: Skip statements whose sample is a judge interaction transcript.
    skip_interactive: bool = False
    #: Judge outputs by the statement's own rules: its stated float tolerance,
    #: case-insensitive verdicts when it says so, and -- when it accepts any of
    #: several answers -- the same tokens in any order.  Each only relaxes the
    #: comparison, and only for statements that grant it.
    judge_rules: bool = False
    #: When a stdin sample contains blank lines and a candidate's run on it does
    #: not match, run it once more on the sample with the blank lines removed.
    #: Some scraped statements (APPS's CodeChef problems) render every line of a
    #: sample with a blank line after it; the official judge inputs have none,
    #: so a correct program that calls ``input()`` per line crashes on the
    #: sample as shown.  A fallback match counts only when the program read
    #: every non-blank character (when reading is tracked), so a program that
    #: prints a lucky constant without reading the input cannot use it.  Raw
    #: matches are never removed.  Experiment: experiments/C3_cycle (E1).
    blank_line_fallback: bool = False
    #: Run candidates the way a judge does: stdin is a *file* (judges redirect
    #: one), so ``os.fstat(0).st_size`` -- the common fast-IO idiom
    #: ``os.read(0, os.fstat(0).st_size)`` -- sees the real size instead of a
    #: pipe's 0; and ``exit``/``quit``, which ``-S`` removes along with
    #: ``site``, are bound to ``sys.exit`` as CPython's ``site`` would.
    #: Return codes and output match plain CPython (checked by tests).
    #: Experiment: experiments/C3_cycle (E1 arm B; a fidelity fix, no score claim).
    judge_like_io: bool = False
    #: Filled in per statement by :func:`apply_execution_rerank`; not a setting.
    statement_rules: object | None = None
    provenance: dict[str, object] = field(default_factory=dict)


def parse_worked_examples(query_text: str, *, limit: int = 2) -> list[WorkedExample]:
    """Extract ``(stdin, expected stdout)`` pairs from a problem statement.

    Returns an empty list when the statement has no machine-readable examples,
    which is the common case for kata-style queries.
    """
    examples: list[WorkedExample] = []

    block_match = _EXAMPLE_BLOCK.search(query_text)
    if block_match:
        for stdin, stdout in _IO_PAIR.findall(block_match.group(1)):
            stdin = stdin.strip()
            stdout = _cut_at_prose(stdout)
            if stdin and stdout:
                examples.append(WorkedExample(stdin + "\n", stdout))
                if len(examples) >= limit:
                    return examples

    if not examples:
        sample_in = _SAMPLE_INPUT.search(query_text)
        sample_out = _SAMPLE_OUTPUT.search(query_text)
        if sample_in and sample_out:
            stdin = _cut_at_prose(sample_in.group(1))
            stdout = _cut_at_prose(sample_out.group(1))
            if stdin and stdout:
                examples.append(WorkedExample(stdin + "\n", stdout))

    if not examples:
        # Layout C -- bare ``Input`` / ``Output`` headings with no wrapper block.
        # Codeforces-rendered statements also title their *specification*
        # sections with bare ``Input`` / ``Output`` ("The first line
        # contains ..."), before the examples; when an ``Example(s)`` heading
        # exists, only what follows it can be an example.
        heading = _EXAMPLES_HEADING.search(query_text)
        search_text = query_text[heading.end():] if heading else query_text
        for stdin, stdout in _IO_PAIR.findall(search_text):
            stdin = stdin.strip()
            stdout = _cut_at_prose(stdout)
            if stdin and stdout and len(stdin) < 4000:
                examples.append(WorkedExample(stdin + "\n", stdout))
                if len(examples) >= limit:
                    break
    return examples[:limit]


def _normalize_output(text: str) -> str:
    """Whitespace-insensitive comparison, line by line."""
    return "\n".join(line.strip() for line in str(text).strip().splitlines() if line.strip())


def _numeric_match(produced: str, expected: str, rel_tol: float) -> bool:
    """Token-wise comparison that treats numbers as numbers.

    A reference solution that prints ``1.0000000000000002`` against a sample
    showing ``1.000000000`` is correct and fails a string comparison.  This
    compares token by token, falling back to equality for anything that is not
    a number, and requires the same token count -- so it is a *relaxation* of
    the string rule, never a different one.
    """
    produced_tokens, expected_tokens = produced.split(), expected.split()
    if not produced_tokens or len(produced_tokens) != len(expected_tokens):
        return False
    for got, want in zip(produced_tokens, expected_tokens):
        if got == want:
            continue
        try:
            got_value, want_value = float(got), float(want)
        except ValueError:
            return False
        if math.isnan(got_value) or math.isnan(want_value):
            return False
        if not math.isclose(got_value, want_value, rel_tol=rel_tol, abs_tol=1e-9):
            return False
    return True


#: Runs inside the child.  It applies the resource limits the platform accepts,
#: runs the candidate as ``__main__`` exactly as ``python candidate.py`` would
#: (a fresh module swapped into ``sys.modules['__main__']``, as ``runpy`` does),
#: and records what happened in a side-channel file -- never on stdout, which
#: belongs to the candidate.  Optionally it also counts how much of stdin the
#: candidate actually read, which a wrong-problem program usually gets wrong even
#: when a right one cannot match the expected text (multiple valid answers).
#:
#: It imports nothing beyond ``sys`` on the success path: measured on Linux,
#: ``import runpy`` costs 5.7 ms and ``import json`` 4.4 ms per child, against a
#: 7.9 ms interpreter start -- per candidate, per example, tens of thousands of
#: times per evaluation.
_RUNNER_SOURCE = r'''
import sys

def _apply_limits(memory_limit_mb, cpu_seconds=0):
    try:
        import resource
    except ImportError:
        return
    limit = memory_limit_mb * 1024 * 1024
    limits = [("RLIMIT_AS", (limit, limit)), ("RLIMIT_NPROC", (64, 64))]
    if cpu_seconds:
        # CPU time, not wall time: a program that needs 1 s of CPU passes on a
        # loaded machine too.  SIGXCPU at the soft limit, SIGKILL at the hard.
        limits.append(("RLIMIT_CPU", (cpu_seconds, cpu_seconds + 1)))
    for name, value in limits:
        which = getattr(resource, name, None)
        if which is None:
            continue
        try:
            resource.setrlimit(which, value)
        except (ValueError, OSError):
            pass  # refused by this platform (macOS rejects RLIMIT_AS): skip, never fail


class _TrackingStdin:
    """Delegates to the real stdin and counts the characters handed out."""

    def __init__(self, inner, stats):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_stats", stats)

    def _saw(self, text):
        if text:
            self._stats["chars_read"] += len(text)
        else:
            self._stats["eof_hit"] = True
        return text

    def read(self, *args):
        text = self._inner.read(*args)
        if not args or args[0] is None or args[0] < 0:
            self._stats["read_all"] = True
        return self._saw(text)

    def readline(self, *args):
        return self._saw(self._inner.readline(*args))

    def readlines(self, *args):
        lines = self._inner.readlines(*args)
        self._saw("".join(lines))
        self._stats["read_all"] = True
        return lines

    def __iter__(self):
        return self

    def __next__(self):
        line = self._inner.readline()
        if not line:
            self._stats["eof_hit"] = True
            raise StopIteration
        return self._saw(line)

    def __getattr__(self, name):
        if name == "buffer":
            self._stats["tracked"] = False  # byte-level reads bypass the counter
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)


def _defined_here(obj):
    return getattr(obj, "__module__", None) == "__candidate__"


def _arity_ok(func, n, bound):
    code = getattr(func, "__code__", None)
    if code is None:
        return True
    skip = 1 if bound else 0
    positional = code.co_argcount - skip
    required = positional - len(getattr(func, "__defaults__", None) or ())
    has_varargs = bool(code.co_flags & 0x04)
    return required <= n and (n <= positional + code.co_kwonlyargcount or has_varargs)


def _resolve(ns, hint, hints, arity):
    """The callable a call-style example refers to, or (None, reason)."""
    for name in ([hint] if hint else []) + list(hints):
        obj = ns.get(name)
        if callable(obj) and not isinstance(obj, type) and _defined_here(obj):
            return obj, "hint"
    solution = ns.get("Solution")
    if isinstance(solution, type) and _defined_here(solution):
        try:
            instance = solution()
        except BaseException:
            instance = None
        if instance is not None:
            names = [n for n, v in vars(solution).items()
                     if not n.startswith("_") and callable(v)]
            for name in ([hint] if hint else []) + list(hints):
                if name in names:
                    return getattr(instance, name), "solution-hint"
            fitting = [n for n in names if _arity_ok(vars(solution)[n], arity, True)]
            if len(names) == 1 or len(fitting) == 1:
                return getattr(instance, (fitting or names)[0]), "solution"
            if fitting:
                return getattr(instance, fitting[-1]), "solution-ambiguous"
    functions = [v for n, v in ns.items()
                 if not n.startswith("_") and callable(v) and not isinstance(v, type)
                 and _defined_here(v) and getattr(v, "__code__", None) is not None]
    fitting = [f for f in functions if _arity_ok(f, arity, False)]
    if len(fitting) == 1:
        return fitting[0], "function"
    if fitting:
        return fitting[-1], "function-ambiguous"  # helpers first, entry point last
    return None, "unresolved"


def _call_one(ns, hint, hints, args, kwargs):
    target, how = _resolve(ns, hint, hints, len(args) + len(kwargs))
    if target is None:
        return ("unresolved", None, None, how)
    try:
        try:
            result = target(*args, **dict(kwargs))
        except TypeError:
            if not kwargs:
                raise
            # Statement names did not match the parameter names: positional.
            result = target(*args, *[v for _, v in kwargs])
        if hasattr(result, "__next__") and not isinstance(result, (str, bytes)):
            result = list(result)
        mutated = None
        if result is None and (args or kwargs):
            first = args[0] if args else kwargs[0][1]
            mutated = repr(first)[:100000]  # in-place problems return None
        return ("ok", repr(result)[:100000], mutated, how)
    except SystemExit:
        return ("error", "SystemExit", None, how)
    except BaseException as error:
        return ("error", type(error).__name__, None, how)


_PRELUDE_NAMES = {
    "typing": None,  # every public typing name: List, Optional, Dict, ...
    "collections": ("Counter", "defaultdict", "deque", "OrderedDict", "namedtuple", "ChainMap"),
    "heapq": ("heappush", "heappop", "heapify", "heapreplace", "heappushpop", "nlargest", "nsmallest"),
    "bisect": ("bisect", "bisect_left", "bisect_right", "insort", "insort_left", "insort_right"),
    "functools": ("lru_cache", "reduce", "cache", "cmp_to_key", "partial"),
    "itertools": ("accumulate", "combinations", "permutations", "product", "chain", "groupby",
                  "zip_longest", "count", "cycle", "islice", "starmap", "combinations_with_replacement"),
}
_PRELUDE_MODULES = ("collections", "heapq", "bisect", "math", "itertools", "functools",
                    "string", "re", "operator", "random")


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def _install_prelude(ns):
    """What the LeetCode judge pre-imports; solutions are written against it.

    Without it 22% of sampled LeetCode golds died at import on ``List[int]``.
    Names are only *defaults*: anything the candidate defines wins.
    """
    import importlib
    for module_name in _PRELUDE_MODULES:
        try:
            ns.setdefault(module_name, importlib.import_module(module_name))
        except ImportError:
            pass
    for module_name, names in _PRELUDE_NAMES.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for name in (names or getattr(module, "__all__", ())):
            if hasattr(module, name):
                ns.setdefault(name, getattr(module, name))
    ns.setdefault("inf", float("inf"))
    ns.setdefault("ListNode", ListNode)
    ns.setdefault("TreeNode", TreeNode)


def _run_call(script, spec_path, stats, side_path):
    module = type(sys)("__candidate__")
    module.__file__ = script
    module.__builtins__ = sys.modules["builtins"]
    _install_prelude(module.__dict__)
    sys.modules["__candidate__"] = module
    sys.argv = [script]
    results = []
    try:
        with open(script, "rb") as handle:
            source = handle.read()
        exec(compile(source, script, "exec"), module.__dict__)
        with open(spec_path, encoding="utf-8") as handle:
            spec_text = handle.read()
        hints, examples = eval(spec_text, {"__builtins__": {}},
                               {"inf": float("inf"), "nan": float("nan")})
        for hint, args, kwargs in examples:
            results.append(_call_one(module.__dict__, hint, hints, args, kwargs))
    except SystemExit:
        stats["exception"] = "SystemExit"
    except BaseException as error:
        stats["exception"] = type(error).__name__
    finally:
        stats["call_results"] = results
        try:
            with open(side_path, "w", encoding="utf-8") as handle:
                handle.write(repr(stats))
        except Exception:
            pass


def _main():
    limits_arg, track, side_path, script = sys.argv[1:5]
    mode = sys.argv[5] if len(sys.argv) > 5 else "stdin"
    memory_limit_mb, _, rest = limits_arg.partition(":")
    cpu_seconds, _, bind_exit = rest.partition(":")
    _apply_limits(int(memory_limit_mb), int(cpu_seconds or 0))
    if bind_exit == "1":
        import builtins
        builtins.exit = builtins.quit = sys.exit
    if mode == "call":
        _run_call(script, sys.argv[6], {"exception": None}, side_path)
        return
    stats = {"tracked": track == "1", "chars_read": 0, "eof_hit": False,
             "read_all": False, "exception": None}
    if track == "1":
        import builtins
        real_open = builtins.open
        def _open(file, *args, **kwargs):
            if file == 0 or file == "/dev/stdin":
                stats["tracked"] = False
            return real_open(file, *args, **kwargs)
        builtins.open = _open
        sys.stdin = _TrackingStdin(sys.stdin, stats)
    main_module = type(sys)("__main__")
    main_module.__file__ = script
    main_module.__builtins__ = sys.modules["builtins"]
    sys.modules["__main__"] = main_module
    sys.argv = [script]
    try:
        with open(script, "rb") as handle:
            source = handle.read()
        exec(compile(source, script, "exec"), main_module.__dict__)
    except SystemExit:
        raise
    except BaseException as error:
        # Record the class, then let the interpreter's own hook print the
        # traceback and exit 1, exactly as for ``python candidate.py`` -- and
        # without importing ``traceback`` (~10 ms) on every crashing candidate.
        stats["exception"] = type(error).__name__
        raise
    finally:
        try:
            with open(side_path, "w", encoding="utf-8") as handle:
                handle.write(repr(stats))
        except Exception:
            pass

_main()
'''


_RUNNER_LOCK = threading.Lock()
_RUNNER_PATH: str | None = None


def _runner_path() -> str:
    """Write the runner once per process; every run reuses the same file."""
    global _RUNNER_PATH
    with _RUNNER_LOCK:
        if _RUNNER_PATH is None or not os.path.exists(_RUNNER_PATH):
            import atexit
            import shutil

            directory = tempfile.mkdtemp(prefix="codedoctor-runner-")
            path = os.path.join(directory, "_runner.py")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(_RUNNER_SOURCE)
            atexit.register(shutil.rmtree, directory, True)
            _RUNNER_PATH = path
        return _RUNNER_PATH


@dataclass
class ExecutionOutcome:
    """Everything one run of one candidate on one input produced.

    ``status`` is ``completed`` (the process ran to an exit code), ``timeout``,
    ``output_cap``, ``refused`` (never started: too large, unwritable) or
    ``spawn_failed``.  Comparison against an expected output is a separate step
    (:func:`compare_output`), so one recorded run can be judged by any
    comparison rule afterwards.
    """

    status: str
    returncode: int | None = None
    stdout: str = ""
    stderr_tail: str = ""
    exception: str | None = None
    #: Only meaningful when input tracking was requested and stayed tracked.
    stdin_tracked: bool = False
    stdin_chars_read: int | None = None
    stdin_eof_hit: bool | None = None
    #: Call mode only: one ``(status, result_repr, mutated_first_arg_repr,
    #: resolution)`` per example, ``status`` in ``ok|error|unresolved``.
    call_results: list | None = None
    detail: str = ""
    #: Wall-clock seconds from process start to exit (``None`` if it never started).
    seconds: float | None = None

    @property
    def ok(self) -> bool:
        return self.status == "completed" and self.returncode == 0

    def failure_detail(self) -> str:
        """The legacy detail string for a run that did not exit cleanly."""
        if self.detail:
            return self.detail
        if self.status == "timeout":
            return "timeout"
        if self.status == "output_cap":
            return "output cap exceeded"
        suffix = f" ({self.exception})" if self.exception else ""
        return f"exit {self.returncode}{suffix}"


_SPAWN_FAILURES = 0
_SPAWN_LOCK = threading.Lock()


def _record_spawn_failure(error: BaseException) -> None:
    """A sandbox that cannot start processes turns every candidate into a
    silent non-match, i.e. dense-only ranking.  Count it and say so."""
    global _SPAWN_FAILURES
    with _SPAWN_LOCK:
        _SPAWN_FAILURES += 1
    warnings.warn(
        f"execution sandbox could not start a process ({error}); "
        "candidates are not being verified",
        RuntimeWarning,
        stacklevel=3,
    )


def spawn_failures() -> int:
    """Process-wide count of sandbox runs that could not start."""
    with _SPAWN_LOCK:
        return _SPAWN_FAILURES


def _run_capped(
    argv: list[str],
    stdin_bytes: bytes,
    *,
    timeout: float,
    max_output_bytes: int,
    cwd: str,
    env: dict[str, str],
    stdin_path: str | None = None,
) -> tuple[str, int | None, bytes, bytes]:
    """``subprocess.run`` with a byte cap on stdout.

    Returns ``(status, returncode, stdout, stderr)`` with ``status`` one of
    ``completed``, ``timeout`` or ``output_cap``.  Raises only what
    ``Popen`` itself raises when the process cannot be started.  With
    ``stdin_path`` the child reads that file as its stdin (as under a judge's
    redirect) instead of a pipe fed by a thread.
    """
    if stdin_path is not None:
        with open(stdin_path, "rb") as stdin_file:
            process = subprocess.Popen(
                argv, stdin=stdin_file, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=env,
            )
        stdin_bytes = b""
    else:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    overflow = threading.Event()

    def feed() -> None:
        if process.stdin is None:  # stdin is a file (judge-like I/O)
            return
        try:
            if stdin_bytes:
                process.stdin.write(stdin_bytes)
        except OSError:
            pass  # the candidate exited without reading everything
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass

    def pump(stream, sink: list[bytes], limit: int, kill_on_overflow: bool) -> None:
        total = 0
        read = getattr(stream, "read1", stream.read)
        while True:
            try:
                chunk = read(65536)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            if total < limit:
                sink.append(chunk[: limit - total])
            total += len(chunk)
            if total > limit and kill_on_overflow:
                overflow.set()
                try:
                    process.kill()
                except OSError:
                    pass
                break

    threads = [
        threading.Thread(target=feed, daemon=True),
        threading.Thread(
            target=pump, args=(process.stdout, stdout_chunks, max_output_bytes, True), daemon=True
        ),
        threading.Thread(target=pump, args=(process.stderr, stderr_chunks, 65536, False), daemon=True),
    ]
    for thread in threads:
        thread.start()
    status = "completed"
    # Wait on the stdout reader rather than ``Popen.wait(timeout)``: on POSIX
    # the latter polls with sleeps of up to 50 ms, measured at +3.6 ms per run.
    # The reader ends when the child closes stdout, which is almost always
    # when it exits; the bounded ``wait`` below covers the rare child that
    # closes stdout and keeps running.
    deadline = time.monotonic() + timeout
    threads[1].join(timeout)
    try:
        if threads[1].is_alive():
            raise subprocess.TimeoutExpired(argv, timeout)
        process.wait(timeout=max(deadline - time.monotonic(), 0.0))
    except subprocess.TimeoutExpired:
        status = "timeout"
        process.kill()
        process.wait()
    for thread in threads:
        thread.join(timeout=5)
    for stream in (process.stdout, process.stderr):
        try:
            stream.close()
        except OSError:
            pass
    if overflow.is_set():
        status = "output_cap"
    return status, process.returncode, b"".join(stdout_chunks), b"".join(stderr_chunks)


def execute_candidate(
    code: str,
    stdin: str,
    config: ExecutionConfig,
    *,
    track_input: bool = False,
    call_spec: tuple | None = None,
) -> ExecutionOutcome:
    """Run one snippet on one input in an isolated subprocess.

    With ``call_spec = (name_hints, [(hint, args, kwargs), ...])`` the snippet
    is imported as a module instead of run as a script, and each example is a
    function call whose result comes back in ``call_results``.

    Never raises: every failure mode is reported in the returned outcome.
    """
    if len(code) > config.max_code_chars:
        return ExecutionOutcome(status="refused", detail="skipped: snippet exceeds max_code_chars")

    with tempfile.TemporaryDirectory(prefix="codedoctor-exec-") as workdir:
        script = os.path.join(workdir, "candidate.py")
        side_channel = os.path.join(workdir, "_outcome.txt")
        spec_path = os.path.join(workdir, "_call_spec.txt")
        try:
            with open(script, "w", encoding="utf-8") as handle:
                handle.write(code)
            if call_spec is not None:
                with open(spec_path, "w", encoding="utf-8") as handle:
                    handle.write(repr(call_spec))
            runner = _runner_path()
        except (OSError, ValueError) as error:
            return ExecutionOutcome(status="refused", detail=f"write failed: {error}")

        environment = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            # Deliberately empty: no proxy, no credentials, no HOME.
            "HOME": workdir,
            "TMPDIR": workdir,
        }
        if os.name == "nt" and os.environ.get("SYSTEMROOT"):
            # Windows needs it to initialise its crypto/random providers.
            environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
        cpu_mode = config.timeout_mode == "cpu" and os.name == "posix"
        limits_arg = str(config.memory_limit_mb)
        wall_timeout = config.timeout_seconds
        if cpu_mode:
            limits_arg += f":{max(1, math.ceil(config.timeout_seconds))}"
            wall_timeout = config.timeout_seconds * config.wall_backstop_factor
        if config.judge_like_io:
            limits_arg += (":" if cpu_mode else ":0:") + "1"
        argv = [
            sys.executable, "-I", "-S", runner,
            limits_arg, "1" if track_input else "0", side_channel, script,
        ]
        if call_spec is not None:
            argv += ["call", spec_path]
            stdin = ""
        stdin_path = None
        if config.judge_like_io and call_spec is None:
            stdin_path = os.path.join(workdir, "_stdin.txt")
            try:
                with open(stdin_path, "wb") as handle:
                    handle.write(stdin.encode("utf-8"))
            except OSError as error:
                return ExecutionOutcome(status="refused", detail=f"write failed: {error}")
        started = time.perf_counter()
        try:
            status, returncode, out, err = _run_capped(
                argv,
                stdin.encode("utf-8"),
                timeout=wall_timeout,
                max_output_bytes=config.max_output_bytes,
                cwd=workdir,
                env=environment,
                stdin_path=stdin_path,
            )
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            _record_spawn_failure(error)
            return ExecutionOutcome(status="spawn_failed", detail=f"spawn failed: {error}")
        elapsed = time.perf_counter() - started

        stats: dict = {}
        try:
            with open(side_channel, encoding="utf-8") as handle:
                parsed = ast.literal_eval(handle.read())
            if isinstance(parsed, dict):
                stats = parsed
        except (OSError, ValueError, SyntaxError, MemoryError, RecursionError):
            stats = {}

    tracked = bool(track_input and stats.get("tracked"))
    if cpu_mode and status == "completed" and returncode in (-24, -9):
        # killed by SIGXCPU (soft CPU limit) or SIGKILL (hard): a CPU timeout
        status = "timeout"
    return ExecutionOutcome(
        status=status,
        returncode=returncode,
        stdout=out.decode("utf-8", errors="replace"),
        stderr_tail=err.decode("utf-8", errors="replace")[-2000:],
        exception=stats.get("exception"),
        stdin_tracked=tracked,
        stdin_chars_read=stats.get("chars_read") if tracked else None,
        stdin_eof_hit=stats.get("eof_hit") if tracked else None,
        call_results=stats.get("call_results") if call_spec is not None else None,
        seconds=round(elapsed, 4),
    )


def _token_shape(text: str) -> list[list[str]]:
    def kind(token: str) -> str:
        try:
            int(token)
            return "i"
        except ValueError:
            pass
        try:
            float(token)
            return "f"
        except ValueError:
            return "w"

    return [[kind(t) for t in line.split()] for line in text.splitlines() if line.strip()]


def compare_output(produced: str, expected: str, config: ExecutionConfig) -> str | None:
    """The name of the rule under which the output matches, or ``None``.

    Without ``config.statement_rules`` this is the shipped rule: exact after
    whitespace normalisation, then numeric within ``numeric_tolerance``.  With
    the statement's rules (``judge_rules``) three relaxations follow, each only
    when the statement grants it.  The order and conditions mirror
    ``scripts/replay_outcomes.py:stdin_verdict`` so replayed results transfer.
    """
    produced_norm = _normalize_output(produced)
    expected_norm = _normalize_output(expected)
    if produced_norm == expected_norm:
        return "match"
    rules = config.statement_rules
    tolerance = config.numeric_tolerance
    if rules is not None and getattr(rules, "tolerance", None):
        tolerance = max(tolerance or 0.0, float(rules.tolerance))
    if tolerance is not None and _numeric_match(produced_norm, expected_norm, tolerance):
        return "numeric match"
    if rules is None:
        return None
    if getattr(rules, "case_insensitive", False) and produced_norm.lower() == expected_norm.lower():
        return "case-insensitive match"
    if (
        getattr(rules, "any_answer", False)
        and sorted(produced_norm.split()) == sorted(expected_norm.split())
        and _token_shape(produced_norm) == _token_shape(expected_norm)
    ):
        return "any-answer multiset match"
    return None


def run_candidate(
    code: str,
    example: WorkedExample,
    config: ExecutionConfig,
) -> tuple[bool, str]:
    """Run one snippet against one example in an isolated subprocess.

    Returns ``(matched, detail)``.  Any failure mode — syntax error, timeout,
    crash, wrong output — returns ``False`` and is never re-raised.
    """
    outcome = execute_candidate(code, example.stdin, config)
    if not outcome.ok:
        return False, outcome.failure_detail()
    verdict = compare_output(outcome.stdout, example.expected_stdout, config)
    if verdict is None:
        return False, "output mismatch"
    return True, verdict


def has_internal_blank_line(stdin: str) -> bool:
    """A whitespace-only line strictly inside the sample input."""
    return any(not line.strip() for line in stdin.strip("\n").split("\n"))


def collapse_blank_lines(stdin: str) -> str:
    """The sample input with every whitespace-only line removed."""
    return "\n".join(line for line in stdin.split("\n") if line.strip()) + "\n"


def consumed_all(outcome_read: int | None, tracked: bool, stdin: str) -> bool | None:
    """Did the run read every non-blank character of ``stdin``?  ``None`` = untracked."""
    if not tracked:
        return None
    return len(stdin.rstrip()) <= (outcome_read or 0)


def run_blank_line_fallback(code: str, example: WorkedExample, config: ExecutionConfig) -> bool:
    """The E1 fallback run: the collapsed sample, accepted only if fully read."""
    collapsed = collapse_blank_lines(example.stdin)
    outcome = execute_candidate(code, collapsed, config, track_input=True)
    if not outcome.ok or compare_output(outcome.stdout, example.expected_stdout, config) is None:
        return False
    return consumed_all(outcome.stdin_chars_read, outcome.stdin_tracked, collapsed) is not False


def verify_candidate(code: str, examples: list[WorkedExample], config: ExecutionConfig) -> bool:
    """``True`` only when every supplied example is reproduced exactly."""
    if not examples:
        return False
    for example in examples:
        matched, _ = run_candidate(code, example, config)
        if (
            not matched
            and config.blank_line_fallback
            and has_internal_blank_line(example.stdin)
        ):
            matched = run_blank_line_fallback(code, example, config)
        if not matched:
            return False
    return True


_READS_STDIN = re.compile(r"\binput\s*\(|sys\.stdin|stdin\.read|open\(\s*0\s*[,)]|/dev/stdin")
_DEFINES = re.compile(r"^\s*(?:def|class)\s+[A-Za-z_]", re.M)


def is_function_style(code: str) -> bool:
    """Defines something callable and never reads stdin: a call-style solution."""
    return bool(_DEFINES.search(code)) and not _READS_STDIN.search(code)


@dataclass
class CallVerification:
    passed: int
    total: int
    details: list[str]

    @property
    def verified(self) -> bool:
        return self.total > 0 and self.passed == self.total


def verify_call_candidate(code: str, facts, config: ExecutionConfig, *, name_hints=()) -> CallVerification:
    """Run a function-style candidate on a statement's call-style examples.

    ``facts`` is a :class:`app.pipeline.statement.StatementFacts`.  All examples
    run in one child; each result is compared structurally
    (:func:`app.pipeline.statement.values_equal`) under the statement's own
    judging rules.
    """
    from app.pipeline.statement import values_equal

    examples = list(facts.call_examples)
    if not examples:
        return CallVerification(0, 0, ["no call examples"])
    spec = (
        tuple(name_hints),
        [(e.callee, tuple(e.args), tuple(e.kwargs)) for e in examples],
    )
    outcome = execute_candidate(code, "", config, call_spec=spec)
    if outcome.status != "completed" or outcome.call_results is None:
        return CallVerification(0, len(examples), [outcome.failure_detail()])
    passed = 0
    details: list[str] = []
    tolerance = facts.rules.tolerance if facts.rules.tolerance is not None else 1e-9
    for example, result in zip(examples, outcome.call_results):
        status, value_repr, mutated_repr, how = (list(result) + [None] * 4)[:4]
        if status != "ok":
            details.append(f"{status}:{value_repr or how}")
            continue
        matched = False
        for candidate_repr in (value_repr, mutated_repr if value_repr == "None" else None):
            if candidate_repr is None:
                continue
            try:
                value = ast.literal_eval(candidate_repr)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                continue
            if values_equal(
                value, example.expected, tolerance=tolerance, case_insensitive=facts.rules.case_insensitive
            ):
                matched = True
                break
        passed += int(matched)
        details.append(f"{'match' if matched else 'mismatch'}:{how}")
    missing = len(examples) - len(outcome.call_results)
    details.extend(["not run"] * max(0, missing))
    return CallVerification(passed, len(examples), details)


def _identity_examples(facts) -> bool:
    """Every example returns its single argument unchanged.

    Such a set cannot tell the task apart from ``lambda x: x``; it is almost
    always explanation prose misread as examples (``23 => 23 * 1 = 23``).
    """
    examples = list(facts.call_examples)
    if not examples:
        return False
    for example in examples:
        values = list(example.args) + [v for _, v in example.kwargs]
        if len(values) != 1 or repr(values[0]) != repr(example.expected):
            return False
    return True


def call_verification_promotable(result: CallVerification, facts, config: ExecutionConfig) -> bool:
    """The low-entropy guard: enough examples, enough distinct answers, not identity."""
    return (
        result.verified
        and result.total >= config.call_min_examples
        and facts.distinct_expected >= config.call_min_distinct
        and not _identity_examples(facts)
    )


#: An interactive statement's sample is a judge transcript, not stdin.
_INTERACTIVE = re.compile(r"interaction with the judge|this is an interactive (?:problem|task)", re.I)


def _statement_examples(query_text: str, config: ExecutionConfig):
    """``(stdin_examples, call_facts)`` for a statement under ``config``.

    With the new flags off this is exactly ``parse_worked_examples``.
    """
    if config.skip_interactive and _INTERACTIVE.search(query_text):
        return [], None
    stdin_examples = parse_worked_examples(query_text, limit=config.max_examples)
    if not (config.call_style or config.extended_stdin):
        return stdin_examples, None
    from app.pipeline.statement import analyze_statement, parse_extra_stdin_examples

    if not stdin_examples and config.extended_stdin:
        stdin_examples = parse_extra_stdin_examples(query_text, limit=config.max_examples)
    facts = None
    if config.call_style:
        from app.pipeline.query_analysis import QueryFamily, analyze_query
        from app.pipeline.statement import _CALL_CUES, parse_call_examples

        # Only statements that look call-style get call examples: "F(3) = 2"
        # in a Codeforces statement is mathematics, not a test case.
        family = analyze_query(query_text).family
        call_like = family in (QueryFamily.LEETCODE_METHOD, QueryFamily.KATA_FUNCTION) or bool(
            _CALL_CUES.search(query_text)
        )
        if call_like:
            base = analyze_statement(query_text, max_examples=config.max_examples)
            calls = list(base.call_examples) or parse_call_examples(query_text)
            if calls:
                facts = replace(
                    base,
                    mode="call",
                    call_examples=tuple(calls),
                    distinct_expected=len({c.expected_repr() for c in calls}),
                )
    return stdin_examples, facts


def has_executable_examples(query_text: str, config: ExecutionConfig) -> bool:
    stdin_examples, facts = _statement_examples(query_text, config)
    return bool(stdin_examples) or facts is not None


def apply_execution_rerank(
    query_text: str,
    candidate_texts: list[str],
    candidate_scores: list[float],
    config: ExecutionConfig,
) -> tuple[list[float], dict[str, object]]:
    """Promote candidates that reproduce the statement's worked examples.

    Returns the adjusted scores and an audit record.  With ``enabled=False``
    the scores are returned untouched and the record says why.
    """
    audit: dict[str, object] = {
        "enabled": config.enabled,
        "examples_found": 0,
        "candidates_executed": 0,
        "candidates_verified": 0,
        "per_result": [{"state": "not_run"} for _ in candidate_texts],
    }
    if not config.enabled:
        audit["reason"] = "execution rerank disabled by configuration"
        return list(candidate_scores), audit

    examples, call_facts = _statement_examples(query_text, config)
    if config.judge_rules and config.statement_rules is None:
        from app.pipeline.statement import parse_judge_rules

        config = replace(config, statement_rules=parse_judge_rules(query_text))
    audit["examples_found"] = len(examples)
    if call_facts is not None:
        audit["call_examples_found"] = len(call_facts.call_examples)
        audit["call_verified"] = 0
    if not examples and call_facts is None:
        audit["reason"] = "statement has no machine-readable worked examples"
        return list(candidate_scores), audit

    scores = list(candidate_scores)
    audit["candidates_skipped_by_gate"] = 0
    audit["candidates_promoted"] = 0
    audit["promotions_blocked"] = 0
    audit["stopped_early"] = False

    best = scores[0] if scores else 0.0
    limit = min(config.max_candidates, len(candidate_texts))
    runnable = limit
    if config.max_dense_gap is not None:
        # Descending order, so the gate is a prefix cut.
        runnable = next(
            (i for i in range(limit) if best - scores[i] > config.max_dense_gap), limit
        )
    deep_limit = min(max(config.deepen_to, 0), len(candidate_texts))
    deep_runnable = deep_limit
    if config.max_dense_gap is not None and deep_limit > limit:
        deep_runnable = next(
            (i for i in range(limit, deep_limit) if best - scores[i] > config.max_dense_gap), deep_limit
        )

    def check(i: int) -> tuple[bool, bool]:
        """``(passed, by_call)`` for candidate ``i``.

        Stdin first, exactly as before; a function-style candidate that did not
        verify on stdin (it cannot: it never reads input) then gets its
        call-style examples.  A strict superset of the old verifications.
        """
        text = candidate_texts[i]
        if examples and verify_candidate(text, examples, config):
            return True, False
        if call_facts is not None and is_function_style(text):
            result = verify_call_candidate(text, call_facts, config)
            return call_verification_promotable(result, call_facts, config), True
        return False, False

    precomputed: list[tuple[bool, bool]] | None = None
    if config.workers > 1 and not config.stop_after_first_match and runnable > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(config.workers, runnable)) as pool:
            precomputed = list(pool.map(check, range(runnable)))

    for index in range(limit):
        if index >= runnable:
            audit["candidates_skipped_by_gate"] = limit - index
            break
        audit["candidates_executed"] = int(audit["candidates_executed"]) + 1
        passed, by_call = precomputed[index] if precomputed is not None else check(index)
        audit["per_result"][index] = {"state": "verified" if passed else "failed", "by_call": by_call}
        if passed:
            audit["candidates_verified"] = int(audit["candidates_verified"]) + 1
            if by_call:
                audit["call_verified"] = int(audit.get("call_verified", 0)) + 1
            if (
                config.max_promotion_gap is not None
                and best - scores[index] > config.max_promotion_gap
            ):
                # Verified, and deliberately not promoted. Counted either way:
                # the whole point is to be able to report how often this fires.
                audit["promotions_blocked"] = int(audit.get("promotions_blocked", 0)) + 1
                continue
            scores[index] += config.verified_bonus
            audit["candidates_promoted"] = int(audit.get("candidates_promoted", 0)) + 1
            if config.stop_after_first_match:
                audit["stopped_early"] = True
                break
        else:
            scores[index] += config.failure_penalty

    if deep_runnable > limit and int(audit.get("candidates_promoted", 0)) == 0:
        # Nothing in the first window: look further down, where the dense pass
        # left golds it ranked below 20 -- but never past the dense gate.
        audit["deepened"] = deep_runnable - limit
        deep_results = [check(i) for i in range(limit, deep_runnable)] if config.workers <= 1 else None
        if deep_results is None:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(config.workers, deep_runnable - limit)) as pool:
                deep_results = list(pool.map(check, range(limit, deep_runnable)))
        for offset, (passed, by_call) in enumerate(deep_results):
            index = limit + offset
            audit["candidates_executed"] = int(audit["candidates_executed"]) + 1
            audit["per_result"][index] = {"state": "verified" if passed else "failed", "by_call": by_call}
            if passed:
                audit["candidates_verified"] = int(audit["candidates_verified"]) + 1
                if by_call:
                    audit["call_verified"] = int(audit.get("call_verified", 0)) + 1
                scores[index] += config.verified_bonus
                audit["candidates_promoted"] = int(audit.get("candidates_promoted", 0)) + 1
    return scores, audit
