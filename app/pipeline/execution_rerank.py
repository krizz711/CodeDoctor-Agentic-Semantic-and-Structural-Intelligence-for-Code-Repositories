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

import logging
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

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


#: A ``-----Note-----`` style divider, which ends an output block.
_DIVIDER_LINE = re.compile(r"^\s*-{3,}")


def _cut_at_prose(block: str) -> str:
    """Keep leading output lines, stop at prose or the next section divider."""
    kept: list[str] = []
    for line in block.splitlines():
        if _PROSE_LINE.match(line) or _DIVIDER_LINE.match(line):
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
        for stdin, stdout in _IO_PAIR.findall(query_text):
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


def _preexec_limits(memory_limit_mb: int):
    """Return a POSIX ``preexec_fn`` applying an address-space cap, or None."""
    if os.name != "posix":
        return None
    try:
        import resource
    except ImportError:  # pragma: no cover - platform dependent
        return None

    limit = memory_limit_mb * 1024 * 1024

    def _apply() -> None:  # pragma: no cover - runs in the child process
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))

    return _apply


def run_candidate(
    code: str,
    example: WorkedExample,
    config: ExecutionConfig,
) -> tuple[bool, str]:
    """Run one snippet against one example in an isolated subprocess.

    Returns ``(matched, detail)``.  Any failure mode — syntax error, timeout,
    crash, wrong output — returns ``False`` and is never re-raised.
    """
    if len(code) > config.max_code_chars:
        return False, "skipped: snippet exceeds max_code_chars"

    with tempfile.TemporaryDirectory(prefix="codedoctor-exec-") as workdir:
        script = os.path.join(workdir, "candidate.py")
        try:
            with open(script, "w", encoding="utf-8") as handle:
                handle.write(code)
        except OSError as error:
            return False, f"write failed: {error}"

        environment = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            # Deliberately empty: no proxy, no credentials, no HOME.
            "HOME": workdir,
            "TMPDIR": workdir,
        }
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", script],
                input=example.stdin,
                text=True,
                capture_output=True,
                timeout=config.timeout_seconds,
                cwd=workdir,
                env=environment,
                preexec_fn=_preexec_limits(config.memory_limit_mb),
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except (OSError, ValueError) as error:
            return False, f"spawn failed: {error}"

        if completed.returncode != 0:
            return False, f"exit {completed.returncode}"
        produced = _normalize_output(completed.stdout)
        expected = _normalize_output(example.expected_stdout)
        if produced == expected:
            return True, "match"
        if config.numeric_tolerance is not None and _numeric_match(
            produced, expected, config.numeric_tolerance
        ):
            return True, "numeric match"
        return False, "output mismatch"


def verify_candidate(code: str, examples: list[WorkedExample], config: ExecutionConfig) -> bool:
    """``True`` only when every supplied example is reproduced exactly."""
    if not examples:
        return False
    for example in examples:
        matched, _ = run_candidate(code, example, config)
        if not matched:
            return False
    return True


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
    }
    if not config.enabled:
        audit["reason"] = "execution rerank disabled by configuration"
        return list(candidate_scores), audit

    examples = parse_worked_examples(query_text, limit=config.max_examples)
    audit["examples_found"] = len(examples)
    if not examples:
        audit["reason"] = "statement has no machine-readable worked examples"
        return list(candidate_scores), audit

    scores = list(candidate_scores)
    audit["candidates_skipped_by_gate"] = 0
    audit["candidates_promoted"] = 0
    audit["promotions_blocked"] = 0
    audit["stopped_early"] = False

    best = scores[0] if scores else 0.0
    for index in range(min(config.max_candidates, len(candidate_texts))):
        if config.max_dense_gap is not None and best - scores[index] > config.max_dense_gap:
            # Descending order, so everything after this is further away too.
            audit["candidates_skipped_by_gate"] = (
                min(config.max_candidates, len(candidate_texts)) - index
            )
            break
        audit["candidates_executed"] = int(audit["candidates_executed"]) + 1
        if verify_candidate(candidate_texts[index], examples, config):
            audit["candidates_verified"] = int(audit["candidates_verified"]) + 1
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
    return scores, audit
