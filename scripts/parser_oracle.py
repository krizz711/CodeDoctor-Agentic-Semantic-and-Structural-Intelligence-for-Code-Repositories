"""V3-oracle: score the statement parser against labels its author did not write.

Pre-registration: ``experiments/V3_parser_gold/prereg_oracle.md``.

    python scripts/parser_oracle.py --out experiments/V3_parser_gold/result_oracle.json

Two oracles, both built by other people for other purposes:

* **O1, APPS train** -- ``input_output`` holds each problem's official tests
  (stdin strings, or ``fn_name`` + argument lists).  Statement examples are
  usually, not always, among them, so agreement is a *lower bound* on
  precision.
* **O2, CodeContests train** -- ``public_tests`` holds exactly the sample tests
  a statement shows, recorded separately from its text.  Statements that
  near-duplicate an APPS statement are dropped.

Inputs (downloaded once, gitignored): ``artifacts/oracle/apps_train.jsonl``
(``codeparrot/apps``, sha256 45e82ef2...) and
``artifacts/oracle/cc_train_statements.parquet`` (``deepmind/code_contests``
partial train conversion; columns name, description, public_tests, source).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.execution_rerank import (  # noqa: E402
    ExecutionConfig,
    _normalize_output,
    _statement_examples,
)
from app.pipeline.query_analysis import analyze_query  # noqa: E402
from app.pipeline.statement import values_equal  # noqa: E402
from scripts.build_external_set import _shingles  # noqa: E402

SEED = 20260924
PARSER_FILES = (
    "app/pipeline/statement.py",
    "app/pipeline/execution_rerank.py",
    "app/pipeline/query_analysis.py",
)
#: What the v2 candidate runs: call-style on, extended stdin off, 2 examples.
PRIMARY = ExecutionConfig(enabled=True, call_style=True, extended_stdin=False)
#: The parser without the cap, for context only.
UNCAPPED = replace(PRIMARY, extended_stdin=True, max_examples=10)
CALL_TOLERANCE = 1e-6
CC_SOURCES = {0: "unknown", 1: "codechef", 2: "codeforces", 3: "hackerearth",
              4: "codejam", 5: "atcoder", 6: "aizu"}


# ------------------------------------------------------------------ matching


def norm_lines(text: Any) -> list[str]:
    """The verifier's normalisation (``_normalize_output``), as a line list."""
    normalized = _normalize_output(text)
    return normalized.split("\n") if normalized else []


def tokens(text: str) -> tuple[str, ...]:
    """Whitespace-free view of a text; a diagnostic, not the pre-registered rule."""
    return tuple(str(text).split())


def find_run(haystack: list[str], needle: list[str], start: int = 0) -> int:
    """Index of ``needle`` as a contiguous run in ``haystack`` at or after ``start``."""
    n = len(needle)
    if n == 0:
        return -1
    for i in range(start, len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return -1


def is_visible(statement_lines: list[str], test_input: str, test_output: str) -> bool:
    """A test is shown in a statement when its input lines and then its output
    lines each appear as contiguous runs of statement lines."""
    a, b = norm_lines(test_input), norm_lines(test_output)
    if not a or not b:
        return False
    i = find_run(statement_lines, a)
    while i >= 0:
        if find_run(statement_lines, b, i + len(a)) >= 0:
            return True
        i = find_run(statement_lines, a, i + 1)
    return False


def as_text(value: Any) -> str:
    """APPS stores some stdin tests as lists of lines."""
    if isinstance(value, list):
        return "\n".join(str(v) for v in value)
    return str(value)


def call_matches(args: tuple, expected: Any, official_args: Any, official_out: Any) -> bool:
    """An extracted call example equals an official test.  APPS wraps many
    Codewars results in a one-element list, so both forms are accepted."""
    if not values_equal(list(args), list(official_args) if isinstance(official_args, (list, tuple)) else [official_args],
                        tolerance=CALL_TOLERANCE):
        return False
    if values_equal(expected, official_out, tolerance=CALL_TOLERANCE):
        return True
    return (
        isinstance(official_out, list)
        and len(official_out) == 1
        and values_equal(expected, official_out[0], tolerance=CALL_TOLERANCE)
    )


def half_of(problem_id: str, seed: int = SEED) -> str:
    digest = hashlib.sha256(f"{seed}:{problem_id}".encode()).digest()
    return "CONF" if digest[0] % 2 else "DIAG"


# ------------------------------------------------------------------- scoring


@dataclass
class Row:
    """One statement's counts; every metric is a ratio of sums over rows."""

    set_name: str
    problem_id: str
    group: str
    half: str
    mode: str  # official mode: stdin | call
    extracted: int = 0  # stdin pairs, or call examples, the parser produced
    confirmed: int = 0  # of those, equal to an official / public test
    confirmed_tokens: int = 0  # diagnostic: equal after collapsing all whitespace
    visible: int = 0  # official tests shown in the statement (stdin only)
    visible_capped: int = 0  # min(visible, cap)
    found: int = 0  # visible tests the parser extracted (<= cap)
    wrong_mode: int = 0  # stdin extracted for a call problem, or vice versa
    exact_set: int = 0  # stdin: extracted set == visible set (capped)
    unconfirmed: list = field(default_factory=list)
    missed: list = field(default_factory=list)


def score_stdin(row: Row, statement: str, tests: list[tuple[str, str]], config: ExecutionConfig) -> Row:
    stdin_examples, facts = _statement_examples(statement, config)
    official = {(_normalize_output(i), _normalize_output(o)) for i, o in tests}
    extracted = [(_normalize_output(e.stdin), _normalize_output(e.expected_stdout)) for e in stdin_examples]
    lines = norm_lines(statement)
    visible = [t for t in official if is_visible(lines, *t)]
    row.extracted = len(extracted)
    row.confirmed = sum(pair in official for pair in extracted)
    official_tokens = {(tokens(i), tokens(o)) for i, o in official}
    row.confirmed_tokens = sum((tokens(i), tokens(o)) in official_tokens for i, o in extracted)
    row.visible = len(visible)
    row.visible_capped = min(len(visible), config.max_examples)
    row.found = min(sum(t in set(extracted) for t in visible), config.max_examples)
    row.wrong_mode = int(facts is not None and bool(facts.call_examples))
    row.exact_set = int(bool(visible) and row.found == row.visible_capped == row.extracted == row.confirmed)
    row.unconfirmed = [p for p in extracted if p not in official]
    row.missed = [t for t in visible if t not in set(extracted)]
    return row


def score_call(row: Row, statement: str, official: list[tuple[Any, Any]], config: ExecutionConfig) -> Row:
    stdin_examples, facts = _statement_examples(statement, config)
    calls = list(facts.call_examples) if facts is not None else []
    row.extracted = len(calls)
    for c in calls:
        flat = tuple(c.args) + tuple(v for _, v in c.kwargs)
        if any(call_matches(flat, c.expected, a, o) for a, o in official):
            row.confirmed += 1
        else:
            row.unconfirmed.append((repr(flat)[:200], repr(c.expected)[:200]))
    row.wrong_mode = int(bool(stdin_examples))
    return row


def ratio_ci(rows: list[Row], num: str, den: str, *, resamples: int = 2000, seed: int = SEED) -> dict:
    n = np.array([getattr(r, num) for r in rows], dtype=float)
    d = np.array([getattr(r, den) for r in rows], dtype=float)
    if d.sum() == 0:
        return {"value": None, "num": int(n.sum()), "den": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rows), size=(resamples, len(rows)))
    boot = n[idx].sum(1) / np.maximum(d[idx].sum(1), 1e-12)
    return {
        "value": round(float(n.sum() / d.sum()), 4),
        "ci95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
        "num": int(n.sum()),
        "den": int(d.sum()),
    }


def summarise(rows: list[Row]) -> dict:
    stdin = [r for r in rows if r.mode == "stdin"]
    call = [r for r in rows if r.mode == "call"]
    out = {"statements": len(rows), "stdin_statements": len(stdin), "call_statements": len(call)}
    if stdin:
        out["stdin"] = {
            "precision": ratio_ci(stdin, "confirmed", "extracted"),
            "precision_token_level_diagnostic": ratio_ci(stdin, "confirmed_tokens", "extracted"),
            "recall_capped": ratio_ci(stdin, "found", "visible_capped"),
            "statements_with_visible_tests": sum(r.visible > 0 for r in stdin),
            "nothing_extracted_but_visible": sum(r.visible > 0 and r.extracted == 0 for r in stdin),
            "exact_set_rate": round(
                sum(r.exact_set for r in stdin) / max(1, sum(r.visible > 0 for r in stdin)), 4
            ),
            "call_examples_on_stdin_problems": sum(r.wrong_mode for r in stdin),
        }
    if call:
        out["call"] = {
            "precision": ratio_ci(call, "confirmed", "extracted"),
            "coverage_any_confirmed": round(sum(r.confirmed > 0 for r in call) / len(call), 4),
            "coverage_any_extracted": round(sum(r.extracted > 0 for r in call) / len(call), 4),
            "stdin_examples_on_call_problems": sum(r.wrong_mode for r in call),
        }
    return out


def by_group(rows: list[Row], minimum: int = 30) -> dict:
    groups: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        groups[r.group].append(r)
    return {g: summarise(rs) for g, rs in sorted(groups.items()) if len(rs) >= minimum}


def audit(rows: list[Row], attribute: str, k: int = 30, seed: int = SEED) -> list[dict]:
    pool = [(r.problem_id, item) for r in rows for item in getattr(r, attribute)]
    random.Random(seed).shuffle(pool)
    return [{"id": pid, "input_or_args": str(a)[:300], "output_or_expected": str(b)[:300]}
            for pid, (a, b) in pool[:k]]


# ------------------------------------------------------------------- oracles


def load_apps(path: Path, coir_queries: Path) -> list[dict]:
    sys.set_int_max_str_digits(0)  # APPS holds a 9,131-digit test value
    by_text: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            by_text.setdefault(" ".join(r["question"].split()), r)
    problems = []
    with coir_queries.open(encoding="utf-8") as handle:
        for line in handle:
            q = json.loads(line)
            r = by_text.get(" ".join(q["text"].split()))
            if r is None or not r["input_output"]:
                continue
            io = json.loads(r["input_output"])
            problems.append({"id": q["id"], "statement": q["text"], "io": io,
                             "group": analyze_query(q["text"]).family.value})
    return problems


def load_codecontests(path: Path, apps_statements: list[str], threshold: float = 0.5) -> tuple[list[dict], int]:
    import pandas as pd

    frame = pd.read_parquet(path)
    index: dict[int, list[int]] = defaultdict(list)
    apps_sizes = []
    for k, text in enumerate(apps_statements):
        shingles = _shingles(text)
        apps_sizes.append(len(shingles))
        for s in shingles:
            index[s].append(k)
    problems, dropped = [], 0
    for _, row in frame.iterrows():
        statement = row["description"] or ""
        tests = row["public_tests"]
        if not statement.strip() or tests is None:
            continue
        mine = _shingles(statement)
        overlap: Counter = Counter()
        for s in mine:
            for k in index.get(s, ()):
                overlap[k] += 1
        best = max((c / (len(mine) + apps_sizes[k] - c) for k, c in overlap.items()), default=0.0)
        if best >= threshold:
            dropped += 1
            continue
        pairs = list(zip(list(tests["input"]), list(tests["output"])))
        problems.append({"id": str(row["name"]), "statement": statement, "tests": pairs,
                         "group": CC_SOURCES.get(int(row["source"]), str(row["source"]))})
    return problems, dropped


def evaluate(problems: list[dict], set_name: str, config: ExecutionConfig) -> list[Row]:
    rows = []
    for p in problems:
        half = half_of(p["id"])
        if set_name == "O2":
            row = Row(set_name, p["id"], p["group"], half, "stdin")
            rows.append(score_stdin(row, p["statement"], [(as_text(i), as_text(o)) for i, o in p["tests"]], config))
            continue
        io = p["io"]
        inputs, outputs = io.get("inputs") or [], io.get("outputs") or []
        if "fn_name" in io:
            row = Row(set_name, p["id"], p["group"], half, "call")
            rows.append(score_call(row, p["statement"], list(zip(inputs, outputs)), config))
        else:
            row = Row(set_name, p["id"], p["group"], half, "stdin")
            tests = [(as_text(i), as_text(o)) for i, o in zip(inputs, outputs)]
            rows.append(score_stdin(row, p["statement"], tests, config))
    return rows


def derange(problems: list[dict], seed: int = SEED) -> list[dict]:
    """Negative control: every statement gets another problem's tests.

    One seeded cycle through the problems (no fixed points), within each
    official mode so stdin statements get stdin tests and call statements get
    call tests.  A matcher that confirms many of these is too lenient.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(problems):
        mode = "call" if "fn_name" in p.get("io", {}) else "stdin"
        groups[mode].append(i)
    out = [dict(p) for p in problems]
    rng = random.Random(seed)
    for members in groups.values():
        if len(members) < 2:
            continue
        order = list(members)
        rng.shuffle(order)
        for a, b in zip(order, order[1:] + order[:1]):
            for key in ("io", "tests"):
                if key in problems[b]:
                    out[a][key] = problems[b][key]
    return out


def report(rows: list[Row]) -> dict:
    out = {}
    for half in ("DIAG", "CONF"):
        part = [r for r in rows if r.half == half]
        out[half] = {"overall": summarise(part), "by_group": by_group(part)}
    out["audit"] = {
        "unconfirmed_extractions": audit(rows, "unconfirmed"),
        "missed_visible_tests": audit(rows, "missed"),
    }
    return out


def parser_hashes() -> dict:
    out = {}
    for f in PARSER_FILES:
        out[f] = subprocess.run(["git", "hash-object", f], cwd=REPO_ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apps", type=Path, default=REPO_ROOT / "artifacts" / "oracle" / "apps_train.jsonl")
    parser.add_argument("--cc", type=Path, default=REPO_ROOT / "artifacts" / "oracle" / "cc_train_statements.parquet")
    parser.add_argument("--coir", type=Path, default=REPO_ROOT / "artifacts" / "apps")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "experiments" / "V3_parser_gold" / "result_oracle.json")
    args = parser.parse_args(argv)

    apps = load_apps(args.apps, args.coir / "texts_queries_train.jsonl")
    apps_statements = []
    for name in ("texts_queries_train.jsonl", "texts_queries_test.jsonl"):
        with (args.coir / name).open(encoding="utf-8") as handle:
            apps_statements += [json.loads(line)["text"] for line in handle]
    cc, dropped = load_codecontests(args.cc, apps_statements)
    print(f"O1 APPS train: {len(apps)} problems with tests | O2 CodeContests train: {len(cc)} "
          f"after dropping {dropped} near-duplicates of APPS statements", flush=True)

    result = {"prereg": "experiments/V3_parser_gold/prereg_oracle.md", "parser_blobs": parser_hashes(),
              "seed": SEED, "o2_dropped_near_duplicates": dropped, "configs": {}}
    for name, config in (("primary", PRIMARY), ("uncapped", UNCAPPED)):
        result["configs"][name] = {
            "settings": {"call_style": config.call_style, "extended_stdin": config.extended_stdin,
                         "max_examples": config.max_examples},
            "O1_apps_train": report(evaluate(apps, "O1", config)),
            "O2_codecontests_train": report(evaluate(cc, "O2", config)),
            "negative_control": {
                "O1_apps_train": summarise(evaluate(derange(apps), "O1", config)),
                "O2_codecontests_train": summarise(evaluate(derange(cc), "O2", config)),
            },
        }
        print(f"{name} done", flush=True)

    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    tmp.replace(args.out)
    for name in ("primary", "uncapped"):
        for set_name in ("O1_apps_train", "O2_codecontests_train"):
            for half in ("DIAG", "CONF"):
                o = result["configs"][name][set_name][half]["overall"]
                s, c = o.get("stdin", {}), o.get("call", {})
                print(f"{name:8s} {set_name:22s} {half}  stdin P={s.get('precision', {}).get('value')} "
                      f"{s.get('precision', {}).get('ci95')}  R@cap={s.get('recall_capped', {}).get('value')}"
                      f"  | call P={c.get('precision', {}).get('value')} cov={c.get('coverage_any_confirmed')}")
    for name in ("primary", "uncapped"):
        for set_name, o in result["configs"][name]["negative_control"].items():
            print(f"{name:8s} NEGATIVE CONTROL {set_name:22s} stdin P={o.get('stdin', {}).get('precision', {}).get('value')}"
                  f"  call P={o.get('call', {}).get('precision', {}).get('value')}  (must be <= 0.05)")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
