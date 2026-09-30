"""V3: build, then score, a hand-labelled gold set for the statement parser.

    python scripts/parser_gold_set.py sample --out experiments/V3_parser_gold/to_label.jsonl
    # a teammate who did NOT write the parser fills in "label" for every row
    python scripts/parser_gold_set.py score experiments/V3_parser_gold/labelled.jsonl

The parser decides which statements execution can check and what it compares
against, so its errors are silent score changes.  Unit tests pin the layouts we
thought of; this measures the ones we did not, on real statements, labelled by
someone other than the parser's author.

Sampling: 250 APPS *train* statements, stratified by what the parser believes
(stdin-original, stdin-extended, call-style, nothing found) so every branch is
measured, seeded.  Each row shows the parser's extraction; the labeller writes
the examples a careful human would extract:

    "label": {"mode": "stdin" | "call" | "none",
              "examples": [{"input": "...", "output": "..."}]}      # stdin
              "examples": [{"call": "f(1, 2)", "returns": "3"}]}    # call

``score`` reports, per parser branch: mode accuracy, and for stdin examples
exact-match precision/recall of (input, output) pairs after whitespace
normalisation; for call examples, agreement of (arguments, expected).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.execution_rerank import ExecutionConfig, _normalize_output, _statement_examples  # noqa: E402
from app.pipeline.statement import parse_literal  # noqa: E402
from scripts.dev_eval import load_jsonl_texts  # noqa: E402

CONFIG = ExecutionConfig(enabled=True, call_style=True, extended_stdin=True, max_examples=5)


def _extract(text: str) -> dict:
    stdin_examples, facts = _statement_examples(text, CONFIG)
    from app.pipeline.execution_rerank import parse_worked_examples

    original = bool(parse_worked_examples(text, limit=5))
    if facts is not None:
        branch = "call"
    elif stdin_examples and original:
        branch = "stdin-original"
    elif stdin_examples:
        branch = "stdin-extended"
    else:
        branch = "none"
    return {
        "branch": branch,
        "stdin": [{"input": e.stdin, "output": e.expected_stdout} for e in stdin_examples],
        "call": [
            {"args": repr(c.args), "kwargs": repr(c.kwargs), "returns": repr(c.expected), "callee": c.callee}
            for c in (facts.call_examples if facts is not None else ())
        ],
    }


def sample(args) -> int:
    ids, texts = load_jsonl_texts(args.artifacts / "texts_queries_train.jsonl")
    rng = random.Random(args.seed)
    by_branch: dict[str, list[int]] = defaultdict(list)
    extractions = {}
    for i, text in enumerate(texts):
        extraction = _extract(text)
        extractions[i] = extraction
        by_branch[extraction["branch"]].append(i)
    per_branch = max(1, args.size // len(by_branch))
    chosen: list[int] = []
    for branch, rows in sorted(by_branch.items()):
        rng.shuffle(rows)
        chosen.extend(rows[:per_branch])
    rng.shuffle(chosen)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for i in chosen:
            handle.write(json.dumps({
                "qid": ids[i], "statement": texts[i], "parser": extractions[i],
                "label": None,
            }) + "\n")
    print(f"wrote {len(chosen)} rows ({per_branch} per branch: {sorted(by_branch)}) -> {args.out}")
    print("Give the file to someone who did not write the parser; they fill in 'label' per row.")
    return 0


def _stdin_pairs(examples: list[dict]) -> set[tuple[str, str]]:
    return {(_normalize_output(e["input"]), _normalize_output(e["output"])) for e in examples}


def _call_pairs(examples: list[dict]) -> set[str]:
    out = set()
    for e in examples:
        if "call" in e:  # labeller format: {"call": "f(1, 2)", "returns": "3"}
            ok, value = parse_literal(e["returns"])
            args_text = e["call"][e["call"].find("(") + 1 : e["call"].rfind(")")]
            ok_args, args_value = parse_literal(f"({args_text},)") if args_text.strip() else (True, ())
            out.add(repr((args_value if ok_args else args_text, value if ok else e["returns"])))
        else:  # parser format
            args_value = parse_literal(e["args"])[1]
            kwargs_value = parse_literal(e["kwargs"])[1]
            flat = tuple(args_value) + tuple(v for _, v in kwargs_value)
            out.add(repr((flat, parse_literal(e["returns"])[1])))
    return out


def score(args) -> int:
    rows = [json.loads(line) for line in args.labelled.read_text(encoding="utf-8").splitlines() if line.strip()]
    unlabelled = [r["qid"] for r in rows if not r.get("label")]
    if unlabelled:
        print(f"{len(unlabelled)} rows are not labelled yet; refusing to report a partial score.")
        return 1
    stats: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        branch, parser, label = row["parser"]["branch"], row["parser"], row["label"]
        parser_mode = "call" if branch == "call" else ("stdin" if parser["stdin"] else "none")
        stats[branch]["n"] += 1
        stats[branch]["mode_correct"] += int(parser_mode == label["mode"])
        if label["mode"] == "stdin" or parser_mode == "stdin":
            got = _stdin_pairs(parser["stdin"]) if parser_mode == "stdin" else set()
            want = _stdin_pairs(label.get("examples", [])) if label["mode"] == "stdin" else set()
        else:
            got = _call_pairs(parser["call"]) if parser_mode == "call" else set()
            want = _call_pairs(label.get("examples", [])) if label["mode"] == "call" else set()
        stats[branch]["extracted"] += len(got)
        stats[branch]["gold"] += len(want)
        stats[branch]["correct"] += len(got & want)
    print(f"{'branch':16s} {'n':>4s} {'mode acc':>9s} {'precision':>10s} {'recall':>8s}")
    for branch, s in sorted(stats.items()):
        precision = s["correct"] / s["extracted"] if s["extracted"] else float("nan")
        recall = s["correct"] / s["gold"] if s["gold"] else float("nan")
        print(f"{branch:16s} {s['n']:4d} {s['mode_correct'] / s['n']:9.3f} {precision:10.3f} {recall:8.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts" / "apps")
    s.add_argument("--out", type=Path, default=REPO_ROOT / "experiments" / "V3_parser_gold" / "to_label.jsonl")
    s.add_argument("--size", type=int, default=250)
    s.add_argument("--seed", type=int, default=20260924)
    c = sub.add_parser("score")
    c.add_argument("labelled", type=Path)
    args = parser.parse_args(argv)
    return sample(args) if args.command == "sample" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
