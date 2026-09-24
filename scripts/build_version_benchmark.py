"""Build a cross-version benchmark from APPS DEV: correct code among buggy versions.

    python scripts/build_version_benchmark.py --queries 150

The Bonus goal says snippets across versions are "very similar, which would
make them hard to rank properly".  ExecRetrieval (2026) quantified it for
embeddings: the correct implementation ranked below a single-edit buggy
variant in 67-78% of queries.  This builds that situation from APPS DEV:

* take a DEV statement (APPS *train* split, all families, the E1 DEV half)
  whose gold solution reproduces its worked examples;
* render the gold through ``ast.unparse`` so every version shares formatting
  (otherwise whitespace, not behaviour, could separate them);
* generate single-edit mutants (operator swap, comparison flip, integer
  constant +/- 1, boolean operator swap) and keep those that still *run* on
  the examples but produce the wrong output -- near misses, not crashes;
* lay the versions out as one lineage, with the correct one at a *random*
  position in the history, so "latest wins" cannot leak the answer.

Output ``artifacts/versions_apps/``: ``texts_queries.jsonl``,
``texts_corpus.jsonl`` (APPS corpus with each chosen gold replaced by its
lineage), ``qrels.json`` (query -> the correct version), ``lineage.json``
(doc -> lineage key and version index), ``splits.json`` (all rows in
``ext_dev`` so the recorder can read it).
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.execution_rerank import (  # noqa: E402
    ExecutionConfig,
    compare_output,
    execute_candidate,
    parse_worked_examples,
)
from app.pipeline.splits import make_development_split  # noqa: E402
from scripts.dev_eval import load_jsonl_texts  # noqa: E402

_SWAPS = {
    ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult,
    ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.And: ast.Or, ast.Or: ast.And,
}


def _sites(tree: ast.AST) -> list[tuple[str, int]]:
    sites = []
    for index, node in enumerate(ast.walk(tree)):
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and type(node.op) in _SWAPS:
            sites.append(("op", index))
        elif isinstance(node, ast.Compare) and node.ops and type(node.ops[0]) in _SWAPS:
            sites.append(("cmp", index))
        elif isinstance(node, ast.BoolOp) and type(node.op) in _SWAPS:
            sites.append(("bool", index))
        elif isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) < 10**6:
            sites.append(("const", index))
    return sites


def _mutate(tree: ast.AST, site: tuple[str, int], rng: random.Random) -> str | None:
    clone = copy.deepcopy(tree)
    kind, index = site
    for i, node in enumerate(ast.walk(clone)):
        if i != index:
            continue
        if kind in ("op", "bool"):
            node.op = _SWAPS[type(node.op)]()
        elif kind == "cmp":
            node.ops[0] = _SWAPS[type(node.ops[0])]()
        elif kind == "const":
            node.value = node.value + rng.choice((-1, 1))
        break
    try:
        return ast.unparse(clone)
    except Exception:
        return None


def _behaviour(code: str, examples, config) -> str:
    """``pass`` / ``wrong`` (runs, wrong output) / ``crash`` on the examples."""
    for example in examples:
        outcome = execute_candidate(code, example.stdin, config)
        if not outcome.ok:
            return "crash"
        if compare_output(outcome.stdout, example.expected_stdout, config) is None:
            return "wrong"
    return "pass"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts" / "apps")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "versions_apps")
    parser.add_argument("--queries", type=int, default=150)
    parser.add_argument("--mutants", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--split-seed", type=int, default=20260919)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)
    config = ExecutionConfig(enabled=True, timeout_seconds=2.0)

    corpus_ids, corpus_texts = load_jsonl_texts(args.artifacts / "texts_corpus.jsonl")
    query_ids, query_texts = load_jsonl_texts(args.artifacts / "texts_queries_train.jsonl")
    qrels = json.loads((args.artifacts / "qrels_train.json").read_text(encoding="utf-8"))
    text_of = dict(zip(corpus_ids, corpus_texts))
    split = make_development_split(query_texts, match_family=None, seed=args.split_seed)
    dev_rows = list(split.dev)
    rng.shuffle(dev_rows)

    candidates = []
    for row in dev_rows:
        examples = parse_worked_examples(query_texts[row], limit=2)
        if not examples:
            continue
        gold_id = next(iter(qrels[query_ids[row]]))
        try:
            tree = ast.parse(text_of[gold_id])
            normalized = ast.unparse(tree)
        except (SyntaxError, ValueError, RecursionError):
            continue
        candidates.append((row, gold_id, tree, normalized, examples))

    def build(item):
        row, gold_id, tree, normalized, examples = item
        local = random.Random(f"{args.seed}-{gold_id}")
        if _behaviour(normalized, examples, config) != "pass":
            return None
        sites = _sites(tree)
        local.shuffle(sites)
        mutants: list[str] = []
        seen = {normalized}
        for site in sites[:40]:
            code = _mutate(tree, site, local)
            if code is None or code in seen:
                continue
            seen.add(code)
            if _behaviour(code, examples, config) == "wrong":
                mutants.append(code)
                if len(mutants) >= args.mutants:
                    break
        if len(mutants) < args.mutants:
            return None
        return row, gold_id, normalized, mutants

    built = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(build, candidates[: args.queries * 4]):
            if result is not None:
                built.append(result)
            if len(built) >= args.queries:
                break
    built = built[: args.queries]

    replaced = {gold_id for _, gold_id, _, _ in built}
    corpus = [{"id": d, "text": t} for d, t in zip(corpus_ids, corpus_texts) if d not in replaced]
    queries, new_qrels, lineage = [], {}, {}
    for row, gold_id, normalized, mutants in built:
        versions = list(mutants)
        correct_at = rng.randrange(len(versions) + 1)
        versions.insert(correct_at, normalized)
        for k, code in enumerate(versions):
            doc_id = f"{gold_id}@v{k + 1}"
            corpus.append({"id": doc_id, "text": code})
            lineage[doc_id] = {"lineage": gold_id, "version": k + 1, "correct": k == correct_at}
        qid = query_ids[row]
        queries.append({"id": qid, "text": query_texts[row]})
        new_qrels[qid] = {f"{gold_id}@v{correct_at + 1}": 1}

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "texts_queries.jsonl").open("w", encoding="utf-8") as handle:
        for q in queries:
            handle.write(json.dumps(q) + "\n")
    with (args.out / "texts_corpus.jsonl").open("w", encoding="utf-8") as handle:
        for d in corpus:
            handle.write(json.dumps(d) + "\n")
    (args.out / "qrels.json").write_text(json.dumps(new_qrels), encoding="utf-8")
    (args.out / "lineage.json").write_text(json.dumps(lineage), encoding="utf-8")
    (args.out / "splits.json").write_text(json.dumps({"ext_dev": sorted(new_qrels)}), encoding="utf-8")
    print(f"built {len(queries)} lineages ({args.mutants} buggy + 1 correct version each) "
          f"from {len(candidates)} eligible DEV statements; corpus {len(corpus)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
