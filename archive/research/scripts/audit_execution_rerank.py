"""Decompose a shipped prediction file into why each query was won or lost.

The headline NDCG@10 says how well the pipeline did.  It does not say which
pass did the work, and in particular it does not separate the two things the
execution pass can do: promote a gold document that the dense pass had buried,
or promote a *wrong* document that happens to pass the same worked examples.

This script separates them, using nothing but the shipped predictions CSV and
the qrels.  Every query lands in exactly one bucket:

    verified_gold     a candidate was execution-verified, and it is the gold
    dense_won         nothing verified, and the dense pass was right anyway
    verified_wrong    something verified, it was not the gold, gold was in pool
    dense_missed      nothing verified, gold was in the pool but not at rank 1
    dense_absent      gold is not in the pool at all, nothing verified
    verified_absent   gold is not in the pool, and a wrong document verified

Run it against two prediction files to see what changing the candidate depth
actually bought.  ``data/predictions_top10.csv`` is the depth-10 run kept for
exactly this comparison; ``submission_predictions.csv`` is what shipped:

    python scripts/audit_execution_rerank.py \\
        --predictions data/predictions_top10.csv \\
        --compare     submission_predictions.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Raw cosine similarity cannot exceed 1.0, and `_apply_execution` adds a
#: bonus of at least 1.0 to every verified candidate, so any score above this
#: threshold identifies a row the execution pass promoted.
PROMOTION_THRESHOLD = 1.0

#: ``data/`` is committed; ``artifacts/`` is generated and gitignored, so a
#: fresh clone finds the first of these and a working tree finds either.
QRELS_CANDIDATES = (
    REPO_ROOT / "data" / "qrels_test.json",
    REPO_ROOT / "artifacts" / "apps" / "qrels_test.json",
)
DEFAULT_PREDICTIONS = REPO_ROOT / "submission_predictions.csv"


def default_qrels() -> Path:
    for path in QRELS_CANDIDATES:
        if path.exists():
            return path
    return QRELS_CANDIDATES[0]

BUCKETS = (
    "verified_gold",
    "dense_won",
    "verified_wrong",
    "dense_missed",
    "dense_absent",
    "verified_absent",
)
CORRECT_BUCKETS = ("verified_gold", "dense_won")


def load_gold(path: Path) -> dict[str, str]:
    """AppsRetrieval has exactly one relevant document per query."""
    qrels = json.loads(path.read_text(encoding="utf-8"))
    gold = {}
    for query_id, relevant in qrels.items():
        positives = [doc for doc, score in relevant.items() if score > 0]
        if len(positives) != 1:
            raise ValueError(
                f"{query_id} has {len(positives)} positives; this audit assumes one"
            )
        gold[query_id] = positives[0]
    return gold


def load_predictions(path: Path) -> dict[str, list[tuple[int, str, float]]]:
    ranked: dict[str, list[tuple[int, str, float]]] = collections.defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ranked[row["query_id"]].append(
                (int(row["rank"]), row["corpus_id"], float(row["score"]))
            )
    for items in ranked.values():
        items.sort()
    return ranked


def classify(
    ranked: dict[str, list[tuple[int, str, float]]], gold: dict[str, str]
) -> collections.Counter:
    buckets: collections.Counter = collections.Counter()
    for query_id, items in ranked.items():
        target = gold[query_id]
        pool = [doc for _, doc, _ in items]
        promoted = [doc for _, doc, score in items if score > PROMOTION_THRESHOLD]
        in_pool = target in pool
        won = bool(items) and items[0][1] == target

        if promoted:
            if won:
                buckets["verified_gold"] += 1
            elif in_pool:
                buckets["verified_wrong"] += 1
            else:
                buckets["verified_absent"] += 1
        else:
            if won:
                buckets["dense_won"] += 1
            elif in_pool:
                buckets["dense_missed"] += 1
            else:
                buckets["dense_absent"] += 1
    return buckets


def report(
    buckets: collections.Counter, total: int, compare: collections.Counter | None
) -> None:
    header = f"{'bucket':18s} {'count':>7s} {'share':>8s}"
    if compare is not None:
        header += f" {'compare':>8s} {'delta':>7s}"
    print(header)
    print("-" * len(header))
    for name in BUCKETS:
        line = f"{name:18s} {buckets[name]:7d} {buckets[name] / total:8.4f}"
        if compare is not None:
            line += f" {compare[name]:8d} {compare[name] - buckets[name]:+7d}"
        print(line)

    correct = sum(buckets[name] for name in CORRECT_BUCKETS)
    line = f"{'RANK-1 CORRECT':18s} {correct:7d} {correct / total:8.4f}"
    if compare is not None:
        other = sum(compare[name] for name in CORRECT_BUCKETS)
        line += f" {other:8d} {other - correct:+7d}"
    print(line)

    in_pool = total - buckets["dense_absent"] - buckets["verified_absent"]
    print(f"\ngold present in pool         : {in_pool} ({in_pool / total:.4f})")

    promoted = (
        buckets["verified_gold"] + buckets["verified_wrong"] + buckets["verified_absent"]
    )
    if promoted:
        precision = buckets["verified_gold"] / promoted
        print(
            f"queries with a verification  : {promoted} ({promoted / total:.4f})"
            f"\n  promotion precision        : {precision:.4f}"
            f"  ({buckets['verified_gold']} right,"
            f" {buckets['verified_wrong'] + buckets['verified_absent']} wrong)"
        )

    recoverable = buckets["dense_missed"] + buckets["verified_wrong"]
    deeper = buckets["dense_absent"] + buckets["verified_absent"]
    print(
        f"\nlosses                       : {total - correct}"
        f"\n  recoverable in this pool   : {recoverable}"
        f"  (verification failed or picked wrong)"
        f"\n  need a deeper pool         : {deeper}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="a second predictions file to diff against the first",
    )
    parser.add_argument("--qrels", type=Path, default=default_qrels())
    args = parser.parse_args(argv)

    for path in filter(None, (args.predictions, args.compare, args.qrels)):
        if not path.exists():
            parser.error(f"no such file: {path}")

    gold = load_gold(args.qrels)
    ranked = load_predictions(args.predictions)

    missing = set(ranked) - set(gold)
    if missing:
        parser.error(f"{len(missing)} predicted queries are absent from the qrels")

    buckets = classify(ranked, gold)
    other = classify(load_predictions(args.compare), gold) if args.compare else None

    print(f"predictions : {args.predictions}")
    if args.compare:
        print(f"compare     : {args.compare}")
    print(f"queries     : {len(ranked)}\n")
    report(buckets, len(ranked), other)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
