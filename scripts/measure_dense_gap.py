"""Where does the gold sit, in cosine terms, relative to the best candidate?

A promotion gate that refuses to execute candidates far below the top of the
pool only helps if the gold is reliably near the top.  This measures that
before any gate threshold is chosen, because a gate tight enough to exclude
golds is worse than no gate at all.

    python scripts/measure_dense_gap.py --model f2llm17b --depth 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.search import build_candidate_pool  # noqa: E402
from app.pipeline.splits import TEST_DOMINANT_FAMILY, make_development_split  # noqa: E402
from scripts.dev_eval import load_jsonl_texts  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts" / "apps")
    parser.add_argument("--model", default="f2llm17b")
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args(argv)

    art = args.artifacts
    query_ids, query_texts = load_jsonl_texts(art / "texts_queries_train.jsonl")
    corpus_ids, _ = load_jsonl_texts(art / "texts_corpus.jsonl")
    q_emb = np.load(art / f"emb_{args.model}_qtrain.npy")
    d_emb = np.load(art / f"emb_{args.model}_corpus.npy")

    split = make_development_split(
        query_texts, match_family=TEST_DOMINANT_FAMILY, seed=args.seed
    )
    rows = split.dev
    q_emb = q_emb[rows]
    query_ids = [query_ids[i] for i in rows]

    # AppsRetrieval qrels are the identity map; resolve it through the ids so
    # this does not quietly depend on that.
    col_of = {doc_id: i for i, doc_id in enumerate(corpus_ids)}
    gold_cols = np.array([col_of["d" + qid[1:]] for qid in query_ids])

    pool = build_candidate_pool(q_emb, d_emb, depth=args.depth)
    top = pool.scores[:, 0]

    in_pool = np.array([
        gold_cols[r] in pool.indices[r] for r in range(len(rows))
    ])
    gold_rank = np.full(len(rows), -1)
    gold_gap = np.full(len(rows), np.nan)
    for r in range(len(rows)):
        hit = np.where(pool.indices[r] == gold_cols[r])[0]
        if len(hit):
            gold_rank[r] = int(hit[0])
            gold_gap[r] = float(top[r] - pool.scores[r][hit[0]])

    print(f"DEV queries              : {len(rows)}")
    print(f"gold inside top-{args.depth} pool   : {in_pool.sum()} ({in_pool.mean():.4f})")
    print()

    gaps = gold_gap[~np.isnan(gold_gap)]
    print("cosine gap between the best candidate and the GOLD, when the gold is in the pool")
    for p in (50, 75, 90, 95, 99, 99.5, 100):
        print(f"  p{p:<5} {np.percentile(gaps, p):.4f}")
    print()

    print("if the gate refuses to execute candidates more than G below the top:")
    print(f"  {'G':>7}  {'golds kept':>11}  {'of in-pool':>10}  {'cands executed':>15}  {'vs depth ' + str(args.depth):>12}")
    full = len(rows) * args.depth
    for g in (0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30):
        kept = int((gaps <= g).sum())
        # scores are descending, so the gate is a prefix cut per row
        per_row = ((top[:, None] - pool.scores) <= g).sum(axis=1)
        executed = int(per_row.sum())
        print(f"  {g:>7.2f}  {kept:>11}  {kept / len(gaps):>10.4f}  {executed:>15,}  {executed / full:>11.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
