"""Record what execution verification actually did, candidate by candidate.

The aggregate audit says how many candidates verified.  It does not say which
ones, how far down the pool they sat, or whether they were the gold -- and
those are exactly the quantities a promotion gate has to be sized against.

This runs the depth-N execution pass once over DEV and dumps, for every
executed candidate: its dense score, its gap from the best candidate in the
row, whether it verified, and whether it is the gold.

Because a *promotion* gate changes only which verified candidates receive the
bonus -- never which candidates are executed -- every threshold can then be
replayed from this dump exactly, at no further execution cost.  See
``scripts/replay_promotion_gate.py``, which checks that replaying an
unlimited gate reproduces the control run's score before trusting any other
threshold.

    python scripts/instrument_execution.py --model f2llm17b --execution-candidates 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.execution_rerank import (  # noqa: E402
    ExecutionConfig,
    parse_worked_examples,
    verify_candidate,
)
from app.pipeline.reranker import RerankerConfig  # noqa: E402
from app.pipeline.search import build_candidate_pool  # noqa: E402
from app.pipeline.splits import TEST_DOMINANT_FAMILY, make_development_split  # noqa: E402
from scripts.dev_eval import load_jsonl_texts  # noqa: E402

#: verified[row, j] encoding
NOT_EXECUTED = -1
FAILED = 0
VERIFIED = 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts" / "apps")
    parser.add_argument("--model", default="f2llm17b")
    parser.add_argument("--execution-candidates", type=int, default=20)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "arms" / "instrumented.npz")
    args = parser.parse_args(argv)

    art = args.artifacts
    depth = args.depth or RerankerConfig.load().candidate_depth
    top_n = args.execution_candidates

    corpus_ids, corpus_texts = load_jsonl_texts(art / "texts_corpus.jsonl")
    corpus_emb = np.load(art / f"emb_{args.model}_corpus.npy").astype(np.float32)
    query_ids, query_texts = load_jsonl_texts(art / "texts_queries_train.jsonl")
    query_emb = np.load(art / f"emb_{args.model}_qtrain.npy").astype(np.float32)
    qrels = json.loads((art / "qrels_train.json").read_text(encoding="utf-8"))

    # Exactly the split dev_eval holds out, so the dump is comparable to the
    # control arm rather than merely similar to it.
    split = make_development_split(
        query_texts, match_family=TEST_DOMINANT_FAMILY, seed=args.seed
    )
    rows = split.dev
    query_ids = [query_ids[i] for i in rows]
    query_texts = [query_texts[i] for i in rows]
    query_emb = query_emb[rows]
    if args.limit:
        query_ids, query_texts = query_ids[: args.limit], query_texts[: args.limit]
        query_emb = query_emb[: args.limit]

    n = len(query_ids)
    print(f"DEV queries {n}, pool depth {depth}, execution depth {top_n}")

    doc_index_of = {doc_id: i for i, doc_id in enumerate(corpus_ids)}
    gold_doc_indices = np.array(
        [doc_index_of[next(iter(qrels[q]))] for q in query_ids], dtype=np.int32
    )

    pool = build_candidate_pool(query_emb, corpus_emb, depth=max(depth, 100))
    scores = pool.scores[:, :depth].copy()
    indices = pool.indices[:, :depth]

    gold_columns = np.full(n, -1, dtype=np.int32)
    for row in range(n):
        hit = np.flatnonzero(indices[row] == gold_doc_indices[row])
        if hit.size:
            gold_columns[row] = int(hit[0])

    config = ExecutionConfig(enabled=True, max_candidates=top_n)
    verified = np.full((n, top_n), NOT_EXECUTED, dtype=np.int8)
    fired = np.zeros(n, dtype=bool)

    t0 = time.time()
    executed = 0
    for row in range(n):
        examples = parse_worked_examples(query_texts[row], limit=config.max_examples)
        if not examples:
            continue
        fired[row] = True
        for j in range(min(top_n, indices.shape[1])):
            text = corpus_texts[indices[row][j]]
            ok = verify_candidate(text, examples, config)
            verified[row, j] = VERIFIED if ok else FAILED
            executed += 1
        if int(fired.sum()) % 200 == 0 and j == top_n - 1:
            print(f"  {int(fired.sum())} queries, {executed:,} executions, "
                  f"{time.time() - t0:.0f}s")
    elapsed = time.time() - t0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        scores=scores.astype(np.float32),
        gold_columns=gold_columns,
        verified=verified,
        fired=fired,
        top_n=np.int32(top_n),
        depth=np.int32(depth),
        seconds=np.float32(elapsed),
        executed=np.int32(executed),
    )
    (args.out.with_suffix(".queries.json")).write_text(
        json.dumps(list(query_ids)), encoding="utf-8"
    )

    total_verified = int((verified == VERIFIED).sum())
    print()
    print(f"fired on {int(fired.sum())}/{n} queries, {executed:,} executions in {elapsed:.0f}s")
    print(f"verifications: {total_verified}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
