"""Does a small-model first pass plus a large-model rerank match the large model?

The standard advice for a CPU budget is to retrieve with a cheap encoder and
rerank the survivors with an expensive one.  This measures whether that holds
on AppsRetrieval: take F2LLM-v2-0.6B's top-K, rescore exactly those K with
F2LLM-v2-1.7B, and compare against running 1.7B over the whole corpus.

It also reports the ceiling the cascade cannot exceed -- the 0.6B pass's
recall@K -- because a rerank can only reorder what the first pass retrieved.

    python scripts/measure_cascade.py --split test

One caveat the arithmetic makes concrete rather than argues: with two
bi-encoders this saves no indexing work.  Reranking with 1.7B needs 1.7B
document vectors, so the corpus must be encoded with BOTH models.  The saving
would be real only for a cross-encoder, which scores query-document pairs at
query time and needs no document index at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.metrics import metrics_from_ranks, paired_bootstrap, per_query_ndcg  # noqa: E402
from app.pipeline.search import build_candidate_pool  # noqa: E402
from scripts.dev_eval import load_jsonl_texts, rank_from_pool_scores  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts" / "apps")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--small", default="f2llm06b")
    parser.add_argument("--large", default="f2llm17b")
    parser.add_argument("--depths", type=int, nargs="*", default=[20, 50, 100, 200])
    args = parser.parse_args(argv)

    art = args.artifacts
    corpus_ids, _ = load_jsonl_texts(art / "texts_corpus.jsonl")
    suffix = "qtest" if args.split == "test" else "qtrain"
    query_ids, _ = load_jsonl_texts(art / f"texts_queries_{args.split}.jsonl")
    qrels = json.loads((art / f"qrels_{args.split}.json").read_text(encoding="utf-8"))

    column_of = {doc: i for i, doc in enumerate(corpus_ids)}
    gold = np.array([column_of[next(iter(qrels[q]))] for q in query_ids], dtype=np.int32)

    small_q = np.load(art / f"emb_{args.small}_{suffix}.npy").astype(np.float32)
    small_d = np.load(art / f"emb_{args.small}_corpus.npy").astype(np.float32)
    large_q = np.load(art / f"emb_{args.large}_{suffix}.npy").astype(np.float32)
    large_d = np.load(art / f"emb_{args.large}_corpus.npy").astype(np.float32)

    n = len(query_ids)
    print(f"split {args.split}: {n} queries over {len(corpus_ids)} documents")
    print(f"small = {args.small}   large = {args.large}")
    print()

    def full(qe, de, label):
        pool = build_candidate_pool(qe, de, depth=200)
        columns = np.full(n, -1, dtype=np.int32)
        for row in range(n):
            hit = np.flatnonzero(pool.indices[row] == gold[row])
            if hit.size:
                columns[row] = int(hit[0])
        ranks = rank_from_pool_scores(pool.scores, columns, 201)
        return metrics_from_ranks(ranks), per_query_ndcg(ranks), pool

    small_metrics, small_ndcg, small_pool = full(small_q, small_d, "small")
    large_metrics, large_ndcg, _ = full(large_q, large_d, "large")

    header = f"{'configuration':<40}{'NDCG@10':>10}{'MRR@10':>9}{'R@1':>8}{'R@10':>8}"
    print(header)
    print("-" * len(header))

    def show(label, m):
        print(f"{label:<40}{m.ndcg_at_10:>10.5f}{m.mrr_at_10:>9.5f}"
              f"{m.recall_at_1:>8.4f}{m.recall_at_10:>8.4f}")

    show(f"{args.small} alone (full corpus)", small_metrics)
    show(f"{args.large} alone (full corpus)", large_metrics)
    print()

    print("cascade: 0.6B retrieves top-K, 1.7B rescores exactly those K")
    print(f"{'K':<8}{'first-pass R@K':>16}{'NDCG@10':>10}{'MRR@10':>9}{'R@1':>8}"
          f"{'vs large alone':>16}{'95% CI':>22}")
    print("-" * 89)

    for depth in args.depths:
        indices = small_pool.indices[:, :depth]
        # Rescore those exact candidates with the large model.
        rescored = np.einsum("qd,qkd->qk", large_q, large_d[indices])
        columns = np.full(n, -1, dtype=np.int32)
        for row in range(n):
            hit = np.flatnonzero(indices[row] == gold[row])
            if hit.size:
                columns[row] = int(hit[0])
        recall_at_k = float((columns >= 0).mean())
        ranks = rank_from_pool_scores(rescored, columns, depth + 1)
        metrics = metrics_from_ranks(ranks)
        boot = paired_bootstrap(large_ndcg, per_query_ndcg(ranks), n_resamples=2000)
        ci = f"[{boot.ci_low:+.5f}, {boot.ci_high:+.5f}]"
        print(f"{depth:<8}{recall_at_k:>16.4f}{metrics.ndcg_at_10:>10.5f}"
              f"{metrics.mrr_at_10:>9.5f}{metrics.recall_at_1:>8.4f}"
              f"{boot.mean_delta:>+16.5f}{ci:>22}")

    print()
    print("first-pass R@K is the ceiling: a rerank cannot retrieve what the")
    print("first pass missed, so the cascade can never beat that column.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
