"""Ablation harness over cached embeddings.

Runs the pipeline stage by stage on a split and reports each stage's gain with
a paired bootstrap, so nothing ships on a difference inside the noise band::

    python scripts/dev_eval.py --model f2llm17b                 # held-out DEV
    python scripts/dev_eval.py --model f2llm17b --execution     # + code execution
    python scripts/dev_eval.py --model f2llm17b --split test --authorize final_run

The ``test`` split is gated: it goes through ``DevelopmentHarness``, which
refuses unless the firewall is lowered with an accepted reason and there is
budget left in ``data/test_split_audit_log.json``.
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

from app.evaluation_harness import (  # noqa: E402
    AUDIT_LOG_PATH,
    get_persistent_audit_log,
    record_persistent_audit,
)
from app.pipeline.execution_rerank import (  # noqa: E402
    ExecutionConfig,
    apply_execution_rerank,
    parse_worked_examples,
)
from app.pipeline.features import (  # noqa: E402
    FEATURE_NAMES,
    CorpusStatistics,
    FeatureExtractor,
    analyze_queries,
)
from app.pipeline.metrics import metrics_from_ranks, paired_bootstrap, per_query_ndcg  # noqa: E402
from app.pipeline.reranker import Reranker, RerankerConfig  # noqa: E402
from app.pipeline.search import (  # noqa: E402
    CandidatePool,
    build_candidate_pool,
    load_hub_profile,
)
from app.pipeline.splits import (  # noqa: E402
    TEST_DOMINANT_FAMILY,
    family_distribution,
    make_development_split,
)


def load_jsonl_texts(path: Path) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    texts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            ids.append(row["id"])
            texts.append(row["text"])
    return ids, texts


def rank_from_pool_scores(
    scores: np.ndarray,
    gold_columns: np.ndarray,
    pool_miss_rank: int,
) -> np.ndarray:
    """1-based gold rank after rescoring the pool, ties broken by dense order."""
    n_queries, depth = scores.shape
    ranks = np.full(n_queries, pool_miss_rank, dtype=np.int32)
    present = np.flatnonzero(gold_columns >= 0)
    if present.size == 0:
        return ranks
    cols = gold_columns[present]
    block = scores[present]
    gold_scores = block[np.arange(len(present)), cols][:, None]
    positions = np.arange(depth)[None, :]
    better = (block > gold_scores).sum(axis=1)
    tied_earlier = ((block == gold_scores) & (positions < cols[:, None])).sum(axis=1)
    ranks[present] = (1 + better + tied_earlier).astype(np.int32)
    return ranks


def report(label: str, ranks: np.ndarray) -> np.ndarray:
    metrics = metrics_from_ranks(ranks)
    print(
        f"  {label:<28s} NDCG@10={metrics.ndcg_at_10:.5f}  MRR@10={metrics.mrr_at_10:.5f}  "
        f"R@1={metrics.recall_at_1:.4f}  R@10={metrics.recall_at_10:.4f}"
    )
    return per_query_ndcg(ranks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts" / "apps")
    parser.add_argument("--model", default="f2llm17b")
    parser.add_argument("--split", choices=["dev", "train", "test"], default="dev")
    parser.add_argument("--authorize", default=None, help="'calibration' or 'final_run' (test split only)")
    parser.add_argument("--depth", type=int, default=None, help="override pool depth")
    parser.add_argument("--execution", action="store_true", help="enable the execution reranker")
    parser.add_argument("--execution-candidates", type=int, default=5)
    parser.add_argument(
        "--execution-early-exit",
        action="store_true",
        help="stop executing a query's candidates at the first verified one",
    )
    parser.add_argument(
        "--execution-max-dense-gap",
        type=float,
        default=None,
        help="skip candidates scoring more than this far below the row's best",
    )
    parser.add_argument(
        "--execution-max-promotion-gap",
        type=float,
        default=None,
        help="execute every candidate, but refuse to promote one scoring more "
             "than this far below the row's best",
    )
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N queries")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--all-families", action="store_true",
                        help="use every train query, not just the test split's family")
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()

    art = args.artifacts
    config = RerankerConfig.load(args.weights)
    depth = args.depth or config.candidate_depth
    print(f"Reranker config: {config.describe()}")
    print(f"Pool depth: {depth}   execution rerank: {'ON' if args.execution else 'off'}")

    corpus_ids, corpus_texts = load_jsonl_texts(art / "texts_corpus.jsonl")
    corpus_emb = np.load(art / f"emb_{args.model}_corpus.npy").astype(np.float32)

    if args.split == "test":
        if args.authorize not in ("calibration", "final_run"):
            print(
                "REFUSED: the test split needs --authorize calibration|final_run.\n"
                "         Tune on --split dev; the test split has a two-access budget."
            )
            return 2
        spent = len(get_persistent_audit_log(AUDIT_LOG_PATH))
        if spent >= 2:
            print(f"REFUSED: test split budget exhausted ({spent}/2 accesses already logged).")
            return 2
        query_ids, query_texts = load_jsonl_texts(art / "texts_queries_test.jsonl")
        query_emb = np.load(art / f"emb_{args.model}_qtest.npy").astype(np.float32)
        qrels = json.loads((art / "qrels_test.json").read_text(encoding="utf-8"))
    else:
        query_ids, query_texts = load_jsonl_texts(art / "texts_queries_train.jsonl")
        query_emb = np.load(art / f"emb_{args.model}_qtrain.npy").astype(np.float32)
        qrels = json.loads((art / "qrels_train.json").read_text(encoding="utf-8"))
        if args.split == "dev":
            # Same seeded, distribution-matched split the tuner held out, so
            # "dev" here is genuinely the half the weights never saw.
            split = make_development_split(
                query_texts,
                match_family=None if args.all_families else TEST_DOMINANT_FAMILY,
                seed=args.seed,
            )
            print(f"Development split: {split.describe()}")
            rows = split.dev
            query_ids = [query_ids[i] for i in rows]
            query_texts = [query_texts[i] for i in rows]
            query_emb = query_emb[rows]

    print(
        "Query-family mix: "
        f"{ {k: f'{v*100:.1f}%' for k, v in family_distribution(query_texts).items()} }"
    )

    if args.limit:
        query_ids, query_texts = query_ids[: args.limit], query_texts[: args.limit]
        query_emb = query_emb[: args.limit]

    print(f"Split '{args.split}': {len(query_ids)} queries over {len(corpus_ids)} documents")

    doc_index_of = {doc_id: i for i, doc_id in enumerate(corpus_ids)}
    gold_doc_indices = np.array(
        [doc_index_of[next(iter(qrels[q]))] for q in query_ids], dtype=np.int32
    )

    t0 = time.time()
    pool = build_candidate_pool(query_emb, corpus_emb, depth=max(depth, 100))
    print(f"Dense first pass in {time.time() - t0:.1f}s")

    gold_columns = np.full(len(query_ids), -1, dtype=np.int32)
    for row in range(len(query_ids)):
        hit = np.flatnonzero(pool.indices[row][:depth] == gold_doc_indices[row])
        if hit.size:
            gold_columns[row] = int(hit[0])
    ceiling = float((gold_columns >= 0).mean())

    print()
    print("=" * 78)
    print(f"ABLATION  (model={args.model}, split={args.split})")
    print("=" * 78)

    stage_scores = pool.scores[:, :depth].copy()
    pool_indices = pool.indices[:, :depth]

    # Every stage is ranked by the same function over the same depth-`depth`
    # pool. Measuring stage 1 against the deeper retrieval pool instead would
    # give it a different recall basis and make the ablation deltas incomparable.
    dense_ranks = rank_from_pool_scores(stage_scores, gold_columns, depth + 1)
    dense_ndcg = report("1. dense only", dense_ranks)
    deep_recall = float((pool.gold_ranks(gold_doc_indices) <= pool.depth).mean())
    print(f"  {'':28s} pool@{depth} contains gold for {ceiling * 100:.2f}% of queries "
          f"(rerank ceiling); recall@{pool.depth}={deep_recall:.4f}")

    stage_ndcg = dense_ndcg
    stage_ranks = dense_ranks
    results: dict[str, object] = {
        "model": args.model,
        "split": args.split,
        "depth": depth,
        "n_queries": len(query_ids),
        "pool_ceiling": ceiling,
        "retrieval_recall_at_pool_depth": deep_recall,
        "retrieval_pool_depth": pool.depth,
        "dense": metrics_from_ranks(dense_ranks).as_dict(),
    }

    # -- stage 2: feature reranker ----------------------------------------
    if not config.is_identity:
        t0 = time.time()
        corpus_stats = CorpusStatistics.build(corpus_texts)
        hub_path = art / f"hubness_{args.model}.npz"
        if hub_path.exists():
            corpus_stats.hubness = load_hub_profile(hub_path, corpus_texts)
            matched = int((corpus_stats.hubness != 0).sum())
            print(f"\n  loaded hub profile from {hub_path.name} "
                  f"({matched} documents matched by content)")
        extractor = FeatureExtractor(
            corpus_stats,
            config.family_compat,
            length_prior_slope=config.length_prior_slope,
            length_prior_intercept=config.length_prior_intercept,
        )
        query_facts = analyze_queries(query_texts)
        weights = np.asarray(config.weights, dtype=np.float32)
        rescored = np.empty_like(stage_scores)
        for row in range(len(query_ids)):
            features = extractor.for_query(query_facts[row], pool_indices[row], stage_scores[row])
            rescored[row] = features @ weights
        rerank_ranks = rank_from_pool_scores(rescored, gold_columns, depth + 1)
        print(f"  (feature rerank computed in {time.time() - t0:.1f}s)")
        rerank_ndcg = report("2. + feature reranker", rerank_ranks)
        boot = paired_bootstrap(stage_ndcg, rerank_ndcg)
        print(f"  {'':28s} vs previous stage: {boot.describe()}")
        results["reranked"] = metrics_from_ranks(rerank_ranks).as_dict()
        results["rerank_bootstrap"] = boot.__dict__
        stage_ndcg, stage_ranks, stage_scores = rerank_ndcg, rerank_ranks, rescored
    else:
        query_facts = analyze_queries(query_texts)
        print("\n  (reranker is the identity configuration -- stage 2 skipped)")

    # -- stage 3: execution verification ----------------------------------
    if args.execution:
        exec_config = ExecutionConfig(
            enabled=True,
            max_candidates=args.execution_candidates,
            verified_bonus=float(np.abs(stage_scores).max()) + 1.0,
            stop_after_first_match=args.execution_early_exit,
            max_dense_gap=args.execution_max_dense_gap,
            max_promotion_gap=args.execution_max_promotion_gap,
        )
        knobs = []
        if exec_config.stop_after_first_match:
            knobs.append("early-exit")
        if exec_config.max_dense_gap is not None:
            knobs.append(f"dense-gate {exec_config.max_dense_gap:g}")
        if exec_config.max_promotion_gap is not None:
            knobs.append(f"promotion-gate {exec_config.max_promotion_gap:g}")
        print(f"  execution depth {exec_config.max_candidates}"
              + (f"   [{', '.join(knobs)}]" if knobs else "   [control]"))

        t0 = time.time()
        exec_scores = stage_scores.copy()
        fired = verified = executed = skipped = stopped = blocked = 0
        for row in range(len(query_ids)):
            if not parse_worked_examples(query_texts[row], limit=exec_config.max_examples):
                continue
            fired += 1
            candidate_texts = [corpus_texts[i] for i in pool_indices[row][: exec_config.max_candidates]]
            adjusted, audit = apply_execution_rerank(
                query_texts[row],
                candidate_texts,
                list(exec_scores[row][: exec_config.max_candidates]),
                exec_config,
            )
            exec_scores[row][: exec_config.max_candidates] = adjusted
            verified += int(audit["candidates_verified"])
            executed += int(audit["candidates_executed"])
            skipped += int(audit.get("candidates_skipped_by_gate", 0))
            stopped += 1 if audit.get("stopped_early") else 0
            blocked += int(audit.get("promotions_blocked", 0))
            if fired % 200 == 0:
                print(f"    executed {fired} queries ({time.time() - t0:.0f}s elapsed)")
        elapsed = time.time() - t0
        exec_ranks = rank_from_pool_scores(exec_scores, gold_columns, depth + 1)
        budget = fired * exec_config.max_candidates
        print(
            f"  (execution fired on {fired}/{len(query_ids)} queries "
            f"({fired / len(query_ids) * 100:.1f}%), {verified} candidates verified, "
            f"{elapsed:.0f}s total, {elapsed / max(1, fired):.2f}s per firing query)"
        )
        print(
            f"  (ran {executed:,} of a possible {budget:,} candidate executions "
            f"= {executed / max(1, budget):.1%}; {skipped:,} gated out, "
            f"{stopped:,} queries stopped early, {blocked:,} promotions blocked)"
        )
        results["execution_candidates_run"] = executed
        results["execution_candidates_budget"] = budget
        results["execution_gated_out"] = skipped
        results["execution_stopped_early"] = stopped
        results["execution_early_exit"] = exec_config.stop_after_first_match
        results["execution_max_dense_gap"] = exec_config.max_dense_gap
        results["execution_max_promotion_gap"] = exec_config.max_promotion_gap
        results["execution_promotions_blocked"] = blocked
        exec_ndcg = report("3. + execution verification", exec_ranks)
        boot = paired_bootstrap(stage_ndcg, exec_ndcg)
        print(f"  {'':28s} vs previous stage: {boot.describe()}")
        results["execution"] = metrics_from_ranks(exec_ranks).as_dict()
        results["execution_bootstrap"] = boot.__dict__
        results["execution_fired"] = fired
        results["execution_verified"] = verified
        results["execution_seconds"] = elapsed
        stage_ndcg, stage_ranks = exec_ndcg, exec_ranks

    print("-" * 78)
    overall = paired_bootstrap(dense_ndcg, stage_ndcg)
    final = metrics_from_ranks(stage_ranks)
    print(f"  FINAL   NDCG@10={final.ndcg_at_10:.5f}  MRR@10={final.mrr_at_10:.5f}")
    print(f"  end-to-end vs dense: {overall.describe()}")
    print("=" * 78)
    results["final"] = final.as_dict()
    results["overall_bootstrap"] = overall.__dict__

    if args.split == "test":
        record_persistent_audit(
            reason=args.authorize,
            row_count=len(query_ids),
            caller="scripts/dev_eval.py",
            extra={
                "metric_ndcg_at_10": round(final.ndcg_at_10, 5),
                "metric_mrr_at_10": round(final.mrr_at_10, 5),
                "model": args.model,
            },
        )
        print(f"Logged test-split access to {AUDIT_LOG_PATH}")

    if args.report:
        # Per-query scores, so two runs of this script can be compared to each
        # other with a paired bootstrap.  The aggregate numbers alone cannot
        # support that: pairing needs the same query in both vectors.
        results["query_ids"] = list(query_ids)
        results["per_query_ndcg"] = {
            "dense": [round(float(v), 6) for v in dense_ndcg],
            "final": [round(float(v), 6) for v in stage_ndcg],
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
