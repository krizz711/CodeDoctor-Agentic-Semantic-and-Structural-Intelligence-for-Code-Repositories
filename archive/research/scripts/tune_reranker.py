"""Fit and validate the second-pass reranker on the CoIR APPS **train** split.

Usage::

    python scripts/tune_reranker.py --artifacts artifacts/apps --model f2llm17b

Expects an artifacts directory produced by ``scripts/encode_apps.py`` (locally)
or by the Kaggle GPU dump, containing:

    ids_corpus.json  ids_queries_train.json  ids_queries_test.json
    texts_corpus.jsonl  texts_queries_train.jsonl
    qrels_train.json
    emb_<model>_corpus.npy  emb_<model>_qtrain.npy

The train split is cut in half: weights are fitted on TUNE and reported on
held-out DEV with a paired bootstrap.  The test split is never opened here —
``app.evaluation`` raises if it is touched during tuning.
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

from app.pipeline.code_analysis import CodeFamily  # noqa: E402
from app.pipeline.features import (  # noqa: E402
    FEATURE_NAMES,
    CorpusStatistics,
    FamilyCompatibility,
    FeatureExtractor,
    analyze_queries,
)
from app.pipeline.metrics import metrics_from_ranks, paired_bootstrap, per_query_ndcg  # noqa: E402
from app.pipeline.reranker import RerankerConfig  # noqa: E402
from app.pipeline.search import (  # noqa: E402
    CandidatePool,
    build_candidate_pool,
    hubness_from_pool,
    save_hub_profile,
)
from app.pipeline.splits import (  # noqa: E402
    TEST_DOMINANT_FAMILY,
    family_distribution,
    make_development_split,
)
from app.pipeline.tuning import fit_length_prior, fit_weights, ranks_for_weights  # noqa: E402


def load_jsonl_texts(path: Path) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    texts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            ids.append(row["id"])
            texts.append(row["text"])
    return ids, texts


def build_feature_tensor(
    extractor: FeatureExtractor,
    query_facts: list,
    pool_indices: np.ndarray,
    pool_scores: np.ndarray,
) -> np.ndarray:
    n_queries, depth = pool_indices.shape
    tensor = np.zeros((n_queries, depth, len(FEATURE_NAMES)), dtype=np.float32)
    for row in range(n_queries):
        tensor[row] = extractor.for_query(query_facts[row], pool_indices[row], pool_scores[row])
    return tensor


def gold_columns_for(pool_indices: np.ndarray, gold_doc_indices: np.ndarray) -> np.ndarray:
    out = np.full(pool_indices.shape[0], -1, dtype=np.int32)
    for row in range(pool_indices.shape[0]):
        hit = np.flatnonzero(pool_indices[row] == gold_doc_indices[row])
        if hit.size:
            out[row] = int(hit[0])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts" / "apps")
    parser.add_argument("--model", default="f2llm17b", help="embedding tag inside the artifacts dir")
    parser.add_argument("--depth", type=int, default=50, help="candidate pool depth to rerank")
    parser.add_argument("--hubness-top-k", type=int, default=10)
    parser.add_argument("--sweeps", type=int, default=4)
    parser.add_argument(
        "--all-families",
        action="store_true",
        help="tune on every train query instead of the family the test split is made of",
    )
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "config" / "reranker.json")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "data" / "reranker_tuning_report.json")
    args = parser.parse_args()

    art = args.artifacts
    started = time.time()

    print(f"Loading artifacts from {art} (model tag: {args.model})")
    corpus_ids, corpus_texts = load_jsonl_texts(art / "texts_corpus.jsonl")
    train_ids, train_texts = load_jsonl_texts(art / "texts_queries_train.jsonl")
    qrels_train = json.loads((art / "qrels_train.json").read_text(encoding="utf-8"))

    corpus_emb = np.load(art / f"emb_{args.model}_corpus.npy").astype(np.float32)
    train_emb = np.load(art / f"emb_{args.model}_qtrain.npy").astype(np.float32)
    print(f"  corpus {corpus_emb.shape}  train queries {train_emb.shape}")

    doc_index_of = {doc_id: i for i, doc_id in enumerate(corpus_ids)}
    gold_doc_indices = np.array(
        [doc_index_of[next(iter(qrels_train[q]))] for q in train_ids], dtype=np.int32
    )

    print("Analyzing corpus (AST + regex fallback) ...")
    t0 = time.time()
    corpus_stats = CorpusStatistics.build(corpus_texts)
    parsed = sum(1 for f in corpus_stats.facts if f.parsed)
    print(f"  {len(corpus_texts)} snippets in {time.time() - t0:.1f}s; "
          f"AST-parsed {parsed} ({parsed / len(corpus_texts) * 100:.1f}%)")
    family_counts: dict[str, int] = {}
    for facts in corpus_stats.facts:
        family_counts[facts.family.value] = family_counts.get(facts.family.value, 0) + 1
    print(f"  snippet families: {family_counts}")

    print("Analyzing train queries ...")
    t0 = time.time()
    train_facts = analyze_queries(train_texts)
    print(f"  {len(train_facts)} queries in {time.time() - t0:.1f}s")
    query_family_counts: dict[str, int] = {}
    for facts in train_facts:
        query_family_counts[facts.family.value] = query_family_counts.get(facts.family.value, 0) + 1
    print(f"  query families: {query_family_counts}")

    print(f"Dense first pass (pool depth {args.depth}) ...")
    t0 = time.time()
    pool = build_candidate_pool(train_emb, corpus_emb, depth=args.depth)
    print(f"  built in {time.time() - t0:.1f}s")

    dense_ranks = pool.gold_ranks(gold_doc_indices)
    dense_metrics = metrics_from_ranks(dense_ranks)
    print(f"  TRAIN dense baseline: NDCG@10={dense_metrics.ndcg_at_10:.5f} "
          f"MRR@10={dense_metrics.mrr_at_10:.5f} recall@{args.depth}="
          f"{float((dense_ranks <= args.depth).mean()):.5f}")

    # ---- split TUNE / DEV -------------------------------------------------
    # The train and test splits have very different query mixes (see
    # app/pipeline/splits.py), so the development set is filtered to the family
    # the test split is actually made of, then split at random rather than
    # contiguously.
    n_train = len(train_ids)
    test_ids, test_texts = load_jsonl_texts(art / "texts_queries_test.jsonl")
    print("\n  query-family distribution (from query TEXT only, no labels):")
    print(f"    train: { {k: f'{v*100:.1f}%' for k, v in family_distribution(train_texts).items()} }")
    print(f"    test : { {k: f'{v*100:.1f}%' for k, v in family_distribution(test_texts).items()} }")
    del test_texts, test_ids

    split = make_development_split(
        train_texts,
        match_family=None if args.all_families else TEST_DOMINANT_FAMILY,
        seed=args.seed,
    )
    print(f"  development split: {split.describe()}")
    tune_slice = split.tune
    dev_slice = split.dev

    # ---- priors estimated on TUNE only ------------------------------------
    tune_gold = gold_doc_indices[tune_slice]
    gold_families = [corpus_stats.facts[i].family for i in tune_gold]
    family_compat = FamilyCompatibility.estimate(
        [train_facts[i].family for i in tune_slice], gold_families
    )
    print("  family compatibility (PMI, estimated on TUNE):")
    for (qf, cf), value in sorted(family_compat.table.items()):
        if abs(value) >= 0.05 and cf != CodeFamily.UNKNOWN.value:
            print(f"    {qf:>18s} x {cf:<18s} {value:+.3f}")

    slope, intercept = fit_length_prior(
        np.array([train_facts[i].char_length for i in tune_slice]),
        np.array([corpus_stats.facts[i].char_length for i in tune_gold]),
    )
    print(f"  length prior: log1p(doc) = {intercept:.3f} + {slope:.3f} * log1p(query)")

    # Hubness from TUNE queries only, so DEV stays a genuine held-out check.
    tune_pool = CandidatePool(pool.indices[tune_slice], pool.scores[tune_slice])
    corpus_stats.hubness = hubness_from_pool(
        tune_pool, len(corpus_ids), top_k=args.hubness_top_k
    )
    hot = int((corpus_stats.hubness > 0.1).sum())
    print(f"  hubness: {hot} documents flagged above 0.1 "
          f"(max raw excess normalised to 1.0)")

    extractor = FeatureExtractor(
        corpus_stats,
        family_compat,
        length_prior_slope=slope,
        length_prior_intercept=intercept,
    )

    print("Building feature tensors ...")
    t0 = time.time()
    features = build_feature_tensor(extractor, train_facts, pool.indices, pool.scores)
    print(f"  {features.shape} in {time.time() - t0:.1f}s")
    gold_cols = gold_columns_for(pool.indices, gold_doc_indices)
    eligible_in_pool = int((gold_cols[split.eligible] >= 0).sum())
    print(f"  gold document inside pool for {eligible_in_pool}/{len(split.eligible)} "
          f"eligible queries ({eligible_in_pool / len(split.eligible) * 100:.2f}%) "
          "-- this is the rerank ceiling")

    print("Coordinate ascent on TUNE ...")
    fit = fit_weights(
        features[tune_slice],
        gold_cols[tune_slice],
        pool_miss_rank=args.depth + 1,
        n_sweeps=args.sweeps,
    )
    weights = np.asarray(fit.weights, dtype=np.float32)
    print(f"  fitted weights: " + "  ".join(
        f"{n}={w:+.4f}" for n, w in zip(FEATURE_NAMES, fit.weights)
    ))

    # ---- held-out DEV evaluation ------------------------------------------
    dev_gold_cols = gold_cols[dev_slice]
    dev_features = features[dev_slice]
    baseline_weights = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    baseline_weights[FEATURE_NAMES.index("dense")] = 1.0

    dev_base_ranks = ranks_for_weights(
        dev_features, dev_gold_cols, baseline_weights, pool_miss_rank=args.depth + 1
    )
    dev_new_ranks = ranks_for_weights(
        dev_features, dev_gold_cols, weights, pool_miss_rank=args.depth + 1
    )
    dev_base = metrics_from_ranks(dev_base_ranks)
    dev_new = metrics_from_ranks(dev_new_ranks)
    boot = paired_bootstrap(per_query_ndcg(dev_base_ranks), per_query_ndcg(dev_new_ranks))

    print()
    print("=" * 74)
    print("HELD-OUT DEV (second half of train; never seen during fitting)")
    print(f"  dense only : NDCG@10={dev_base.ndcg_at_10:.5f}  MRR@10={dev_base.mrr_at_10:.5f}")
    print(f"  + reranker : NDCG@10={dev_new.ndcg_at_10:.5f}  MRR@10={dev_new.mrr_at_10:.5f}")
    print(f"  paired bootstrap on NDCG@10: {boot.describe()}")
    print("=" * 74)

    ship = boot.significant and boot.mean_delta > 0
    if not ship:
        print("VERDICT: improvement is inside the noise band -> shipping the")
        print("         identity (dense-only) configuration instead.")
        config = RerankerConfig(candidate_depth=args.depth)
    else:
        print("VERDICT: improvement is outside the noise band -> shipping tuned weights.")
        config = RerankerConfig(
            weights=fit.weights,
            candidate_depth=args.depth,
            family_compat=family_compat,
            length_prior_slope=slope,
            length_prior_intercept=intercept,
            hubness_top_k=args.hubness_top_k,
        )

    # Refit hubness over the FULL train split for the shipped artifact: more
    # reference queries, and still disjoint from test.
    config.provenance = {
        "fitted_on": "CoIR-Retrieval/apps train split",
        "embedding_model_tag": args.model,
        "family_filter": split.family_filter,
        "split_seed": split.seed,
        "train_queries_total": n_train,
        "tune_queries": int(len(split.tune)),
        "dev_queries": int(len(split.dev)),
        "candidate_depth": args.depth,
        "dev_dense_ndcg_at_10": dev_base.ndcg_at_10,
        "dev_reranked_ndcg_at_10": dev_new.ndcg_at_10,
        "dev_bootstrap": {
            "mean_delta": boot.mean_delta,
            "ci_low": boot.ci_low,
            "ci_high": boot.ci_high,
            "p_value": boot.p_value,
            "significant": boot.significant,
        },
        "shipped_tuned_weights": ship,
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    written = config.save(args.output)
    print(f"Wrote {written}")

    hubness_full = hubness_from_pool(pool, len(corpus_ids), top_k=args.hubness_top_k)
    hub_path = save_hub_profile(art / f"hubness_{args.model}.npz", corpus_texts, hubness_full)
    print(f"Wrote {hub_path} (reference = all {n_train} train queries, content-keyed)")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "train_dense": dense_metrics.as_dict(),
                "dev_dense": dev_base.as_dict(),
                "dev_reranked": dev_new.as_dict(),
                "weights": dict(zip(FEATURE_NAMES, fit.weights)),
                "history": fit.history,
                "corpus_families": family_counts,
                "query_families": query_family_counts,
                "provenance": config.provenance,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
