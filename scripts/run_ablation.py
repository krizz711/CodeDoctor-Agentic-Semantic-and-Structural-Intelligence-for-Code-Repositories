from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.evaluation_harness import DevelopmentHarness, paired_bootstrap_test, compute_metrics_for_query
from app.retrieval.encode import CodeEncoder, _select_device


def rank_and_score(
    queries_data: list[dict],
    corpus_lookup: dict[str, str],
    doc_ids: list[str],
    sim_matrix: np.ndarray,
    use_fusion: bool = False,
) -> list[float]:
    """Computes per-query NDCG@10 from a precomputed similarity matrix."""
    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    scores_ndcg: list[float] = []

    for q_idx, q in enumerate(queries_data):
        gold_id = q["gold_id"]
        if gold_id not in doc_id_to_idx:
            scores_ndcg.append(0.0)
            continue

        dense_scores = sim_matrix[q_idx]
        top_indices = np.argsort(-dense_scores)[:50]
        ranked_doc_ids = [doc_ids[idx] for idx in top_indices]

        if use_fusion:
            # Simple lexical boost for exact keyword overlap in BM25 leg
            q_terms = set(q["query"].lower().split())
            fused_ranks: list[tuple[float, str]] = []
            for rank_d, did in enumerate(ranked_doc_ids, start=1):
                d_text = corpus_lookup[did].lower()
                lexical_hit = any(t in d_text for t in q_terms if len(t) > 4)
                # RRF dense + lexical
                dense_rrf = 1.0 / (60.0 + rank_d)
                lex_rrf = (1.0 / 61.0) if lexical_hit else 0.0
                fused_ranks.append((dense_rrf + 0.5 * lex_rrf, did))
            fused_ranks.sort(key=lambda x: x[0], reverse=True)
            final_ranked_doc_ids = [did for _, did in fused_ranks[:10]]
        else:
            final_ranked_doc_ids = ranked_doc_ids[:10]

        metrics = compute_metrics_for_query(final_ranked_doc_ids, gold_id)
        scores_ndcg.append(metrics["ndcg_at_10"])

    return scores_ndcg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation matrix with paired-bootstrap CI (Plan T6, Gap 3).")
    parser.add_argument("--max-queries", type=int, default=40, help="Number of train queries for ablation")
    parser.add_argument("--max-corpus", type=int, default=1200, help="Max corpus size (0 for full 8,765 corpus)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device (auto/cuda/cpu)")
    args = parser.parse_args()

    device = _select_device(args.device)
    print(f"Plan T6 Ablation Matrix: running on device={device}")

    harness = DevelopmentHarness(firewall_enabled=True)
    train_split = harness.get_train_split()
    corpus = harness.get_corpus()

    eval_queries = train_split[: args.max_queries]
    gold_ids = {q["gold_id"] for q in eval_queries}

    if args.max_corpus > 0 and len(corpus) > args.max_corpus:
        selected_ids = list(gold_ids)
        for doc_id in corpus:
            if len(selected_ids) >= args.max_corpus:
                break
            if doc_id not in gold_ids:
                selected_ids.append(doc_id)
        corpus = {did: corpus[did] for did in selected_ids}

    doc_ids = list(corpus.keys())
    doc_texts = [corpus[doc_id] for doc_id in doc_ids]

    print(f"Plan T6 Ablation Matrix: Running on {len(eval_queries)} real queries against {len(corpus)} documents")

    # 1. Base Encoder: F2LLM-v2-0.6B (no prefix)
    print("\n[Ablation 1/3] Evaluating Base Encoder (F2LLM-v2-0.6B) vs Baseline (MiniLM)...")
    cache_f2llm = Path(f"data/f2llm_corpus_emb_{len(doc_ids)}.npy")
    f2llm_encoder_noprefix = CodeEncoder(query_prefix="", document_prefix="", device=device)

    if cache_f2llm.exists():
        print(f"-> Loading cached F2LLM corpus embeddings from {cache_f2llm}...")
        f2llm_doc_emb = np.load(cache_f2llm)
    else:
        print("-> Encoding corpus with F2LLM-v2-0.6B...")
        f2llm_doc_emb = f2llm_encoder_noprefix.encode(
            doc_texts, mode="document", batch_size=args.batch_size, show_progress_bar=True
        )
        cache_f2llm.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_f2llm, f2llm_doc_emb)

    query_texts = [q["query"] for q in eval_queries]
    f2llm_q_emb_noprefix = f2llm_encoder_noprefix.encode(query_texts, mode="query", batch_size=args.batch_size)
    sim_base = np.dot(f2llm_q_emb_noprefix, f2llm_doc_emb.T)
    scores_base = rank_and_score(eval_queries, corpus, doc_ids, sim_base, use_fusion=False)

    cache_minilm = Path(f"data/minilm_corpus_emb_{len(doc_ids)}.npy")
    minilm_encoder = CodeEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2", query_prefix="", document_prefix="", device=device)

    if cache_minilm.exists():
        print(f"-> Loading cached MiniLM corpus embeddings from {cache_minilm}...")
        minilm_doc_emb = np.load(cache_minilm)
    else:
        print("-> Encoding corpus with sentence-transformers/all-MiniLM-L6-v2 baseline...")
        minilm_doc_emb = minilm_encoder.encode(
            doc_texts, mode="document", batch_size=args.batch_size, show_progress_bar=True
        )
        cache_minilm.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_minilm, minilm_doc_emb)

    minilm_q_emb = minilm_encoder.encode(query_texts, mode="query", batch_size=args.batch_size)
    sim_minilm = np.dot(minilm_q_emb, minilm_doc_emb.T)
    scores_weak = rank_and_score(eval_queries, corpus, doc_ids, sim_minilm, use_fusion=False)

    ci_base = paired_bootstrap_test(scores_base, scores_weak)
    print(f"base_encoder: mean_diff={ci_base['mean_diff']:.4f} CI=[{ci_base['ci_lower']:.4f}, {ci_base['ci_upper']:.4f}] significant={ci_base['significant']}")

    # 2. +Prefixes: F2LLM with instruction prefix
    print("\n[Ablation 2/3] Evaluating +Prefixes vs without prefixes...")
    f2llm_encoder_prefix = CodeEncoder(device=device)  # Uses frozen config prefix
    f2llm_q_emb_prefix = f2llm_encoder_prefix.encode(query_texts, mode="query", batch_size=args.batch_size)
    sim_prefix = np.dot(f2llm_q_emb_prefix, f2llm_doc_emb.T)
    scores_prefix = rank_and_score(eval_queries, corpus, doc_ids, sim_prefix, use_fusion=False)
    ci_prefix = paired_bootstrap_test(scores_prefix, scores_base)
    print(f"+prefixes:    mean_diff={ci_prefix['mean_diff']:.4f} CI=[{ci_prefix['ci_lower']:.4f}, {ci_prefix['ci_upper']:.4f}] significant={ci_prefix['significant']}")

    # 3. +Fusion: F2LLM with instruction prefix + RRF BM25 lexical fusion
    print("\n[Ablation 3/3] Evaluating +Fusion vs pure dense...")
    scores_fusion = rank_and_score(eval_queries, corpus, doc_ids, sim_prefix, use_fusion=True)
    ci_fusion = paired_bootstrap_test(scores_fusion, scores_prefix)
    print(f"+fusion:      mean_diff={ci_fusion['mean_diff']:.4f} CI=[{ci_fusion['ci_lower']:.4f}, {ci_fusion['ci_upper']:.4f}] significant={ci_fusion['significant']}")

    print("\n--- Plan T6 Summary Table ---")
    print("| Component | Mean Diff (NDCG@10) | 95% Confidence Interval | Significant |")
    print("|---|---|---|---|")
    print(f"| base_encoder | {ci_base['mean_diff']:+.4f} | [{ci_base['ci_lower']:.4f}, {ci_base['ci_upper']:.4f}] | {ci_base['significant']} |")
    print(f"| +prefixes    | {ci_prefix['mean_diff']:+.4f} | [{ci_prefix['ci_lower']:.4f}, {ci_prefix['ci_upper']:.4f}] | {ci_prefix['significant']} |")
    print(f"| +fusion      | {ci_fusion['mean_diff']:+.4f} | [{ci_fusion['ci_lower']:.4f}, {ci_fusion['ci_upper']:.4f}] | {ci_fusion['significant']} |")


if __name__ == "__main__":
    main()
