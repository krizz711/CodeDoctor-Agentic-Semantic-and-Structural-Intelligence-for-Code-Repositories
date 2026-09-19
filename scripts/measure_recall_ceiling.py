from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.evaluation_harness import DevelopmentHarness
from app.retrieval.encode import CodeEncoder


def compute_recall_curve(
    encoder: CodeEncoder,
    queries_data: list[dict],
    corpus_lookup: dict[str, str],
    k_vals: tuple[int, ...] = (10, 50, 100, 200),
    batch_size: int = 64,
) -> dict[int, float]:
    """Computes Recall@K over real queries and corpus."""
    doc_ids = list(corpus_lookup.keys())
    doc_texts = [corpus_lookup[doc_id] for doc_id in doc_ids]
    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

    print(f"Encoding corpus of {len(doc_texts)} documents...")
    doc_embeddings = encoder.encode(doc_texts, mode="document", batch_size=batch_size)

    query_texts = [q["query"] for q in queries_data]
    print(f"Encoding {len(query_texts)} development queries...")
    query_embeddings = encoder.encode(query_texts, mode="query", batch_size=batch_size)

    print("Computing dense similarity matrix...")
    # Both are unit-normalized float32 vectors, dot product = cosine similarity
    sim_matrix = np.dot(query_embeddings, doc_embeddings.T)

    recalls = {k: 0.0 for k in k_vals}
    max_k = max(k_vals)

    valid_queries = 0
    for q_idx, q in enumerate(queries_data):
        gold_id = q["gold_id"]
        if gold_id not in doc_id_to_idx:
            continue
        valid_queries += 1
        gold_doc_idx = doc_id_to_idx[gold_id]

        scores = sim_matrix[q_idx]
        # Partition top max_k
        top_indices = np.argpartition(-scores, kth=min(max_k, len(scores) - 1))[:max_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        for k in k_vals:
            if gold_doc_idx in top_indices[:k]:
                recalls[k] += 1.0

    if valid_queries > 0:
        for k in k_vals:
            recalls[k] /= valid_queries

    return recalls


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure candidate-pool recall ceiling (Plan T3, Gap 3).")
    parser.add_argument("--model", default=None, help="Model name or override")
    parser.add_argument("--max-queries", type=int, default=300, help="Number of train queries to evaluate")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device")
    args = parser.parse_args()

    harness = DevelopmentHarness(firewall_enabled=True)
    train_split = harness.get_train_split()
    corpus = harness.get_corpus()

    eval_queries = train_split[: args.max_queries]
    print(f"Plan T3 Recall Ceiling: Evaluating {len(eval_queries)} queries against {len(corpus)} documents (device={args.device})")

    encoder = CodeEncoder(model_name=args.model, device=args.device)
    recalls = compute_recall_curve(
        encoder=encoder,
        queries_data=eval_queries,
        corpus_lookup=corpus,
        k_vals=(10, 50, 100, 200),
        batch_size=args.batch_size,
    )

    print("\n--- Plan T3 Candidate-Pool Recall Ceiling ---")
    for k in (10, 50, 100, 200):
        print(f"recall@{k} = {recalls[k]:.4f}")

    headroom_50 = recalls[50] - recalls[10]
    headroom_200 = recalls[200] - recalls[10]
    print(f"\nReranker Headroom (recall@50 - recall@10): {headroom_50:.4f}")
    print(f"Extended Headroom (recall@200 - recall@10): {headroom_200:.4f}")

    if recalls[50] >= 0.85 and headroom_50 < 0.10:
        print("[DECISION] Headroom is narrow (< 10%). Heavy reranker cancelled per Plan A5 / Section 8.3.")
    else:
        print(f"[DECISION] Headroom is {headroom_50:.4f}.")


if __name__ == "__main__":
    main()
