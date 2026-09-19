from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation import evaluate_queries
from app.retrieval.hybrid import Document, HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CodeDoctor retrieval on a local JSONL relevance set.")
    parser.add_argument("dataset", type=Path, help="JSONL rows with query, documents, and relevant_ids")
    parser.add_argument("--output", type=Path, default=Path("retrieval_results.json"))
    args = parser.parse_args()

    rows = []
    predictions = []
    for line in args.dataset.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        documents = [Document(**document) for document in item["documents"]]
        results = HybridRetriever(documents).search(item["query"], limit=10)
        ranked_ids = [result.document.document_id for result in results]
        rows.append((ranked_ids, set(item["relevant_ids"])))
        predictions.append({"query": item["query"], "results": ranked_ids})

    metrics = evaluate_queries(rows)
    payload = {
        "metrics": metrics,
        "scores": {"test": [{
            "ndcg_at_10": metrics["ndcg_at_10"],
            "mrr_at_10": metrics["mrr"],
        }]},
        "predictions": predictions,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
