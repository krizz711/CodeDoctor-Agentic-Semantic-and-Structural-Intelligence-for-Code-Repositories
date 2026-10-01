"""Export the ranked responses as CSV alongside the MTEB JSON.

The Theme 01 brief describes the screening submission twice: as "a csv file
with the responses from the test split", and as the MTEB results JSON attached
to a GitHub release.  Producing both removes the ambiguity — the JSON is the
scored artifact, the CSVs are the readable responses behind it.

    submission_predictions.csv   query_id, corpus_id, rank, score
    submission_metrics.csv       metric, value
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: How many ranked documents to write per query.
DEFAULT_TOP_K = 10


def _load_predictions(prediction_folder: Path, *, strict: bool = False) -> dict[str, dict[str, float]]:
    """Read MTEB's prediction dump, whatever nesting version it used."""
    candidates = sorted(prediction_folder.rglob("*AppsRetrieval*predictions*.json"))
    if not candidates:
        candidates = sorted(prediction_folder.rglob("*.json"))
    found = []
    for path in candidates:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for unwrap in (raw, *(v for v in raw.values() if isinstance(v, dict))):
            if not isinstance(unwrap, dict):
                continue
            if "test" in unwrap and isinstance(unwrap["test"], dict):
                found.append(unwrap["test"])
                break
            sample = next(iter(unwrap.values()), None)
            if isinstance(sample, dict) and sample and all(
                isinstance(v, (int, float)) for v in sample.values()
            ):
                found.append(unwrap)
                break
    if strict and len(found) != 1:
        raise ValueError(f"expected exactly one prediction dump, found {len(found)}")
    return found[0] if found else {}


def validate_predictions(predictions: dict, *, query_ids: set[str], corpus_ids: set[str], top_k: int = 10) -> None:
    """Check response identity and scores without using labels in retrieval."""
    if set(predictions) != query_ids:
        raise ValueError("prediction query IDs differ from the evaluated task")
    for query, scores in predictions.items():
        if not isinstance(scores, dict) or len(scores) < top_k:
            raise ValueError(f"query {query} does not contain {top_k} distinct predictions")
        if set(scores) - corpus_ids:
            raise ValueError(f"query {query} contains an unknown corpus ID")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in scores.values()):
            raise ValueError(f"query {query} contains non-finite or nonnumeric scores")


def verify_prediction_metrics(predictions: dict, qrels: dict, scores: dict) -> dict:
    """Recompute the two required metrics from raw predictions, never CSV rounding."""
    import pytrec_eval
    from mteb._evaluators.retrieval_metrics import mrr

    ndcg_rows = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"}).evaluate(predictions)
    if set(ndcg_rows) != set(predictions) or set(qrels) != set(predictions):
        raise ValueError("prediction metrics do not cover exactly the evaluated query IDs")
    ndcg = round(sum(row["ndcg_cut_10"] for row in ndcg_rows.values()) / len(ndcg_rows), 5)
    reciprocal = mrr(qrels, predictions, [10])["MRR@10"]
    measured = {"ndcg_at_10": ndcg, "mrr_at_10": sum(reciprocal) / len(reciprocal)}
    for name, value in measured.items():
        if name not in scores or not math.isclose(float(scores[name]), value, rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"raw predictions disagree with official {name}: {value} versus {scores.get(name)}")
    return measured


def export_from_predictions(
    prediction_folder: Path,
    task_payload: dict,
    output_dir: Path,
    *,
    top_k: int = DEFAULT_TOP_K,
    expected_query_ids: set[str] | None = None,
    valid_corpus_ids: set[str] | None = None,
    qrels: dict | None = None,
) -> list[Path]:
    """Write both CSVs. Returns the paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    strict = expected_query_ids is not None
    predictions = _load_predictions(prediction_folder, strict=strict)
    if not predictions:
        raise ValueError(f"No MTEB prediction dump found under {prediction_folder}")
    if strict:
        if valid_corpus_ids is None or qrels is None:
            raise ValueError("strict prediction validation requires corpus IDs and qrels")
        validate_predictions(predictions, query_ids=expected_query_ids, corpus_ids=valid_corpus_ids, top_k=top_k)
        checked = verify_prediction_metrics(predictions, qrels, task_payload["scores"]["test"][0])
        (output_dir / "prediction-verification.json").write_text(json.dumps({"queries": len(predictions),
            "corpus": len(valid_corpus_ids), "top_k": top_k, "recomputed_metrics": checked}, indent=2), encoding="utf-8")

    metrics_path = output_dir / "submission_metrics.csv"
    scores = task_payload.get("scores", {}).get("test", [{}])[0]
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key in ("ndcg_at_10", "mrr_at_10", "map_at_10", "recall_at_10", "recall_at_100"):
            if key in scores:
                writer.writerow([key, scores[key]])
        for key, value in sorted(scores.items()):
            if key.startswith("nauc_") or key in {
                "ndcg_at_10", "mrr_at_10", "map_at_10", "recall_at_10", "recall_at_100"
            }:
                continue
            if isinstance(value, (int, float, str)):
                writer.writerow([key, value])
    written.append(metrics_path)

    predictions_path = output_dir / "submission_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "corpus_id", "rank", "score"])
        if not predictions:
            raise SystemExit(
                f"No MTEB prediction dump found under {prediction_folder}. "
                "Re-run scripts/evaluate_mteb.py with --predictions."
            )
        for query_id in sorted(predictions, key=lambda q: int(q[1:]) if q[1:].isdigit() else 0):
            ranked = sorted(predictions[query_id].items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            for rank, (doc_id, score) in enumerate(ranked[:top_k], start=1):
                writer.writerow([query_id, doc_id, rank, repr(float(score))])
    written.append(predictions_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="MTEB prediction_folder")
    parser.add_argument("--results", type=Path, default=Path("results/appsretrieval_results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    for path in export_from_predictions(
        args.predictions, payload, args.output_dir, top_k=args.top_k
    ):
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
