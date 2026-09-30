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
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: How many ranked documents to write per query.
DEFAULT_TOP_K = 10


def _load_predictions(prediction_folder: Path) -> dict[str, dict[str, float]]:
    """Read MTEB's prediction dump, whatever nesting version it used."""
    candidates = sorted(prediction_folder.rglob("*AppsRetrieval*predictions*.json"))
    if not candidates:
        candidates = sorted(prediction_folder.rglob("*.json"))
    for path in candidates:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for unwrap in (raw, *(v for v in raw.values() if isinstance(v, dict))):
            if not isinstance(unwrap, dict):
                continue
            if "test" in unwrap and isinstance(unwrap["test"], dict):
                return unwrap["test"]
            sample = next(iter(unwrap.values()), None)
            if isinstance(sample, dict) and sample and all(
                isinstance(v, (int, float)) for v in sample.values()
            ):
                return unwrap
    return {}


def export_from_predictions(
    prediction_folder: Path,
    task_payload: dict,
    output_dir: Path,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[Path]:
    """Write both CSVs. Returns the paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

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

    predictions = _load_predictions(prediction_folder)
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
            ranked = sorted(predictions[query_id].items(), key=lambda kv: kv[1], reverse=True)
            for rank, (doc_id, score) in enumerate(ranked[:top_k], start=1):
                writer.writerow([query_id, doc_id, rank, f"{float(score):.6f}"])
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
