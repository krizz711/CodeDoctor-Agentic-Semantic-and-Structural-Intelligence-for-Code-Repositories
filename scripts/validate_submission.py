from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"Cannot read JSON: {error}"]

    if not isinstance(payload, dict):
        return ["Submission result must be a JSON object"]

    metrics = payload.get("metrics")
    if metrics is not None:
        for name in ("mrr", "ndcg_at_10"):
            value = metrics.get(name) if isinstance(metrics, dict) else None
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                errors.append(f"metrics.{name} must be a number between 0 and 1")

    task_results = payload.get("task_results")
    if task_results is not None and not isinstance(task_results, list):
        errors.append("task_results must be a list when present")

    predictions = payload.get("predictions")
    if predictions is not None and not isinstance(predictions, list):
        errors.append("predictions must be a list when present")

    if not metrics and not task_results:
        errors.append("Expected local metrics or MTEB task_results")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a local or MTEB retrieval result JSON.")
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    errors = validate(args.result)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Valid retrieval result: {args.result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
