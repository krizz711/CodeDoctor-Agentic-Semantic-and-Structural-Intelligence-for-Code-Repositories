from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def validate(path: Path, *, official_apps: bool = False) -> list[str]:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"Cannot read JSON: {error}"]

    if not isinstance(payload, dict):
        return ["Submission result must be a JSON object"]
    if official_apps:
        from app.indexing.cache_validation import APPS_DATASET

        for field, expected in (("task_name", "AppsRetrieval"), ("dataset_revision", APPS_DATASET["revision"]),
                                ("mteb_version", "2.21.0")):
            if payload.get(field) != expected:
                errors.append(f"Official AppsRetrieval result requires {field}={expected!r}")

    # Official MTEB TaskResult format {"scores": {"test": [{"ndcg_at_10": ..., "mrr_at_10": ...}]}}
    scores = payload.get("scores")
    if not isinstance(scores, dict) or not scores:
        return ["Missing or invalid 'scores' object in MTEB results"]
    if official_apps and (set(scores) != {"test"} or not isinstance(scores["test"], list) or len(scores["test"]) != 1):
        errors.append("Official AppsRetrieval result requires exactly one test split score record")

    found_valid_split = False
    for split_name, split_metrics in scores.items():
        items = split_metrics if isinstance(split_metrics, list) else [split_metrics]
        for sub in items:
            if isinstance(sub, dict) and ("ndcg_at_10" in sub or "mrr_at_10" in sub):
                found_valid_split = True
                for key in ("ndcg_at_10", "mrr_at_10"):
                    if key not in sub:
                        errors.append(f"Missing required metric '{key}' in split '{split_name}'")
                        continue
                    val = sub[key]
                    if type(val) not in (int, float):
                        errors.append(f"scores.{split_name}.{key} must be a number (got {type(val).__name__})")
                    elif not math.isfinite(val):
                        errors.append(f"scores.{split_name}.{key} must be finite")
                    elif val < 0.0 or (val == 0.0 and not official_apps):
                        errors.append(
                            f"scores.{split_name}.{key} must be strictly positive (> 0.0), got {val} "
                            "(poisoned / unranked artifact)"
                        )
                    elif val > 1.0:
                        errors.append(f"scores.{split_name}.{key} must be <= 1.0, got {val}")

    if not found_valid_split:
        errors.append("No evaluation split in 'scores' contains valid 'ndcg_at_10' and 'mrr_at_10' metrics")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a local or MTEB retrieval result JSON.")
    parser.add_argument("result", type=Path)
    parser.add_argument("--official-apps", action="store_true", help="require the pinned official AppsRetrieval test result")
    args = parser.parse_args()
    errors = validate(args.result, official_apps=args.official_apps)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Valid retrieval result: {args.result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
