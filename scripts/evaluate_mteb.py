from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the official MTEB AppsRetrieval result JSON on CPU.")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output", type=Path, default=Path("appsretrieval_results.json"))
    args = parser.parse_args()

    try:
        import mteb
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise SystemExit("Install evaluator dependencies first: pip install -r requirements-eval.txt") from error

    model = SentenceTransformer(args.model, device="cpu")
    task = mteb.get_task("AppsRetrieval")
    results = mteb.evaluate(model, [task], encode_kwargs={"batch_size": 32})
    task_result = next(iter(results.task_results))
    args.output.write_text(json.dumps(task_result.to_dict(), indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
