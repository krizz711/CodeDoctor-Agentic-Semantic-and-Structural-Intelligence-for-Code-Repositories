"""Generate the official MTEB AppsRetrieval result JSON using CodeDoctor's own encoder.

This script must score CodeDoctor's PrePostPipelineEncoder, not a stock baseline.
Run:
    pip install -r requirements-eval.txt
    python scripts/evaluate_mteb.py --output appsretrieval_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Guard rails: results below these thresholds almost certainly mean the encoder is
# broken (e.g. untrained / random vectors) rather than merely weak.
RANDOM_BASELINE_NDCG = 0.02
RANDOM_BASELINE_MRR = 0.02


def _extract_metrics(payload: dict) -> dict[str, float]:
    """Pull ndcg_at_10 and mrr_at_10 out of an MTEB task result dict."""
    found: dict[str, float] = {}
    for split_scores in (payload.get("scores") or {}).values():
        for entry in split_scores:
            for key in ("ndcg_at_10", "mrr_at_10", "recall_at_10", "map_at_10"):
                if key in entry and key not in found:
                    found[key] = float(entry[key])
    return found


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the official MTEB AppsRetrieval result JSON on CPU."
    )
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="dense backbone")
    parser.add_argument("--output", type=Path, default=Path("appsretrieval_results.json"))
    parser.add_argument("--batch-size", type=int, default=64, help="required by the brief")
    parser.add_argument("--lexical-weight", type=float, default=0.35,
                        help="set 0.0 for a dense-only ablation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-sanity-check", action="store_true",
                        help="write the JSON even if scores look like random noise")
    args = parser.parse_args()

    try:
        import mteb
    except ImportError as error:
        raise SystemExit(
            "Install evaluator dependencies first: pip install -r requirements-eval.txt"
        ) from error

    from app.retrieval.mteb_encoder import PrePostPipelineEncoder

    encoder = PrePostPipelineEncoder(
        model_name=args.model,
        device=args.device,
        lexical_weight=args.lexical_weight,
    )

    task = mteb.get_task("AppsRetrieval")
    results = mteb.evaluate(
        encoder,
        [task],
        encode_kwargs={"batch_size": args.batch_size},
    )

    task_result = next(iter(results.task_results))
    payload = task_result.to_dict()
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    metrics = _extract_metrics(payload)
    print(f"Wrote {args.output}")
    for name, value in sorted(metrics.items()):
        print(f"  {name}: {value:.4f}")

    ndcg = metrics.get("ndcg_at_10")
    mrr = metrics.get("mrr_at_10")
    suspicious = (ndcg is not None and ndcg < RANDOM_BASELINE_NDCG) or (
        mrr is not None and mrr < RANDOM_BASELINE_MRR
    )
    if suspicious and not args.skip_sanity_check:
        print(
            "\nSANITY CHECK FAILED: scores are at or below random-retrieval level.\n"
            "This usually means the encoder is not actually embedding anything "
            "(untrained weights, random vectors, or a broken preprocessing stage).\n"
            "Do NOT upload this file as a release artifact until it is investigated.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
