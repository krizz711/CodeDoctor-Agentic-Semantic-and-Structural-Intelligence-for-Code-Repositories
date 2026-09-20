"""Run the shipped pipeline against a retrieval task it was never tuned on.

    python scripts/blind_eval.py --task CosQA --tier interactive

The point of this script is what it does NOT contain. Every component -- the
encoder, the prefixes, the reranker weights, the execution pass -- is imported
from ``scripts/evaluate_mteb.py``, the module that produces the screening
submission. The only thing that changes is the dataset handed to
``mteb.evaluate``. If the number here collapses, that is the system failing to
generalise, not a different system being measured.

It also counts, before evaluating, how many of the task's queries carry worked
examples that ``parse_worked_examples`` can read. That number is the ceiling on
what execution verification can possibly contribute: on AppsRetrieval it is
~98%, and on a task of short natural-language questions it is 0. Reporting it
stops the execution pass from being credited on data where it never ran.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.encoder import PrePostPipelineEncoder  # noqa: E402
from app.pipeline.execution_rerank import (  # noqa: E402
    ExecutionConfig,
    parse_worked_examples,
)
from app.pipeline.reranker import RerankerConfig  # noqa: E402
from app.retrieval.encode import CodeEncoder, _select_device, select_dtype  # noqa: E402

# The submission path itself. Imported, not copied.
from scripts.evaluate_mteb import (  # noqa: E402
    build_mteb_encoder,
    get_default_config_path,
    get_pipeline_config,
    get_tier_model,
)


def count_execution_eligible(task: Any) -> tuple[int, int]:
    """``(eligible, total)`` test queries that carry parseable worked examples.

    mteb 2.x exposes loaded splits as
    ``task.dataset[subset][split]["queries"]`` -- a HF Dataset, not the
    ``task.queries`` mapping older versions had. Getting this wrong returns
    (-1, -1) rather than raising, so a failure here never costs an eval run.
    """
    try:
        dataset = task.dataset
    except Exception:
        return (-1, -1)
    if not isinstance(dataset, dict):
        return (-1, -1)

    texts: list[str] = []
    for subset in dataset.values():
        if not isinstance(subset, dict):
            continue
        for split_name, split in subset.items():
            if split_name != "test" or not isinstance(split, dict):
                continue
            queries = split.get("queries")
            if queries is None:
                continue
            for row in queries:
                if isinstance(row, str):
                    texts.append(row)
                elif isinstance(row, dict):
                    body = row.get("text") or row.get("query") or ""
                    title = row.get("title") or ""
                    texts.append(" ".join(str(x) for x in (title, body) if x))
    if not texts:
        return (-1, -1)
    return (sum(1 for t in texts if parse_worked_examples(t)), len(texts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="MTEB task name, e.g. CosQA")
    parser.add_argument(
        "--tier",
        choices=["accuracy", "interactive"],
        default="interactive",
        help="interactive = the 0.6B live-service encoder (the only tier whose "
             "weights are present on a CPU-only machine)",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-execution", dest="execution", action="store_false", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        import mteb
    except ImportError as error:
        raise SystemExit("pip install -r requirements-eval.txt") from error

    config_path = args.config or get_default_config_path()
    model_name, revision = get_tier_model(args.tier, config_path)
    device = _select_device(args.device)

    print(f"Task            : {args.task}   (BLIND -- never used for tuning)")
    print(f"Config          : {config_path}")
    print(f"Tier            : {args.tier}")
    print(f"Model           : {model_name}@{(revision or '')[:8]}")
    print(f"Device          : {device}")

    encoder = CodeEncoder(
        model_name=model_name,
        revision=revision,
        device=device,
        config_path=config_path,
        torch_dtype=select_dtype(device),
    )
    print(f"Query prefix    : {encoder.query_prefix!r}")
    print(f"max_seq_length  : {encoder.max_seq_length}   add_eos_token: {encoder.add_eos_token}")

    reranker_config = RerankerConfig.load(None)
    print("Reranker        : " + ("identity" if reranker_config.is_identity
                                  else reranker_config.describe()))

    defaults = get_pipeline_config(config_path)
    execution_config = ExecutionConfig(
        enabled=defaults["execution_rerank"] if args.execution is None else args.execution,
        max_candidates=defaults["execution_candidates"],
        timeout_seconds=defaults["execution_timeout_seconds"],
        numeric_tolerance=defaults["execution_numeric_tolerance"],
    )
    print(f"Execution pass  : {'ON' if execution_config.enabled else 'off'} "
          f"(top {execution_config.max_candidates})")

    pipeline = PrePostPipelineEncoder(encoder, reranker_config,
                                      execution_config=execution_config)
    mteb_model = build_mteb_encoder(pipeline, model_name)

    print(f"\nFetching task {args.task} ...")
    task = mteb.get_task(args.task)
    try:
        task.load_data()
    except Exception as error:
        print(f"  (load_data: {error})")

    eligible, total = count_execution_eligible(task)
    if total > 0:
        pct = 100.0 * eligible / total
        print(f"Execution-eligible queries: {eligible}/{total} ({pct:.1f}%)"
              " -- the ceiling on what the execution pass can contribute here")
    else:
        print("Execution-eligible queries: could not inspect this task's queries")

    print("Evaluating ...")
    started = time.perf_counter()
    # cache=None is load-bearing, not tidiness.
    #
    # mteb.evaluate defaults to cache=ResultCache(~/.cache/mteb) with
    # overwrite_strategy="only-missing". build_mteb_encoder advertises the
    # UPSTREAM model meta (name + revision), so on any machine that has pulled
    # MTEB's public results repository the evaluator finds a published result
    # for codefuse-ai/F2LLM-v2-* at this exact revision and returns it WITHOUT
    # running a single forward pass. It comes back in ~11s wearing the base
    # encoder's score -- 0.34186 on CodeTransOceanDL, 0.93692 on AppsRetrieval
    # -- and nothing in the output says the pipeline never ran.
    #
    # The submitted 0.96558 predates this hazard only because it was produced
    # on a machine with a cold cache. Disabling the cache is what makes the
    # number belong to this pipeline.
    results = mteb.evaluate(
        mteb_model,
        [task],
        encode_kwargs={"batch_size": args.batch_size},
        cache=None,
        overwrite_strategy="always",
    )
    elapsed = time.perf_counter() - started

    task_result = list(results.task_results)[0]
    payload = task_result.to_dict()
    scores = payload.get("scores", {}).get("test", [{}])[0]

    payload["codedoctor_blind_run"] = {
        "task": args.task,
        "tier": args.tier,
        "model": model_name,
        "revision": revision,
        "device": device,
        "execution_enabled": execution_config.enabled,
        "execution_eligible_queries": eligible,
        "total_queries": total,
        "wall_clock_seconds": round(elapsed, 1),
        "tuned_on_this_task": False,
    }

    out = args.output or REPO_ROOT / "data" / "blind" / f"{args.task}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")

    print(f"\n{args.task}:  NDCG@10 = {scores.get('ndcg_at_10')}   "
          f"MRR@10 = {scores.get('mrr_at_10')}   "
          f"Recall@10 = {scores.get('recall_at_10')}")
    print(f"Wall clock: {elapsed:.1f}s")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
