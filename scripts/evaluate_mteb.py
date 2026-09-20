"""Produce the official MTEB ``AppsRetrieval`` result JSON for submission.

    python scripts/evaluate_mteb.py --device cpu --output appsretrieval_results.json

Runs the full two-pass pipeline through MTEB's own evaluator, so the emitted
JSON is exactly the artifact the screening stage expects — no hand-assembled
scores anywhere in the path.

``--tier interactive`` swaps in the smaller encoder used by the live service;
``--no-rerank`` runs the frozen dense model alone, which is the ablation
baseline.  Both write the same schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.encoder import PrePostPipelineEncoder  # noqa: E402
from app.pipeline.execution_rerank import ExecutionConfig  # noqa: E402
from app.pipeline.reranker import RerankerConfig  # noqa: E402
from app.retrieval.encode import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    FROZEN_CONFIG_PATH,
    CodeEncoder,
    _select_device,
    select_dtype,
)


def get_default_config_path() -> Path:
    """The submission path reads the frozen config, never the live one."""
    if FROZEN_CONFIG_PATH.exists():
        return FROZEN_CONFIG_PATH
    return DEFAULT_CONFIG_PATH


def _read_config(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def get_default_model(config_path: Path | None = None) -> str:
    data = _read_config(config_path or get_default_config_path())
    return data.get("encoder", {}).get("production_model", "codefuse-ai/F2LLM-v2-1.7B")


#: CoIR AppsRetrieval test split, fixed by the dataset revision this pins.
APPS_TEST_QUERIES = 3765


def get_pipeline_config(config_path: Path | None = None) -> dict:
    """The ``pipeline`` block of the frozen config, with safe fallbacks."""
    data = _read_config(config_path or get_default_config_path())
    block = data.get("pipeline", {}) or {}
    return {
        "execution_rerank": bool(block.get("execution_rerank", False)),
        "execution_candidates": int(block.get("execution_candidates", 10)),
        "execution_timeout_seconds": float(block.get("execution_timeout_seconds", 2.0)),
        "execution_numeric_tolerance": (
            None
            if block.get("execution_numeric_tolerance") in (None, "", False)
            else float(block["execution_numeric_tolerance"])
        ),
    }


def get_tier_model(tier: str, config_path: Path | None = None) -> tuple[str | None, str | None]:
    """``(model_name, revision)`` for the requested tier, or ``(None, None)``."""
    data = _read_config(config_path or get_default_config_path())
    section = "interactive_encoder" if tier == "interactive" else "encoder"
    block = data.get(section, {}) or {}
    return block.get("production_model"), block.get("revision")


def build_mteb_encoder(pipeline: PrePostPipelineEncoder, model_name: str) -> Any:
    """Wrap the pipeline in MTEB's ``AbsEncoder`` interface."""
    import mteb
    from mteb.models.abs_encoder import AbsEncoder

    try:
        model_meta = mteb.get_model_meta(model_name)
    except Exception:
        model_meta = type(
            "SimpleModelMeta",
            (),
            {"name": model_name, "revision": "main", "similarity_fn_name": "cosine"},
        )()

    class CodeDoctorMTEBEncoder(AbsEncoder):
        def __init__(self) -> None:
            self.pipeline = pipeline
            self.model = pipeline.encoder.model
            self.mteb_model_meta = model_meta

        def encode(self, inputs, **kwargs) -> np.ndarray:
            return self.pipeline.encode(inputs, **kwargs)

        def similarity(self, embeddings1, embeddings2):
            return self.pipeline.similarity(embeddings1, embeddings2)

    return CodeDoctorMTEBEncoder()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="override the configured production model")
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--tier",
        choices=["accuracy", "interactive"],
        default="accuracy",
        help="accuracy = submission encoder; interactive = the live-service encoder",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None, help="reranker weights JSON")
    parser.add_argument("--no-rerank", action="store_true", help="dense only (ablation baseline)")
    execution_group = parser.add_mutually_exclusive_group()
    execution_group.add_argument(
        "--execution",
        dest="execution",
        action="store_true",
        default=None,
        help="force the execution-verification pass on (default comes from "
             "config/frozen.yaml: pipeline.execution_rerank)",
    )
    execution_group.add_argument(
        "--no-execution",
        dest="execution",
        action="store_false",
        help="force it off. The run then reports the dense-only score.",
    )
    parser.add_argument("--execution-candidates", type=int, default=None)
    parser.add_argument("--execution-timeout", type=float, default=None)
    parser.add_argument("--hubness", type=Path, default=None, help="reference hub profile .npz")
    parser.add_argument("--output", type=Path, default=Path("appsretrieval_results.json"))
    parser.add_argument("--predictions", type=Path, default=None, help="also write ranked CSVs here")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    try:
        import mteb
    except ImportError as error:
        raise SystemExit(
            "Install evaluator dependencies first: pip install -r requirements-eval.txt"
        ) from error

    config_path = args.config or get_default_config_path()
    tier_model, tier_revision = get_tier_model(args.tier, config_path)
    model_name = args.model or tier_model or get_default_model(config_path)
    revision = args.revision or (tier_revision if args.model is None else None)
    device = _select_device(args.device)

    print(f"Config          : {config_path}")
    print(f"Tier            : {args.tier}")
    print(f"Model           : {model_name}" + (f"@{revision[:8]}" if revision else ""))
    print(f"Device          : {device}")

    encoder = CodeEncoder(
        model_name=model_name,
        revision=revision,
        device=device,
        config_path=config_path,
        torch_dtype=select_dtype(device),
    )
    print(f"Query prefix    : {encoder.query_prefix!r}")
    print(f"Document prefix : {encoder.document_prefix!r}")
    print(f"max_seq_length  : {encoder.max_seq_length}   add_eos_token: {encoder.add_eos_token}")

    reranker_config = RerankerConfig.load(args.weights)
    if args.no_rerank:
        reranker_config = RerankerConfig()
        print("Reranker        : DISABLED (--no-rerank, dense-only ablation)")
    elif reranker_config.is_identity:
        print("Reranker        : identity (no fitted weights found -- dense only)")
    else:
        print(f"Reranker        : {reranker_config.describe()}")

    pipeline_defaults = get_pipeline_config(config_path)
    execution_config = ExecutionConfig(
        enabled=(
            pipeline_defaults["execution_rerank"] if args.execution is None else args.execution
        ),
        max_candidates=args.execution_candidates or pipeline_defaults["execution_candidates"],
        timeout_seconds=args.execution_timeout or pipeline_defaults["execution_timeout_seconds"],
        numeric_tolerance=pipeline_defaults["execution_numeric_tolerance"],
    )
    if execution_config.enabled:
        print(f"Execution pass  : ON (top {execution_config.max_candidates}, "
              f"{execution_config.timeout_seconds}s timeout)")
        print("                  This RUNS PYTHON FROM THE DATASET in a sandboxed")
        print("                  subprocess. Pass --no-execution to disable it.")
    else:
        print("Execution pass  : off")

    pipeline = PrePostPipelineEncoder(
        encoder, reranker_config, execution_config=execution_config
    )
    if args.hubness and args.hubness.exists():
        pipeline.hub_profile_path = args.hubness
        print(f"Hub profile     : {args.hubness} (aligned by document content)")

    mteb_model = build_mteb_encoder(pipeline, model_name)

    print("\nFetching task AppsRetrieval ...")
    task = mteb.get_task("AppsRetrieval")
    print("Evaluating ...")
    evaluate_kwargs: dict[str, Any] = {
        "encode_kwargs": {"batch_size": args.batch_size},
        "cache": None,
        "overwrite_strategy": "always",
    }
    if args.predictions:
        args.predictions.mkdir(parents=True, exist_ok=True)
        evaluate_kwargs["prediction_folder"] = str(args.predictions)

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
    results = mteb.evaluate(mteb_model, [task], **evaluate_kwargs)
    task_result = list(results.task_results)[0]
    payload = task_result.to_dict()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")

    scores = payload.get("scores", {}).get("test", [{}])[0]
    print(f"\nNDCG@10 = {scores.get('ndcg_at_10')}   MRR@10 = {scores.get('mrr_at_10')}")
    print(f"Wrote {args.output}")

    # This is a test-split evaluation, so it belongs in the same audit log as
    # every other one.  Until 2026-09-19 it was not recorded: the log only ever
    # saw DevelopmentHarness.get_test_split, while the runs that produced the
    # submitted numbers all came through here.  The control existed and did not
    # cover the path that mattered.
    #
    # Recorded, never refused.  A budget that silently blocks the submission
    # run would be worse than one that reports honestly, and the firewall's job
    # on this path is disclosure rather than prevention.
    try:
        from app.evaluation_harness import record_persistent_audit

        record_persistent_audit(
            reason="final_run",
            row_count=APPS_TEST_QUERIES,
            caller="scripts/evaluate_mteb.py",
            extra={
                "metric_ndcg_at_10": scores.get("ndcg_at_10"),
                "metric_mrr_at_10": scores.get("mrr_at_10"),
                "model": model_name,
                "execution_rerank": execution_config.enabled,
                "execution_candidates": execution_config.max_candidates,
                "execution_numeric_tolerance": execution_config.numeric_tolerance,
                "enforced": False,
            },
        )
        print("Logged test-split access to data/test_split_audit_log.json")
    except Exception as error:  # pragma: no cover - never fail a run over logging
        print(f"WARNING: could not record the test-split access: {error}")

    if args.predictions:
        from scripts.export_submission_csv import export_from_predictions

        written = export_from_predictions(args.predictions, payload, args.output.parent)
        for path in written:
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
