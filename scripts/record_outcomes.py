"""Execute once, replay many: record what every candidate *did*, not a verdict.

``scripts/instrument_execution.py`` stores one bit per candidate (verified or
not) under one fixed comparison rule, so a new comparison rule, a new
promotion policy or a learned verifier each needed a fresh multi-minute
execution pass.  This records the raw evidence instead -- per candidate and per
worked example:

* how the process ended (``completed`` / ``timeout`` / ``output_cap``), exit
  code and exception class;
* its normalised stdout (truncated), so *any* comparator can be replayed;
* how much of stdin it consumed and whether it read past the end;
* for call-style statements and function-style candidates, each call's result.

``scripts/replay_outcomes.py`` then scores any policy over this store in
seconds.  The store never contains qrels: gold positions are recomputed at
replay time from the artifacts, so the evidence and the labels stay separate.

    python scripts/record_outcomes.py --model f2llm06b --split dev --exec-depth 20

Runs on TRAIN-split rows only (TUNE and DEV of an all-family development
split); it refuses to touch the test split.
"""

from __future__ import annotations

import argparse
import gzip
import os
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.execution_rerank import (  # noqa: E402
    ExecutionConfig,
    _normalize_output,
    _statement_examples,
    execute_candidate,
    is_function_style,
    parse_worked_examples,
)
from app.pipeline.query_analysis import analyze_query  # noqa: E402
from app.pipeline.search import build_candidate_pool  # noqa: E402
from app.pipeline.splits import TEST_DOMINANT_FAMILY, make_development_split  # noqa: E402
from app.pipeline.statement import parse_judge_rules  # noqa: E402
from scripts.dev_eval import load_jsonl_texts  # noqa: E402

STDOUT_KEEP = 4000


def _stdin_record(outcome) -> dict:
    return {
        "status": outcome.status,
        "rc": outcome.returncode,
        "exc": outcome.exception,
        "out": _normalize_output(outcome.stdout)[:STDOUT_KEEP],
        "tracked": outcome.stdin_tracked,
        "read": outcome.stdin_chars_read,
        "eof": outcome.stdin_eof_hit,
        "detail": outcome.detail or None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts" / "apps")
    parser.add_argument("--model", default="f2llm06b")
    parser.add_argument("--split", choices=["tune", "dev", "both"], default="dev")
    parser.add_argument(
        "--family",
        choices=["all", "stdin"],
        default="all",
        help="all = every train family (call-style included); stdin = the repo's original DEV",
    )
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--depth", type=int, default=50, help="dense pool kept for replay")
    parser.add_argument("--exec-depth", type=int, default=20, help="candidates executed per query")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=0, help="0 = one per CPU")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--external",
        type=Path,
        default=None,
        help="a directory from build_external_set.py (texts_queries.jsonl, texts_corpus.jsonl, "
             "splits.json) instead of APPS train; embeddings emb_<model>_{queries,corpus}.npy",
    )
    parser.add_argument("--external-split", default="ext_dev", help="key in splits.json")
    parser.add_argument(
        "--hubness-bank",
        type=Path,
        default=None,
        help="query-embedding .npy used as a hubness bank (PRE-5); pools are built from "
             "cosine - beta * mean top-k bank similarity",
    )
    parser.add_argument("--hubness-beta", type=float, default=0.3)
    parser.add_argument("--hubness-k", type=int, default=10)
    args = parser.parse_args(argv)


    workers = args.workers or (os.cpu_count() or 1)
    if args.external is not None:
        art = args.external
        corpus_ids, corpus_texts = load_jsonl_texts(art / "texts_corpus.jsonl")
        corpus_emb = np.load(art / f"emb_{args.model}_corpus.npy").astype(np.float32)
        query_ids, query_texts = load_jsonl_texts(art / "texts_queries.jsonl")
        query_emb = np.load(art / f"emb_{args.model}_queries.npy").astype(np.float32)
        wanted = set(json.loads((art / "splits.json").read_text(encoding="utf-8"))[args.external_split])
        rows = np.array([i for i, q in enumerate(query_ids) if q in wanted], dtype=np.int64)
        args.split = args.external_split
        args.family = "external"
        print(f"external set {art.name}: {len(rows)} queries in {args.external_split}")
    else:
        art = args.artifacts
        corpus_ids, corpus_texts = load_jsonl_texts(art / "texts_corpus.jsonl")
        corpus_emb = np.load(art / f"emb_{args.model}_corpus.npy").astype(np.float32)
        query_ids, query_texts = load_jsonl_texts(art / "texts_queries_train.jsonl")
        query_emb = np.load(art / f"emb_{args.model}_qtrain.npy").astype(np.float32)

        family = None if args.family == "all" else TEST_DOMINANT_FAMILY
        split = make_development_split(query_texts, match_family=family, seed=args.seed)
        rows = {"tune": split.tune, "dev": split.dev, "both": np.concatenate([split.tune, split.dev])}[args.split]
        print(f"{split.describe()}  -> recording {args.split}: {len(rows)} queries")
    rows = np.sort(rows)
    if args.limit:
        rows = rows[: args.limit]

    if args.hubness_bank is not None:
        from scripts.hubness_eval import hub_scores

        bank = np.load(args.hubness_bank).astype(np.float32)
        penalty = hub_scores(corpus_emb, bank, args.hubness_k)
        scores_all = query_emb[rows] @ corpus_emb.T - args.hubness_beta * penalty[None, :]
        order = np.argsort(-scores_all, axis=1, kind="stable")[:, : args.depth]
        pool = type("Pool", (), {})()
        pool.indices = order
        pool.scores = np.take_along_axis(scores_all, order, axis=1)
        print(f"hubness-corrected pools: bank {args.hubness_bank.name} ({len(bank)}), "
              f"beta {args.hubness_beta}, k {args.hubness_k}")
    else:
        pool = build_candidate_pool(query_emb[rows], corpus_emb, depth=args.depth)

    # One config for recording: everything the replay might want is *recorded*,
    # so the comparison rule here is irrelevant (stdout is kept verbatim).
    record_config = ExecutionConfig(
        enabled=True,
        timeout_seconds=args.timeout,
        call_style=True,
        extended_stdin=True,
        max_examples=5,
    )

    jobs = []
    meta = []
    for position, row in enumerate(rows):
        text = query_texts[row]
        legacy = parse_worked_examples(text, limit=2)
        stdin_examples, call_facts = _statement_examples(text, record_config)
        rules = parse_judge_rules(text)
        entry = {
            "qid": query_ids[row],
            "family": analyze_query(text).family.value,
            "rules": {
                "any_answer": rules.any_answer,
                "tolerance": rules.tolerance,
                "case_insensitive": rules.case_insensitive,
                "modulus": rules.modulus,
            },
            "legacy_examples": len(legacy),
            "stdin_examples": [{"stdin": e.stdin, "expected": e.expected_stdout} for e in stdin_examples],
            "call_examples": (
                [
                    {"args": repr(c.args), "kwargs": repr(c.kwargs), "expected": repr(c.expected),
                     "callee": c.callee, "layout": c.layout}
                    for c in call_facts.call_examples
                ]
                if call_facts is not None
                else []
            ),
            "distinct_expected": call_facts.distinct_expected if call_facts is not None else 0,
            "pool": [corpus_ids[i] for i in pool.indices[position]],
            "scores": [round(float(s), 6) for s in pool.scores[position]],
            "candidates": [],
        }
        meta.append((entry, stdin_examples, call_facts))
        if not stdin_examples and call_facts is None:
            continue
        for j in range(min(args.exec_depth, pool.indices.shape[1])):
            jobs.append((position, j, int(pool.indices[position][j])))

    print(f"executing {len(jobs):,} (query, candidate) pairs with {workers} workers ...", flush=True)

    def run(job):
        position, j, doc_index = job
        _, stdin_examples, call_facts = meta[position]
        code = corpus_texts[doc_index]
        record = {"j": j, "doc": corpus_ids[doc_index], "function_style": is_function_style(code)}
        if stdin_examples:
            record["stdin_runs"] = [
                _stdin_record(execute_candidate(code, e.stdin, record_config, track_input=True))
                for e in stdin_examples
            ]
        if call_facts is not None and record["function_style"]:
            spec = ((), [(c.callee, tuple(c.args), tuple(c.kwargs)) for c in call_facts.call_examples])
            outcome = execute_candidate(code, "", record_config, call_spec=spec)
            record["call_run"] = {
                "status": outcome.status,
                "exc": outcome.exception,
                "results": outcome.call_results,
                "detail": outcome.detail or None,
            }
        return position, record

    started = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for position, record in executor.map(run, jobs):
            meta[position][0]["candidates"].append(record)
            done += 1
            if done % 2000 == 0:
                print(f"  {done:,}/{len(jobs):,} in {time.time() - started:.0f}s", flush=True)
    elapsed = time.time() - started

    out = args.out or REPO_ROOT / "data" / "outcomes" / f"{args.model}_{args.family}_{args.split}.jsonl.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Temporary name, then an atomic rename: two processes aimed at the same
    # store can never interleave their bytes into one corrupt file (which is
    # exactly what happened once, with an accidental second queue).
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        header = {
            "_header": True,
            "model": args.model,
            "split": args.split,
            "family": args.family,
            "seed": args.seed,
            "depth": args.depth,
            "exec_depth": args.exec_depth,
            "timeout": args.timeout,
            "workers": workers,
            "seconds": round(elapsed, 1),
            "pairs": len(jobs),
        }
        handle.write(json.dumps(header) + "\n")
        for entry, _, _ in meta:
            entry["candidates"].sort(key=lambda r: r["j"])
            handle.write(json.dumps(entry) + "\n")
    os.replace(tmp, out)
    print(f"recorded {len(jobs):,} pairs over {len(rows)} queries in {elapsed:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
