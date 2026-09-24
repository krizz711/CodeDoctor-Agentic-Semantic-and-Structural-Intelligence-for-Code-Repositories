"""POST-7 support: execute ranks beyond the first window, only where it can matter.

    python scripts/deepen_outcomes.py --store data/outcomes/f2llm06b_all_dev.jsonl.gz --to 50

Adaptive deepening executes candidates ranked ``exec_depth+1 .. --to`` only for
statements where *nothing* in the first window reproduced the samples under
the shipped rule -- elsewhere the answer is already found and deeper runs only
add false-verification risk.  This appends those runs (same record format) to a
copy of the store, so ``replay_outcomes.py`` can replay deepening policies.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.execution_rerank import (  # noqa: E402
    ExecutionConfig,
    execute_candidate,
    is_function_style,
)
from scripts.dev_eval import load_jsonl_texts  # noqa: E402
from scripts.record_outcomes import _stdin_record  # noqa: E402
from scripts.replay_outcomes import Policy, candidate_tier, load_store  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "artifacts" / "apps" / "texts_corpus.jsonl")
    parser.add_argument("--to", type=int, default=50)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    header, entries = load_store(args.store)
    corpus_ids, corpus_texts = load_jsonl_texts(args.corpus)
    code_of = dict(zip(corpus_ids, corpus_texts))
    start = int(header["exec_depth"])
    shipped = Policy("legacy+numeric", numeric_tolerance=1e-6)
    config = ExecutionConfig(enabled=True, timeout_seconds=float(header["timeout"]), call_style=True,
                             extended_stdin=True, max_examples=5)

    jobs = []
    for position, entry in enumerate(entries):
        if not entry["stdin_examples"] and not entry["call_examples"]:
            continue
        if any(candidate_tier(r, entry, shipped) == "strong" for r in entry["candidates"]):
            continue
        for j in range(start, min(args.to, len(entry["pool"]))):
            jobs.append((position, j, entry["pool"][j]))
    print(f"{len(entries)} statements; deepening {len({p for p, _, _ in jobs})} with nothing verified "
          f"in the first {start}: {len(jobs):,} extra runs")

    def run(job):
        position, j, doc = job
        entry = entries[position]
        code = code_of[doc]
        record = {"j": j, "doc": doc, "function_style": is_function_style(code), "deep": True}
        if entry["stdin_examples"]:
            record["stdin_runs"] = [
                _stdin_record(execute_candidate(code, e["stdin"], config, track_input=True))
                for e in entry["stdin_examples"]
            ]
        if entry["call_examples"] and record["function_style"]:
            import ast

            spec = ((), [(c["callee"], ast.literal_eval(c["args"]), ast.literal_eval(c["kwargs"]))
                         for c in entry["call_examples"]])
            outcome = execute_candidate(code, "", config, call_spec=spec)
            record["call_run"] = {"status": outcome.status, "exc": outcome.exception,
                                  "results": outcome.call_results, "detail": outcome.detail or None}
        return position, record

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for position, record in pool.map(run, jobs):
            entries[position]["candidates"].append(record)
    for entry in entries:
        entry["candidates"].sort(key=lambda r: r["j"])
    header = dict(header, deepened_to=args.to, deep_runs=len(jobs), deep_seconds=round(time.time() - started, 1))
    out = args.out or args.store.with_name(args.store.name.replace(".jsonl.gz", f"_deep{args.to}.jsonl.gz"))
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")  # atomic: see record_outcomes.py
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(header, _header=True)) + "\n")
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    os.replace(tmp, out)
    print(f"wrote {out} ({len(jobs):,} runs in {time.time() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
