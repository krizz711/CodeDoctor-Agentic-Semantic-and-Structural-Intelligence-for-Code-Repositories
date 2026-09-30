"""E1: record the blank-line fallback runs for a store, as a sidecar.

    python scripts/record_blank_fallback.py --store data/outcomes/f2llm17b_all_dev_deep50.jsonl.gz
    python scripts/record_blank_fallback.py --store ... --placebo      # the negative control

For every statement whose first two stdin examples (the ones the policies use)
contain an internal blank line, every recorded candidate is run once more on the
input with the blank lines removed, with the recorder's configuration and input
tracking.  The original store is not touched, so the unmodified replay stays
byte-identical; ``replay_outcomes.attach_fallback`` merges the sidecar in.

``--placebo`` (pre-registered negative control, experiments/C3_cycle/PREREG.md):
each affected statement gets the collapsed first example of a *different*
affected statement (a fixed-seed derangement) instead of its own, judged against
its own expected output.  It must not recover golds.

``--retry`` (pre-registered retry control): the same extra run, on the
*uncollapsed* input -- what a plain second chance would recover.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.execution_rerank import (  # noqa: E402
    ExecutionConfig,
    collapse_blank_lines,
    execute_candidate,
    has_internal_blank_line,
    spawn_failures,
)
from scripts.record_outcomes import _stdin_record  # noqa: E402

#: the policies read the first two recorded stdin examples (Policy.stdin_examples)
USED_EXAMPLES = 2


def affected(entry: dict) -> list[int]:
    """Indices of used stdin examples that contain an internal blank line."""
    if entry["legacy_examples"] <= 0:
        return []
    return [i for i, e in enumerate(entry["stdin_examples"][:USED_EXAMPLES]) if has_internal_blank_line(e["stdin"])]


def derangement(items: list[str], seed: int) -> dict[str, str]:
    order = sorted(items)
    rng = random.Random(seed)
    shuffled = order[:]
    while True:
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(order, shuffled)):
            return dict(zip(order, shuffled))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--placebo", action="store_true")
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "artifacts" / "apps" / "texts_corpus.jsonl")
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args(argv)
    if args.placebo and args.retry:
        parser.error("--placebo and --retry are separate controls")

    with gzip.open(args.store, "rt", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        entries = {e["qid"]: e for e in map(json.loads, handle)}
    hit = {qid: affected(e) for qid, e in entries.items()}
    hit = {qid: idx for qid, idx in hit.items() if idx}
    partner = derangement(list(hit), args.seed) if args.placebo else {}

    inputs: dict[str, dict[str, str]] = {}
    for qid, idx in hit.items():
        if args.placebo:
            other = entries[partner[qid]]["stdin_examples"][0]["stdin"]
            inputs[qid] = {str(i): collapse_blank_lines(other) for i in idx}
        elif args.retry:
            inputs[qid] = {str(i): entries[qid]["stdin_examples"][i]["stdin"] for i in idx}
        else:
            inputs[qid] = {str(i): collapse_blank_lines(entries[qid]["stdin_examples"][i]["stdin"]) for i in idx}

    config = ExecutionConfig(enabled=True, timeout_seconds=float(header["timeout"]), call_style=True,
                             extended_stdin=True, max_examples=5)
    jobs = [(qid, record["doc"], i, inputs[qid][str(i)])
            for qid in hit for record in entries[qid]["candidates"] if record.get("stdin_runs")
            for i in hit[qid]]
    corpus = {}
    with open(args.corpus, encoding="utf-8") as handle:
        for row in map(json.loads, handle):
            corpus[row["id"]] = row["text"]

    def run(job):
        qid, doc, i, text = job
        return qid, doc, i, _stdin_record(execute_candidate(corpus[doc], text, config, track_input=True))

    mode = "placebo" if args.placebo else "retry" if args.retry else "collapsed"
    print(f"{len(hit)} affected statements, {len(jobs)} fallback runs, {args.workers} workers ({mode})", flush=True)
    started = time.time()
    runs: dict[str, dict[str, dict[str, dict]]] = {qid: {} for qid in hit}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for qid, doc, i, record in pool.map(run, jobs):
            runs[qid].setdefault(doc, {})[str(i)] = record
    elapsed = time.time() - started

    suffix = {"placebo": "_blankfb_placebo", "retry": "_blankfb_retry"}.get(mode, "_blankfb")
    out = args.out or args.store.with_name(args.store.name.replace(".jsonl.gz", f"{suffix}.jsonl.gz"))
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"_header": True, "store": args.store.name, "placebo": args.placebo, "mode": mode,
                                 "seed": args.seed, "timeout": header["timeout"], "workers": args.workers,
                                 "runs": len(jobs), "seconds": round(elapsed, 1),
                                 "spawn_failures": spawn_failures()}) + "\n")
        for qid in sorted(hit):
            handle.write(json.dumps({"qid": qid, "inputs": inputs[qid], "partner": partner.get(qid),
                                     "runs": runs[qid]}) + "\n")
    os.replace(tmp, out)
    print(f"recorded {len(jobs)} runs in {elapsed:.0f}s, spawn failures {spawn_failures()} -> {out}")
    return 0 if spawn_failures() == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
