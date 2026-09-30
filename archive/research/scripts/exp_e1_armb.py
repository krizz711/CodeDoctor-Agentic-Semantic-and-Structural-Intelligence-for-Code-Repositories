"""E1 arm B (experiments/C3_cycle/PREREG.md): judge-like I/O, a fidelity fix with no score claim.

    python scripts/exp_e1_armb.py --out experiments/C3_cycle/E1_armB_dev.json

Every recorded stdin run that did not verify, for candidates whose code uses the
``exit``/``quit`` or ``fstat(0)``/``os.read(0)`` idioms (the selection of
skeptic_scripts/armb_tune.py), is re-run with ``judge_like_io`` on and scored
by replay against the unchanged store.  Accepted iff no statement loses NDCG.
Candidates outside the selection are not re-run; the result covers only them.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.execution_rerank import ExecutionConfig, execute_candidate, spawn_failures  # noqa: E402
from scripts.c3_stats import compare  # noqa: E402
from scripts.record_outcomes import _stdin_record  # noqa: E402
from scripts.replay_outcomes import Policy, candidate_tier, load_store, metrics, rerank  # noqa: E402

EXIT = re.compile(r"(?<![\w.])(?:exit|quit)\s*\(")
FSTAT = re.compile(r"fstat\s*\(\s*0|os\.read\s*\(\s*0")
V2 = Policy("call-style+deep50", numeric_tolerance=1e-6, call_style=True, deepen_to=50)
STOP_SECONDS = 1800
SETS = {
    "dev": (REPO_ROOT / "data/outcomes/f2llm17b_all_dev_deep50.jsonl.gz", REPO_ROOT / "artifacts/apps/texts_corpus.jsonl",
            REPO_ROOT / "artifacts/apps/qrels_train.json"),
    "extcc": (REPO_ROOT / "data/outcomes/f2llm17b_extcc_dev_deep50.jsonl.gz", REPO_ROOT / "artifacts/ext_cc/texts_corpus.jsonl",
              REPO_ROOT / "artifacts/ext_cc/qrels.json"),
}


def run_set(name: str, workers: int) -> dict:
    store, corpus_path, qrels_path = SETS[name]
    header, entries = load_store(store)
    with open(corpus_path, encoding="utf-8") as handle:
        corpus = {r["id"]: r["text"] for r in map(json.loads, handle)}
    gold = {q: next(iter(v)) for q, v in json.loads(qrels_path.read_text(encoding="utf-8")).items()}
    entries = [e for e in entries if e["qid"] in gold]
    config = ExecutionConfig(enabled=True, timeout_seconds=float(header["timeout"]), call_style=True,
                             extended_stdin=True, max_examples=5, judge_like_io=True)
    treated = copy.deepcopy(entries)
    jobs = []
    for entry in treated:
        if not entry["stdin_examples"]:
            continue
        for record in entry["candidates"]:
            code = corpus[record["doc"]]
            if record.get("stdin_runs") and (EXIT.search(code) or FSTAT.search(code)) \
                    and candidate_tier(record, entry, V2) != "strong":
                jobs.append((entry, record))

    def run(job):
        entry, record = job
        return [_stdin_record(execute_candidate(corpus[record["doc"]], e["stdin"], config, track_input=True))
                for e in entry["stdin_examples"]]

    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (entry, record), runs in zip(jobs, pool.map(run, jobs)):
            record["stdin_runs"] = runs
    seconds = time.time() - started
    before = np.array([metrics(rerank(e, V2), gold[e["qid"]])[0] for e in entries])
    after = np.array([metrics(rerank(e, V2), gold[e["qid"]])[0] for e in treated])
    newly = sum(candidate_tier(r, e, V2) == "strong" for e in treated for r in e["candidates"]
                if any(r is rec for _, rec in jobs))
    return {"set": name, "candidates_rerun": len(jobs), "seconds": round(seconds, 1), "workers": workers,
            "newly_verified": newly, "ndcg10": compare(before, after),
            "statements_lower": [e["qid"] for e, b, a in zip(entries, before, after) if a < b],
            "statements_higher": [e["qid"] for e, b, a in zip(entries, before, after) if a > b],
            "stopped": seconds > STOP_SECONDS, "spawn_failures": spawn_failures()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    result = {"dev": run_set("dev", args.workers)}
    result["extcc_descriptive_in_sample"] = run_set("extcc", args.workers)
    dev = result["dev"]
    result["accepted"] = (not dev["statements_lower"] and not dev["stopped"] and dev["spawn_failures"] == 0)
    result["note"] = "CI on ubuntu/macOS/Windows is the other acceptance condition (tests/test_judge_like_io.py)"
    text = json.dumps(result, indent=1)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
