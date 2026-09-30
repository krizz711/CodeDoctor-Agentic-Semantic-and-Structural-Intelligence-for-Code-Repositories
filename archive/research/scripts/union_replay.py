"""Replay execution policies over a NEW dense order, reusing recorded outcomes.

A candidate's execution outcome is a property of the (statement, document) pair,
not of its rank, so a dense order produced by a different first pass can be
scored from the outcomes already recorded for the same statement by any store
made with the same recording code.  Pairs nobody recorded are reported as
*missing* and can be executed live with the recorder's exact configuration
(``fill_missing``); a replay with missing pairs in the executed window is never
presented as a result.

Stores are combined only when they parsed the statement identically (same
worked examples and call examples); otherwise the other store's records for that
statement are ignored.

Library use (see scripts/exp_*.py):
    base = load_store_entries(STORES["dev"][0])
    index = outcome_index(base, STORES["dev"])
    entry, missing = entry_for(base[qid], docs, scores, index[qid], depth=50)
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT = REPO_ROOT / "data" / "outcomes"
#: Stores recorded by the same code (R2 re-recorded 0.6B under the 1.7B code);
#: the first of each list is the base whose parse and rules are used.
STORES = {
    "dev": [OUT / "f2llm17b_all_dev_deep50.jsonl.gz", OUT / "f2llm17b_all_dev.jsonl.gz",
            OUT / "f2llm06b_all_dev_r2_deep50.jsonl.gz", OUT / "f2llm06b_all_dev_r2.jsonl.gz"],
    "tune": [OUT / "f2llm17b_all_tune_deep50.jsonl.gz", OUT / "f2llm17b_all_tune.jsonl.gz",
             OUT / "f2llm06b_all_tune_r2_deep50.jsonl.gz", OUT / "f2llm06b_all_tune_r2.jsonl.gz"],
}
_PARSE_KEYS = ("stdin_examples", "call_examples", "legacy_examples", "distinct_expected")


def load_store_entries(path: Path) -> dict[str, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        handle.readline()
        return {e["qid"]: e for e in map(json.loads, handle)}


def outcome_index(base: dict[str, dict], paths: list[Path]) -> dict[str, dict[str, dict]]:
    """qid -> doc -> recorded candidate record (rank field dropped)."""
    index: dict[str, dict[str, dict]] = {q: {} for q in base}
    for path in paths:
        entries = base if path == paths[0] else load_store_entries(path)
        for qid, entry in entries.items():
            if qid not in base:
                continue
            if any(entry.get(k) != base[qid].get(k) for k in _PARSE_KEYS):
                continue
            for record in entry["candidates"]:
                index[qid].setdefault(record["doc"], {k: v for k, v in record.items() if k != "j"})
    return index


def needs_execution(entry: dict) -> bool:
    return bool(entry["stdin_examples"]) or bool(entry["call_examples"])


def entry_for(base_entry: dict, docs: list[str], scores: list[float], records: dict[str, dict],
              depth: int = 50) -> tuple[dict, list[str]]:
    """A store-shaped entry for a new dense order; returns (entry, docs lacking a record)."""
    docs, scores = list(docs[:depth]), [float(s) for s in scores[:depth]]
    candidates, missing = [], []
    if needs_execution(base_entry):
        for j, doc in enumerate(docs):
            if doc in records:
                candidates.append({"j": j, **records[doc]})
            else:
                missing.append(doc)
    entry = {k: v for k, v in base_entry.items() if k not in ("pool", "scores", "candidates")}
    entry.update(pool=docs, scores=scores, candidates=candidates)
    return entry, missing


def fill_missing(pairs: list[tuple[dict, str]], corpus_text: dict[str, str], workers: int = 6,
                 timeout: float = 2.0) -> list[tuple[str, str, dict]]:
    """Execute (base_entry, doc) pairs live with the recorder's configuration.

    Mirrors scripts/record_outcomes.py so a filled record is indistinguishable
    from a recorded one.  Returns (qid, doc, record) triples.
    """
    from app.pipeline.execution_rerank import ExecutionConfig, execute_candidate, is_function_style
    from scripts.record_outcomes import _stdin_record

    config = ExecutionConfig(enabled=True, timeout_seconds=timeout, call_style=True, extended_stdin=True,
                             max_examples=5)

    def run(pair):
        entry, doc = pair
        code = corpus_text[doc]
        record = {"doc": doc, "function_style": is_function_style(code)}
        if entry["stdin_examples"]:
            record["stdin_runs"] = [
                _stdin_record(execute_candidate(code, e["stdin"], config, track_input=True))
                for e in entry["stdin_examples"]
            ]
        if entry["call_examples"] and record["function_style"]:
            import ast
            spec = ((), [(c["callee"], tuple(ast.literal_eval(c["args"])), tuple(ast.literal_eval(c["kwargs"])))
                         for c in entry["call_examples"]])
            outcome = execute_candidate(code, "", config, call_spec=spec)
            record["call_run"] = {"status": outcome.status, "exc": outcome.exception,
                                  "results": outcome.call_results, "detail": outcome.detail or None,
                                  "secs": outcome.seconds}
        return entry["qid"], doc, record

    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(run, pairs))
    print(f"filled {len(out)} missing pairs live in {time.time() - started:.0f}s ({workers} workers)", flush=True)
    return out
