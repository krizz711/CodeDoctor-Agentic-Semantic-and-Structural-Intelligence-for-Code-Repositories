"""Replay must equal a live run for every policy whose numbers we report.

``replay_outcomes.py --check-legacy`` covers the shipped legacy rule only.
This checks the v2 policies: the live ``apply_execution_rerank`` top-10 against
the replayed top-10 on a seeded sample of statements that have examples.

    python scripts/check_replay_fidelity.py --store data/outcomes/f2llm17b_all_dev_deep50.jsonl.gz

Bar (STRESS_TEST.md, step 8): at least n-1 of n identical per policy.  Live
runs use wall-clock timeouts, so a candidate near its limit can differ; each
mismatch is printed for inspection.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.execution_rerank import ExecutionConfig, apply_execution_rerank  # noqa: E402
from scripts.dev_eval import load_jsonl_texts  # noqa: E402
from scripts.replay_outcomes import Policy, rerank  # noqa: E402

ARMS = {
    "call-style": (dict(call_style=True), Policy("call-style", numeric_tolerance=1e-6, call_style=True)),
    "call-style+deep50": (dict(call_style=True, deepen_to=50),
                          Policy("call-style+deep50", numeric_tolerance=1e-6, call_style=True, deepen_to=50)),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts" / "apps")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    corpus_ids, corpus_texts = load_jsonl_texts(args.artifacts / "texts_corpus.jsonl")
    text_of = dict(zip(corpus_ids, corpus_texts))
    query_ids, query_texts = load_jsonl_texts(args.artifacts / "texts_queries_train.jsonl")
    query_of = dict(zip(query_ids, query_texts))
    with gzip.open(args.store, "rt", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        entries = [e for e in map(json.loads, handle) if e["candidates"]]
    random.Random(args.seed).shuffle(entries)
    sample = entries[: args.n]
    depth = int(header["exec_depth"])

    report = {"store": str(args.store), "n": len(sample), "arms": {}}
    failed = False
    for name, (switches, policy) in ARMS.items():
        config = ExecutionConfig(enabled=True, max_candidates=depth, timeout_seconds=float(header["timeout"]),
                                 numeric_tolerance=1e-6, verified_bonus=10.0, workers=args.workers, **switches)
        width = max(depth, config.deepen_to)
        mismatches = []
        for entry in sample:
            pool = entry["pool"][:width]
            scores, _ = apply_execution_rerank(query_of[entry["qid"]], [text_of[d] for d in pool],
                                               entry["scores"][:width], config)
            live = [pool[j] for j in sorted(range(len(pool)), key=lambda j: -scores[j])]
            if live[:10] != rerank(entry, policy)[:10]:
                mismatches.append(entry["qid"])
        identical = len(sample) - len(mismatches)
        report["arms"][name] = {"identical": identical, "mismatches": mismatches}
        failed |= identical < len(sample) - 1
        print(f"{name}: replay vs live {identical}/{len(sample)} identical top-10"
              + (f"  mismatches: {mismatches}" if mismatches else ""), flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
