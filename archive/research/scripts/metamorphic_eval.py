"""Measure metamorphic sample amplification on a recorded outcome store.

    python scripts/metamorphic_eval.py --store data/outcomes/f2llm06b_extcc_dev.jsonl.gz \
        --queries artifacts/ext_cc/texts_queries.jsonl --corpus artifacts/ext_cc/texts_corpus.jsonl \
        --qrels artifacts/ext_cc/qrels.json

For every candidate that reproduces the samples (shipped rule), run the tests
derived by ``app.pipeline.metamorphic`` and report:

* soundness      -- sample-verified *gold* solutions that also pass the
  derived tests (should be ~100%; failures mean the relation did not hold);
* discrimination -- sample-verified *non-gold* candidates the derived tests
  reject;
* ranking        -- verified candidates that also pass the derived tests are
  placed above verified candidates that fail them.

Gold labels are used only to measure.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.execution_rerank import (  # noqa: E402
    ExecutionConfig,
    WorkedExample,
    compare_output,
    execute_candidate,
)
from app.pipeline.metamorphic import derive  # noqa: E402
from scripts.dev_eval import load_jsonl_texts  # noqa: E402
from scripts.replay_outcomes import Policy, candidate_tier, load_store, metrics, paired_bootstrap  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--copies", type=int, default=2)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    header, entries = load_store(args.store)
    qids, qtexts = load_jsonl_texts(args.queries)
    statement_of = dict(zip(qids, qtexts))
    cids, ctexts = load_jsonl_texts(args.corpus)
    code_of = dict(zip(cids, ctexts))
    qrels = json.loads(args.qrels.read_text(encoding="utf-8"))
    gold = {q: next(iter(v)) for q, v in qrels.items()}
    policy = Policy("legacy+numeric", numeric_tolerance=1e-6)
    config = ExecutionConfig(enabled=True, timeout_seconds=float(header["timeout"]), numeric_tolerance=1e-6)

    jobs, derived_by_query, eligible = [], {}, 0
    for entry in entries:
        if entry["legacy_examples"] == 0:
            continue
        examples = [WorkedExample(e["stdin"], e["expected"]) for e in entry["stdin_examples"][:2]]
        tests = derive(statement_of[entry["qid"]], examples, copies=args.copies)
        derived_by_query[entry["qid"]] = tests
        if not tests:
            continue
        eligible += 1
        for record in entry["candidates"]:
            if candidate_tier(record, entry, policy) == "strong":
                jobs.append((entry["qid"], record["doc"]))

    def run(job):
        qid, doc = job
        for test in derived_by_query[qid]:
            outcome = execute_candidate(code_of[doc], test.example.stdin, config)
            if not outcome.ok or compare_output(outcome.stdout, test.example.expected_stdout, config) is None:
                return qid, doc, False
        return qid, doc, True

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, jobs))
    passed = {(q, d): ok for q, d, ok in results}

    gold_rows = [ok for (q, d), ok in passed.items() if d == gold[q]]
    other_rows = [ok for (q, d), ok in passed.items() if d != gold[q]]
    print(f"{args.store.name}: {len(entries)} statements, {eligible} with a multi-test sample "
          f"(metamorphic test derivable); {len(jobs)} sample-verified candidates re-run")
    print(f"  soundness: sample-verified golds passing the derived test: "
          f"{sum(gold_rows)}/{len(gold_rows)} ({np.mean(gold_rows) if gold_rows else float('nan'):.1%})")
    print(f"  discrimination: sample-verified non-golds rejected: "
          f"{len(other_rows) - sum(other_rows)}/{len(other_rows)} "
          f"({1 - np.mean(other_rows) if other_rows else float('nan'):.1%})")

    base_rows, meta_rows = [], []
    for entry in entries:
        pool_docs = entry["pool"]
        tiers = {r["doc"]: candidate_tier(r, entry, policy) for r in entry["candidates"]}
        strong = [d for d in pool_docs if tiers.get(d) == "strong"]
        rest = [d for d in pool_docs if tiers.get(d) != "strong"]
        base_rank = strong + rest
        confirmed = [d for d in strong if passed.get((entry["qid"], d), True)]
        refuted = [d for d in strong if not passed.get((entry["qid"], d), True)]
        meta_rank = confirmed + refuted + rest
        base_rows.append(metrics(base_rank, gold[entry["qid"]]))
        meta_rows.append(metrics(meta_rank, gold[entry["qid"]]))
    base, meta = np.array(base_rows), np.array(meta_rows)
    d, lo, hi, p = paired_bootstrap(base[:, 0], meta[:, 0])
    print(f"  ranking: NDCG@10 {base[:, 0].mean():.5f} -> {meta[:, 0].mean():.5f}  "
          f"delta {d:+.5f} [{lo:+.5f}, {hi:+.5f}]  p={p:.3f}   R@1 {base[:, 2].mean():.4f} -> {meta[:, 2].mean():.4f}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "store": args.store.name, "eligible_statements": eligible, "rerun": len(jobs),
            "soundness": {"passed": int(sum(gold_rows)), "of": len(gold_rows)},
            "discrimination": {"rejected": len(other_rows) - int(sum(other_rows)), "of": len(other_rows)},
            "ndcg": {"base": float(base[:, 0].mean()), "metamorphic": float(meta[:, 0].mean())},
            "delta": {"delta": d, "ci": [lo, hi], "p": p},
        }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
