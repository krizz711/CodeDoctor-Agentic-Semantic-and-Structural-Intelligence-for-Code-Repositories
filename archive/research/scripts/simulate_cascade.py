"""POST-1: an evidence-driven cascade, simulated exactly from recorded outcomes.

    python scripts/simulate_cascade.py \
        --small data/outcomes/f2llm160m_all_dev.jsonl.gz --small-params 0.16 \
        --large data/outcomes/f2llm06b_all_dev.jsonl.gz  --large-params 0.6

The FAQ prices what is paid *per query*; the corpus is encoded once, offline.
So the question is not "is a cascade cheaper to index" (it is not: both tiers
need document vectors) but "how often does a query need the large encoder at
all".  Policy:

1. encode the query with the small tier, retrieve, execute its top-N;
2. if a candidate reproduces the statement's examples (a *strong* verdict),
   stop: that is the answer, and the large encoder never runs;
3. otherwise escalate: encode with the large tier, retrieve, execute its top-N
   (reusing any execution already done), rank verified-first as usual.

Execution evidence depends only on (statement, document), so outcomes recorded
from either tier's pool are merged by document id and the cascade is replayed
exactly.  Reported against the large tier alone: NDCG@10 with a paired
bootstrap, the escalation rate, executions per query, and query-encoding cost
in parameter-weighted units (a proxy until CPU latency is measured).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.replay_outcomes import (  # noqa: E402
    Policy,
    candidate_tier,
    load_store,
    metrics,
    paired_bootstrap,
)


def _ranked(entry: dict, evidence: dict[str, dict], policy: Policy, depth: int) -> tuple[list[str], bool, int]:
    """Verified-first ranking of ``entry``'s pool using merged evidence.

    Returns ``(ranking, any_strong, executions_needed)``.
    """
    pool = entry["pool"]
    strong, rest, executed = [], [], 0
    for j, doc in enumerate(pool):
        record = evidence.get(doc) if j < depth else None
        if record is not None:
            executed += 1
            if candidate_tier(dict(record, j=j), entry, policy) == "strong":
                strong.append(doc)
                continue
        rest.append(doc)
    return strong + rest, bool(strong), executed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small", type=Path, required=True)
    parser.add_argument("--large", type=Path, required=True)
    parser.add_argument("--small-params", type=float, required=True, help="billions, for the cost proxy")
    parser.add_argument("--large-params", type=float, required=True)
    parser.add_argument("--qrels", type=Path, default=REPO_ROOT / "artifacts" / "apps" / "qrels_train.json")
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    qrels = json.loads(args.qrels.read_text(encoding="utf-8"))
    gold = {q: next(iter(v)) for q, v in qrels.items()}
    _, small = load_store(args.small)
    _, large = load_store(args.large)
    large_by_q = {e["qid"]: e for e in large}
    policy = Policy("judge-aware", numeric_tolerance=1e-6, statement_tolerance=True, case_rule=True,
                    any_answer_multiset=True, call_style=True, extended_stdin=True)

    rows_cascade, rows_large, rows_small = [], [], []
    escalated = 0
    executions_cascade = executions_large = 0
    cost_cascade = cost_large = 0.0
    for s_entry in small:
        qid = s_entry["qid"]
        l_entry = large_by_q.get(qid)
        if l_entry is None:
            continue
        evidence = {r["doc"]: r for r in l_entry["candidates"]}
        evidence.update({r["doc"]: r for r in s_entry["candidates"]})

        large_rank, _, large_exec = _ranked(l_entry, evidence, policy, args.depth)
        small_rank, small_strong, small_exec = _ranked(s_entry, evidence, policy, args.depth)
        rows_large.append(metrics(large_rank, gold[qid]))
        rows_small.append(metrics(small_rank, gold[qid]))
        executions_large += large_exec
        cost_large += args.large_params

        if small_strong:
            rows_cascade.append(metrics(small_rank, gold[qid]))
            executions_cascade += small_exec
            cost_cascade += args.small_params
        else:
            escalated += 1
            rows_cascade.append(metrics(large_rank, gold[qid]))
            already = set(s_entry["pool"][: args.depth])
            extra = sum(1 for d in l_entry["pool"][: args.depth] if d not in already and d in evidence)
            executions_cascade += small_exec + extra
            cost_cascade += args.small_params + args.large_params

    n = len(rows_cascade)
    cascade, large_only, small_only = map(np.array, (rows_cascade, rows_large, rows_small))
    d, lo, hi, p = paired_bootstrap(large_only[:, 0], cascade[:, 0])
    print(f"{n} statements")
    for name, rows in (("small tier + execution", small_only), ("large tier + execution", large_only),
                       ("cascade", cascade)):
        m = rows.mean(axis=0)
        print(f"  {name:24s} NDCG@10 {m[0]:.5f}  MRR@10 {m[1]:.5f}  R@1 {m[2]:.4f}")
    print(f"  cascade vs large: Δ NDCG@10 {d:+.5f} [{lo:+.5f}, {hi:+.5f}]  p={p:.3f}")
    print(f"  escalated to the large encoder: {escalated}/{n} ({escalated / n:.1%})")
    print(f"  query-encoding cost (param-weighted): cascade {cost_cascade / n:.3f} vs large {cost_large / n:.3f} "
          f"-> {1 - cost_cascade / cost_large:.1%} less")
    print(f"  executions per query: cascade {executions_cascade / n:.1f} vs large {executions_large / n:.1f}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "n": n, "escalation_rate": escalated / n,
            "ndcg": {"small": float(small_only[:, 0].mean()), "large": float(large_only[:, 0].mean()),
                     "cascade": float(cascade[:, 0].mean())},
            "delta_vs_large": {"delta": d, "ci": [lo, hi], "p": p},
            "cost_per_query": {"cascade": cost_cascade / n, "large": cost_large / n},
            "executions_per_query": {"cascade": executions_cascade / n, "large": executions_large / n},
        }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
