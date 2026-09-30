"""POST-6: a small, sign-constrained verifier over execution evidence (LEVER-style).

    python scripts/fit_evidence_fusion.py --tune data/outcomes/f2llm06b_all_tune.jsonl.gz \
        --dev data/outcomes/f2llm06b_all_dev.jsonl.gz

The shipped rule is binary: a candidate that reproduces every worked example is
lifted above everything, and nothing else about its run is used.  LEVER (Ni et
al., ICML'23) showed that a verifier learned over *execution results* plus the
generator's score beats such rules.  This fits the smallest honest version of
that idea:

* features per (statement, candidate) in the executed top-N: the dense score,
  its gap to the row's best, the dense rank; the fraction of worked examples
  reproduced (judge-aware comparator), parse-type crashes (ValueError /
  EOFError / IndexError -- the program expected a different input), other
  crashes and timeouts, same-shape output on any-answer statements, exact
  consumption of the input; call-style pass fraction and promotability;
  structural incompatibility (a function-only candidate for a stdin
  statement);
* logistic regression, L2-regularised, with *sign constraints* where the
  direction is not in doubt (reproducing more examples can never lower the
  score; nor can the dense score);
* fitted on TUNE rows only, evaluated once on DEV against the binary rules
  with a paired bootstrap.

The nine-feature reranker failed because its features restated what the
encoder already knows.  Execution features do not -- that is the hypothesis
this tests.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.replay_outcomes import (  # noqa: E402
    Policy,
    call_verdict,
    load_store,
    metrics,
    paired_bootstrap,
    rerank,
    stdin_verdict,
)

JUDGE = Policy("judge-aware", numeric_tolerance=1e-6, statement_tolerance=True, case_rule=True,
               any_answer_multiset=True, any_answer_shape=True, call_style=True, extended_stdin=True)
PARSE_CRASHES = {"ValueError", "EOFError", "IndexError", "TypeError"}

FEATURES = [
    # name, sign constraint: +1 non-negative, -1 non-positive, 0 free
    ("dense", +1),
    ("gap_to_best", 0),
    ("log_rank", 0),
    ("stdin_pass_frac", +1),
    ("stdin_all_pass", +1),
    ("stdin_parse_crash", 0),
    ("stdin_other_failure", 0),
    ("stdin_shape_match", +1),
    ("stdin_consumed_frac", 0),
    ("call_pass_frac", +1),
    ("call_promotable", +1),
    ("incompatible", 0),
    ("has_examples", 0),
]


def candidate_features(entry: dict, record: dict) -> list[float]:
    j = record["j"]
    scores = entry["scores"]
    runs = record.get("stdin_runs") or []
    use_stdin = entry["legacy_examples"] > 0 or JUDGE.extended_stdin
    pairs = list(zip(runs, entry["stdin_examples"]))[: JUDGE.stdin_examples] if use_stdin else []
    verdicts = [stdin_verdict(r, e["expected"], entry["rules"], JUDGE) for r, e in pairs]
    strong = [v == "strong" for v in verdicts]
    weak = [v in ("strong", "weak") for v in verdicts]
    parse_crash = any(
        r["status"] == "completed" and r["rc"] != 0 and (r.get("exc") in PARSE_CRASHES) for r, _ in pairs
    )
    other_failure = any(
        (r["status"] != "completed") or (r["rc"] != 0 and r.get("exc") not in PARSE_CRASHES) for r, _ in pairs
    )
    consumed = []
    for r, e in pairs:
        if r.get("tracked"):
            consumed.append(float((r.get("read") or 0) >= len(e["stdin"].rstrip())))
    call_run = record.get("call_run")
    call_pass = 0.0
    call_strong = 0.0
    if call_run and entry["call_examples"] and record.get("function_style"):
        results = call_run.get("results") or []
        # per-example pass fraction under the same structural comparison
        passed = 0
        for example, result in zip(entry["call_examples"], results):
            single = dict(entry, call_examples=[example], distinct_expected=99)
            if call_verdict({"call_run": dict(call_run, results=[result])}, single,
                            Policy("x", call_style=True, call_min_examples=1, call_min_distinct=1)) == "strong":
                passed += 1
        call_pass = passed / max(1, len(entry["call_examples"]))
        call_strong = float(call_verdict(record, entry, JUDGE) == "strong")
    stdin_family = entry["family"] == "competitive_stdin"
    return [
        scores[j],
        scores[0] - scores[j],
        math.log(j + 1),
        float(np.mean(strong)) if strong else 0.0,
        float(bool(strong) and all(strong)),
        float(parse_crash),
        float(other_failure),
        float(bool(weak) and all(weak) and not all(strong)),
        float(np.mean(consumed)) if consumed else 0.0,
        call_pass,
        call_strong,
        float(stdin_family and record.get("function_style", False)),
        float(bool(pairs) or bool(entry["call_examples"])),
    ]


def build(entries: list[dict], gold: dict[str, str]):
    X, y, groups = [], [], []
    for qi, entry in enumerate(entries):
        g = gold[entry["qid"]]
        for record in entry["candidates"]:
            X.append(candidate_features(entry, record))
            y.append(float(record["doc"] == g))
            groups.append(qi)
    return np.array(X, dtype=np.float64), np.array(y), np.array(groups)


def fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0):
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = (X - mean) / std
    bounds = [(0, None) if s > 0 else (None, 0) if s < 0 else (None, None) for _, s in FEATURES]
    bounds = [(None, None)] + bounds  # intercept

    def loss(w):
        logits = w[0] + Z @ w[1:]
        # logistic loss, numerically stable
        value = np.mean(np.logaddexp(0, logits) - y * logits) + l2 * np.sum(w[1:] ** 2) / len(y)
        p = 1.0 / (1.0 + np.exp(-logits))
        grad0 = np.mean(p - y)
        grad = Z.T @ (p - y) / len(y) + 2 * l2 * w[1:] / len(y)
        return value, np.concatenate([[grad0], grad])

    result = minimize(loss, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B", bounds=bounds)
    return result.x, mean, std


def rank_with_model(entry: dict, w, mean, std) -> list[str]:
    pool = list(entry["pool"])
    executed = {}
    for record in entry["candidates"]:
        z = (np.array(candidate_features(entry, record)) - mean) / std
        executed[record["j"]] = w[0] + z @ w[1:]
    if not executed:
        return pool
    top = sorted(executed, key=lambda j: -executed[j])
    rest = [j for j in range(len(pool)) if j not in executed]
    return [pool[j] for j in top + rest]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, default=REPO_ROOT / "artifacts" / "apps" / "qrels_train.json")
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    qrels = json.loads(args.qrels.read_text(encoding="utf-8"))
    gold = {q: next(iter(v)) for q, v in qrels.items()}
    _, tune = load_store(args.tune)
    _, dev = load_store(args.dev)

    X, y, _ = build(tune, gold)
    w, mean, std = fit(X, y, args.l2)
    print(f"fitted on TUNE: {len(tune)} statements, {len(y):,} candidates, {int(y.sum())} gold")
    for (name, _), coef in zip(FEATURES, w[1:]):
        print(f"  {name:22s} {coef:+.3f}")

    control = np.array([metrics(rerank(e, Policy("legacy")), gold[e["qid"]]) for e in dev])
    judge = np.array([metrics(rerank(e, JUDGE), gold[e["qid"]]) for e in dev])
    model = np.array([metrics(rank_with_model(e, w, mean, std), gold[e["qid"]]) for e in dev])
    print(f"\nDEV ({len(dev)} statements)      NDCG@10   MRR@10    R@1")
    for name, rows in (("legacy (binary)", control), ("judge-aware (binary)", judge), ("evidence fusion", model)):
        m = rows.mean(axis=0)
        print(f"  {name:24s} {m[0]:.5f}  {m[1]:.5f}  {m[2]:.4f}")
    report = {"weights": dict(zip([n for n, _ in FEATURES], map(float, w[1:]))), "l2": args.l2}
    for name, base in (("vs legacy", control), ("vs judge-aware", judge)):
        d, lo, hi, p = paired_bootstrap(base[:, 0], model[:, 0])
        print(f"  fusion {name:16s} Δ NDCG@10 {d:+.5f} [{lo:+.5f}, {hi:+.5f}]  p={p:.3f}")
        report[name] = {"delta": d, "ci": [lo, hi], "p": p}
    families = sorted({e["family"] for e in dev})
    for fam in families:
        mask = np.array([e["family"] == fam for e in dev])
        print(f"  {fam:18s} n={int(mask.sum()):4d}  legacy {control[mask, 0].mean():.4f}  "
              f"judge {judge[mask, 0].mean():.4f}  fusion {model[mask, 0].mean():.4f}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
