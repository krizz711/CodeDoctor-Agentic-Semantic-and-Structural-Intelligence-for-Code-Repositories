"""Score any verification policy over a recorded outcome store, in seconds.

    python scripts/replay_outcomes.py --store data/outcomes/f2llm06b_all_dev.jsonl.gz

Every policy is a pure function of the recorded evidence (see
``scripts/record_outcomes.py``), so comparing twenty policies costs one
execution pass, not twenty.  The ``legacy`` policy reproduces the shipped rule
(stdin, first two worked examples, whitespace-insensitive exact match, every
verified candidate lifted above the rest in dense order); ``--check-legacy``
asserts that against a live ``apply_execution_rerank`` on a sample, which is
what makes the other policies' numbers trustworthy.

Gold documents come from ``qrels_train.json`` and are used only here, to
*measure*; no policy can see them.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.execution_rerank import _normalize_output, _numeric_match  # noqa: E402
from app.pipeline.statement import values_equal  # noqa: E402


@dataclass(frozen=True)
class Policy:
    name: str
    #: stdin examples used: 2 = the shipped parser's limit
    stdin_examples: int = 2
    #: also use stdin examples from the extended layouts (legacy parser found none)
    extended_stdin: bool = False
    numeric_tolerance: float | None = None
    #: numeric tolerance taken from the statement when it states one
    statement_tolerance: bool = False
    case_rule: bool = False
    #: any-answer statements: accept same output *shape* as a weaker tier
    any_answer_shape: bool = False
    #: any-answer statements: accept same token multiset as a strong match
    any_answer_multiset: bool = False
    call_style: bool = False
    call_min_examples: int = 2
    call_min_distinct: int = 2
    exec_depth: int = 20
    #: weak-tier candidates go below the dense top-1 instead of above it
    weak_below_top1: bool = True
    #: POST-7: when nothing in the first ``exec_depth`` verifies, also use
    #: candidates up to this rank (recorded by scripts/deepen_outcomes.py)
    deepen_to: int = 0
    #: deep verifications go to rank 2 (behind the unverified dense top-1)
    #: instead of rank 1 -- they are further from the dense answer, so the
    #: evidence is weighed against it rather than overriding it
    deep_soft: bool = False
    #: cost gate: candidates more than this below the row's best dense score
    #: are never executed (``ExecutionConfig.max_dense_gap``)
    max_dense_gap: float | None = None
    #: E1 (experiments/C3_cycle): a used stdin example with blank lines that the
    #: raw run did not match gets the recorded collapsed-input run, accepted
    #: only if it read every non-blank character (untracked runs are accepted).
    #: Needs a sidecar merged by :func:`attach_fallback`.
    blank_fallback: bool = False
    #: sensitivity arm: reject fallback matches whose input reading was not tracked
    fallback_untracked_ok: bool = True


# -- comparators -------------------------------------------------------------


def _token_type(token: str) -> str:
    try:
        int(token)
        return "i"
    except ValueError:
        pass
    try:
        float(token)
        return "f"
    except ValueError:
        return "w"


def _shape(text: str) -> list[list[str]]:
    return [[_token_type(t) for t in line.split()] for line in text.splitlines() if line.strip()]


def stdin_verdict(run: dict, expected: str, rules: dict, policy: Policy) -> str:
    """``strong`` / ``weak`` / ``none`` for one run against one example."""
    if run["status"] != "completed" or run["rc"] != 0:
        return "none"
    out = run["out"]
    want = _normalize_output(expected)
    if out == want:
        return "strong"
    tolerance = policy.numeric_tolerance
    if policy.statement_tolerance and rules.get("tolerance"):
        tolerance = max(tolerance or 0.0, float(rules["tolerance"]))
    if tolerance is not None and _numeric_match(out, want, tolerance):
        return "strong"
    if policy.case_rule and rules.get("case_insensitive") and out.lower() == want.lower():
        return "strong"
    if rules.get("any_answer"):
        if policy.any_answer_multiset and sorted(out.split()) == sorted(want.split()) and _shape(out) == _shape(want):
            return "strong"
        if policy.any_answer_shape and _shape(out) == _shape(want) and out:
            return "weak"
    return "none"


def call_verdict(record: dict, entry: dict, policy: Policy) -> str:
    run = record.get("call_run")
    if not run or run["status"] != "completed" or not run.get("results"):
        return "none"
    examples = entry["call_examples"]
    results = run["results"]
    if len(results) < len(examples):
        return "none"
    rules = entry["rules"]
    tolerance = rules.get("tolerance") or 1e-9
    for example, result in zip(examples, results):
        status, value_repr, mutated, _how = (list(result) + [None] * 4)[:4]
        if status != "ok":
            return "none"
        expected = ast.literal_eval(example["expected"])
        matched = False
        for candidate in (value_repr, mutated if value_repr == "None" else None):
            if candidate is None:
                continue
            try:
                value = ast.literal_eval(candidate)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                continue
            if values_equal(value, expected, tolerance=tolerance, case_insensitive=bool(rules.get("case_insensitive"))):
                matched = True
                break
        if not matched:
            return "none"
    identity = all(
        len(ast.literal_eval(e["args"])) + len(ast.literal_eval(e["kwargs"])) == 1
        and repr((list(ast.literal_eval(e["args"])) + [v for _, v in ast.literal_eval(e["kwargs"])])[0])
        == repr(ast.literal_eval(e["expected"]))
        for e in examples
    )
    if (
        len(examples) >= policy.call_min_examples
        and entry.get("distinct_expected", 0) >= policy.call_min_distinct
        and not identity
    ):
        return "strong"
    return "none"


def fallback_verdict(record: dict, entry: dict, index: int, expected: str, policy: Policy) -> bool:
    """E1: does the recorded collapsed-input run for example ``index`` count as a match?"""
    run = (record.get("stdin_runs_fallback") or {}).get(str(index))
    fed = (entry.get("fallback_inputs") or {}).get(str(index))
    if run is None or fed is None:
        return False
    if stdin_verdict(run, expected, entry["rules"], policy) != "strong":
        return False
    consumed = _consumed_exactly(run, fed)
    return policy.fallback_untracked_ok if consumed is None else consumed


def attach_fallback(entries: list[dict], sidecar: Path) -> dict:
    """Merge a ``record_blank_fallback`` sidecar into store entries (in place)."""
    with gzip.open(sidecar, "rt", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        rows = {r["qid"]: r for r in map(json.loads, handle)}
    for entry in entries:
        row = rows.get(entry["qid"])
        if row is None:
            continue
        entry["fallback_inputs"] = row["inputs"]
        for record in entry["candidates"]:
            if record["doc"] in row["runs"]:
                record["stdin_runs_fallback"] = row["runs"][record["doc"]]
    return header


def candidate_tier(record: dict, entry: dict, policy: Policy) -> str:
    runs = record.get("stdin_runs") or []
    use_stdin = entry["legacy_examples"] > 0 or policy.extended_stdin
    if runs and use_stdin:
        limit = policy.stdin_examples
        chosen = list(zip(runs, entry["stdin_examples"]))[:limit]
        verdicts = [stdin_verdict(r, e["expected"], entry["rules"], policy) for r, e in chosen]
        if policy.blank_fallback:
            verdicts = [
                "strong" if v != "strong" and fallback_verdict(record, entry, i, e["expected"], policy) else v
                for i, (v, (_, e)) in enumerate(zip(verdicts, chosen))
            ]
        if verdicts and all(v == "strong" for v in verdicts):
            return "strong"
        stdin_weak = bool(verdicts) and all(v in ("strong", "weak") for v in verdicts)
    else:
        stdin_weak = False
    if policy.call_style and record.get("function_style") and entry["call_examples"]:
        if call_verdict(record, entry, policy) == "strong":
            return "strong"
    return "weak" if stdin_weak else "none"


# -- ranking and metrics -----------------------------------------------------


def _gated(entry: dict, j: int, policy: Policy) -> bool:
    return policy.max_dense_gap is not None and entry["scores"][0] - entry["scores"][j] > policy.max_dense_gap


def rerank(entry: dict, policy: Policy) -> list[str]:
    pool = list(entry["pool"])
    tiers = {}
    for record in entry["candidates"]:
        if record["j"] < policy.exec_depth and not _gated(entry, record["j"], policy):
            tiers[record["j"]] = candidate_tier(record, entry, policy)
    if policy.deepen_to > policy.exec_depth and not any(t == "strong" for t in tiers.values()):
        deep_strong = [
            record["j"]
            for record in entry["candidates"]
            if policy.exec_depth <= record["j"] < policy.deepen_to
            and not _gated(entry, record["j"], policy)
            and candidate_tier(record, entry, policy) == "strong"
        ]
        if deep_strong:
            deep_strong.sort()
            rest = [j for j in range(len(pool)) if j not in set(deep_strong)]
            order = (rest[:1] + deep_strong + rest[1:]) if policy.deep_soft else (deep_strong + rest)
            return [pool[j] for j in order]
    strong = [j for j in range(len(pool)) if tiers.get(j) == "strong"]
    weak = [j for j in range(len(pool)) if tiers.get(j) == "weak"]
    rest = [j for j in range(len(pool)) if j not in set(strong) | set(weak)]
    if policy.weak_below_top1 and weak and not strong and rest and rest[0] == 0:
        order = [0] + weak + rest[1:]
    else:
        order = strong + weak + rest
    return [pool[j] for j in order]


def executions(entry: dict, policy: Policy) -> int:
    """Candidate executions the policy pays for on this statement: the first
    window minus gated ranks, plus the deep window when nothing verified."""

    def used(record: dict) -> bool:
        stdin = bool(record.get("stdin_runs")) and (entry["legacy_examples"] > 0 or policy.extended_stdin)
        call = policy.call_style and record.get("call_run") is not None
        return stdin or call

    first = [r for r in entry["candidates"] if r["j"] < policy.exec_depth and not _gated(entry, r["j"], policy)]
    count = sum(used(r) for r in first)
    if policy.deepen_to > policy.exec_depth and not any(
        candidate_tier(r, entry, policy) == "strong" for r in first
    ):
        count += sum(
            used(r)
            for r in entry["candidates"]
            if policy.exec_depth <= r["j"] < policy.deepen_to and not _gated(entry, r["j"], policy)
        )
    return count


def metrics(ranking: list[str], gold: str) -> tuple[float, float, float, float]:
    try:
        rank = ranking.index(gold) + 1
    except ValueError:
        return 0.0, 0.0, 0.0, 0.0
    ndcg = 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0
    mrr = 1.0 / rank if rank <= 10 else 0.0
    return ndcg, mrr, float(rank == 1), float(rank <= 10)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, resamples: int = 2000, seed: int = 7):
    rng = np.random.default_rng(seed)
    diff = b - a
    n = len(diff)
    means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(resamples)])
    low, high = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(diff.mean()), float(low), float(high), float(min(1.0, p))


POLICIES = [
    Policy("legacy"),
    Policy("legacy+numeric1e-6", numeric_tolerance=1e-6),
    Policy("judge-aware", numeric_tolerance=1e-6, statement_tolerance=True, case_rule=True,
           any_answer_multiset=True),
    Policy("judge-aware+shape", numeric_tolerance=1e-6, statement_tolerance=True, case_rule=True,
           any_answer_multiset=True, any_answer_shape=True),
    Policy("call-style", numeric_tolerance=1e-6, call_style=True),
    Policy("call+extended-stdin", numeric_tolerance=1e-6, call_style=True, extended_stdin=True),
    Policy("all", numeric_tolerance=1e-6, statement_tolerance=True, case_rule=True,
           any_answer_multiset=True, call_style=True, extended_stdin=True),
    Policy("call-style+deep50", numeric_tolerance=1e-6, call_style=True, deepen_to=50),
    Policy("call-style+deep50-soft", numeric_tolerance=1e-6, call_style=True, deepen_to=50, deep_soft=True),
    # the cost gate (candidates more than 0.20 below the row's best are never
    # executed), alone and with deepening: the configuration in candidate_v2.yaml
    Policy("call-style+gate0.2", numeric_tolerance=1e-6, call_style=True, max_dense_gap=0.2),
    Policy("call-style+deep50+gate0.2", numeric_tolerance=1e-6, call_style=True, deepen_to=50,
           max_dense_gap=0.2),
]


def check_legacy(entries: list[dict], header: dict, artifacts: Path, n: int) -> int:
    """Replay must equal a live run of the shipped rule, or no replay is trusted."""
    from app.pipeline.execution_rerank import ExecutionConfig, apply_execution_rerank
    from scripts.dev_eval import load_jsonl_texts

    corpus_ids, corpus_texts = load_jsonl_texts(artifacts / "texts_corpus.jsonl")
    text_of = dict(zip(corpus_ids, corpus_texts))
    query_ids, query_texts = load_jsonl_texts(artifacts / "texts_queries_train.jsonl")
    query_of = dict(zip(query_ids, query_texts))
    depth = int(header["exec_depth"])
    config = ExecutionConfig(enabled=True, max_candidates=depth, timeout_seconds=float(header["timeout"]),
                             verified_bonus=10.0, workers=8)
    policy = Policy("legacy", exec_depth=depth)
    fired = [e for e in entries if e["legacy_examples"] > 0][:n]
    mismatches = 0
    for entry in fired:
        pool = entry["pool"]
        scores, _ = apply_execution_rerank(
            query_of[entry["qid"]], [text_of[d] for d in pool[:depth]], entry["scores"][:depth], config
        )
        live_order = [pool[j] for j in sorted(range(depth), key=lambda j: -scores[j])] + pool[depth:]
        if live_order[:10] != rerank(entry, policy)[:10]:
            mismatches += 1
            print(f"  MISMATCH {entry['qid']}")
    print(f"legacy replay vs live: {len(fired) - mismatches}/{len(fired)} identical top-10")
    return 0 if mismatches == 0 else 1


def _consumed_exactly(run: dict, stdin: str) -> bool | None:
    """Did the run read every non-blank character of its input? ``None`` = untracked."""
    if not run.get("tracked"):
        return None
    read = run.get("read") or 0
    return len(stdin.rstrip()) <= read


def analyze(entries: list[dict], gold: dict[str, str], policies: list[Policy]) -> dict:
    """Mechanism checks: who verifies, what consumption says, what pruning saves."""
    from collections import Counter, defaultdict

    report: dict = {"verification": {}, "consumption": {}, "pruning": {}}
    for policy in policies:
        counts = defaultdict(Counter)
        for entry in entries:
            g = gold[entry["qid"]]
            for record in entry["candidates"]:
                tier = candidate_tier(record, entry, policy)
                role = "gold" if record["doc"] == g else "other"
                counts[entry["family"]][(role, tier)] += 1
        report["verification"][policy.name] = {
            fam: {f"{role}:{tier}": n for (role, tier), n in c.items()} for fam, c in counts.items()
        }

    buckets = Counter()
    for entry in entries:
        g = gold[entry["qid"]]
        for record in entry["candidates"]:
            role = "gold" if record["doc"] == g else "other"
            for run, example in zip(record.get("stdin_runs") or [], entry["stdin_examples"]):
                if run["status"] != "completed" or run["rc"] != 0:
                    kind = "crash-" + (run.get("exc") or "exit")
                    consumed = _consumed_exactly(run, example["stdin"])
                    buckets[(role, kind, consumed)] += 1
                    continue
                matched = run["out"] == _normalize_output(example["expected"])
                consumed = _consumed_exactly(run, example["stdin"])
                buckets[(role, "match" if matched else "mismatch", consumed)] += 1
    report["consumption"] = {f"{r}|{k}|{c}": n for (r, k, c), n in buckets.items()}

    pruned = gold_pruned = total_stdin_runs = 0
    for entry in entries:
        if not entry["stdin_examples"] or entry["call_examples"]:
            continue
        g = gold[entry["qid"]]
        for record in entry["candidates"]:
            runs = len(record.get("stdin_runs") or [])
            total_stdin_runs += runs
            if record.get("function_style"):
                pruned += runs
                gold_pruned += int(record["doc"] == g)
    report["pruning"] = {"stdin_runs": total_stdin_runs, "function_style_runs_skippable": pruned,
                         "golds_among_skipped": gold_pruned}
    return report


def load_store(path: Path) -> tuple[dict, list[dict]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        entries = [json.loads(line) for line in handle]
    return header, entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts" / "apps")
    parser.add_argument("--qrels", type=Path, default=None,
                        help="qrels JSON (default: <artifacts>/qrels_train.json)")
    parser.add_argument("--control", default="legacy")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--check-legacy",
        type=int,
        default=0,
        help="re-run the shipped apply_execution_rerank live on N queries and assert the "
        "replayed legacy ranking is identical",
    )
    args = parser.parse_args(argv)

    header, entries = load_store(args.store)
    if args.check_legacy:
        return check_legacy(entries, header, args.artifacts, args.check_legacy)
    qrels_path = args.qrels or args.artifacts / "qrels_train.json"
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    gold = {q: next(iter(v)) for q, v in qrels.items()}
    print(f"store: {args.store.name}  queries={len(entries)}  model={header['model']}  "
          f"split={header['split']}  family={header['family']}  exec_depth={header['exec_depth']}")

    dense_rows = np.array([metrics(e["pool"], gold[e["qid"]]) for e in entries])
    results = {"dense": dense_rows}
    runs = {"dense": 0}
    for policy in POLICIES:
        results[policy.name] = np.array([metrics(rerank(e, policy), gold[e["qid"]]) for e in entries])
        runs[policy.name] = sum(executions(e, policy) for e in entries)
    # Upper bounds, not policies: the gold moved to rank 1 whenever it is in the
    # top ``depth`` of the dense pool -- what perfect post-processing could reach.
    for depth in (20, 50):
        name = f"ceiling@{depth}"
        results[name] = np.array([
            metrics([gold[e["qid"]]] + [d for d in e["pool"] if d != gold[e["qid"]]]
                    if gold[e["qid"]] in e["pool"][:depth] else e["pool"], gold[e["qid"]])
            for e in entries
        ])
        runs[name] = 0

    families = sorted({e["family"] for e in entries})
    control = results[args.control]
    report = {"header": header, "policies": {}}
    print(f"\n{'policy':24s} {'NDCG@10':>8s} {'MRR@10':>8s} {'R@1':>7s} {'R@10':>7s}   "
          f"{'runs':>7s}   Δ NDCG vs {args.control} [95% CI]  p")
    for name, rows in results.items():
        mean = rows.mean(axis=0)
        line = f"{name:26s} {mean[0]:8.5f} {mean[1]:8.5f} {mean[2]:7.4f} {mean[3]:7.4f} {runs[name]:7d}"
        entry = {"ndcg10": mean[0], "mrr10": mean[1], "r1": mean[2], "r10": mean[3], "executions": runs[name]}
        if name != args.control:
            d, lo, hi, p = paired_bootstrap(control[:, 0], rows[:, 0])
            line += f"   {d:+.5f} [{lo:+.5f}, {hi:+.5f}]  {p:.3f}"
            entry.update({"delta": d, "ci": [lo, hi], "p": p})
        per_family = {}
        for fam in families:
            mask = np.array([e["family"] == fam for e in entries])
            per_family[fam] = {"n": int(mask.sum()), "ndcg10": float(rows[mask, 0].mean())}
        entry["per_family"] = per_family
        report["policies"][name] = entry
        print(line)

    print("\nper family NDCG@10:")
    print(f"{'policy':24s}" + "".join(f"{f[:14]:>16s}" for f in families))
    for name in results:
        print(f"{name:24s}" + "".join(
            f"{report['policies'][name]['per_family'][f]['ndcg10']:16.5f}" for f in families))
    print(" " * 24 + "".join(f"{'n=' + str(report['policies']['dense']['per_family'][f]['n']):>16s}" for f in families))

    mechanism = analyze(entries, gold, POLICIES)
    report["mechanism"] = mechanism
    print("\nverification by family (gold:strong / gold total, other:strong / other total):")
    for name in ("legacy", "judge-aware", "call-style", "all"):
        line = f"  {name:22s}"
        for fam, c in sorted(mechanism["verification"][name].items()):
            gold_total = sum(v for k, v in c.items() if k.startswith("gold:"))
            other_total = sum(v for k, v in c.items() if k.startswith("other:"))
            line += (f"  {fam[:10]}: {c.get('gold:strong', 0)}/{gold_total}"
                     f" | {c.get('other:strong', 0)}/{other_total}")
        print(line)
    print("\nconsumption (role|outcome|consumed_exactly -> runs):")
    for key, n in sorted(mechanism["consumption"].items(), key=lambda kv: -kv[1])[:14]:
        print(f"  {key:40s} {n}")
    print(f"\npruning: {mechanism['pruning']}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
