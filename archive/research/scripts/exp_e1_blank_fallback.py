"""E1 (experiments/C3_cycle/PREREG.md): score the blank-line fallback.

    # sidecars first (scripts/record_blank_fallback.py): own inputs, --placebo, --retry
    python scripts/exp_e1_blank_fallback.py --split dev --live --out experiments/C3_cycle/E1_result_dev.json

The replay is the primary number.  ``--live`` adds the sanity controls that
execute code: S3 (re-executed raw runs reproduce their recorded verdicts) and S4
(the live verifier with the switch on gives the replayed top-10), plus the added
wall-clock per affected query.  The pre-registered decision rule is applied
mechanically; every number it rests on is written to the result.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.c3_stats import compare  # noqa: E402
from scripts.record_blank_fallback import affected  # noqa: E402
from scripts.replay_outcomes import (  # noqa: E402
    Policy,
    _consumed_exactly,
    attach_fallback,
    candidate_tier,
    load_store,
    metrics,
    rerank,
    stdin_verdict,
)

OUT = REPO_ROOT / "data" / "outcomes"
APPS, EXT = REPO_ROOT / "artifacts" / "apps", REPO_ROOT / "artifacts" / "ext_cc"
SPLITS = {
    "dev": (OUT / "f2llm17b_all_dev_deep50.jsonl.gz", APPS / "qrels_train.json",
            APPS / "texts_queries_train.jsonl", APPS / "texts_corpus.jsonl"),
    "tune": (OUT / "f2llm17b_all_tune_deep50.jsonl.gz", APPS / "qrels_train.json",
             APPS / "texts_queries_train.jsonl", APPS / "texts_corpus.jsonl"),
    "extcc": (OUT / "f2llm17b_extcc_dev_deep50.jsonl.gz", EXT / "qrels.json",
              EXT / "texts_queries.jsonl", EXT / "texts_corpus.jsonl"),
}
V2 = Policy("call-style+deep50", numeric_tolerance=1e-6, call_style=True, deepen_to=50)
FB = replace(V2, name="call-style+deep50+blankfb", blank_fallback=True)
FROZEN = Policy("legacy+numeric1e-6", numeric_tolerance=1e-6)
#: sanity control S1: the unmodified replay must reproduce these exactly
KNOWN = {"dev": (0.95327, 0.95993), "tune": (0.95849, 0.96362)}
STDIN = "competitive_stdin"
SEED = 20260926
MEANINGFUL = re.compile(r"(blank|empty)\s+line", re.I)


def texts(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as handle:
        return {r["id"]: r["text"] for r in map(json.loads, handle)}


def ranks(entries, policy, gold):
    orders = [rerank(e, policy) for e in entries]
    rows = np.array([metrics(o, gold[e["qid"]])[:2] for o, e in zip(orders, entries)])
    return rows[:, 0], rows[:, 1], orders


def fallback_only(record, entry, policy=FB) -> bool:
    return (candidate_tier(record, entry, policy) == "strong"
            and candidate_tier(record, entry, replace(policy, blank_fallback=False)) != "strong")


def mechanism(entries, gold, policy=FB) -> dict:
    gold_rec, nongold, untracked, statements = 0, 0, 0, []
    for entry in entries:
        if "fallback_inputs" not in entry:
            continue
        g = gold[entry["qid"]]
        for record in entry["candidates"]:
            if fallback_only(record, entry, policy):
                gold_rec += record["doc"] == g
                nongold += record["doc"] != g
                statements.append(entry["qid"])
                for i, run in (record.get("stdin_runs_fallback") or {}).items():
                    if _consumed_exactly(run, entry["fallback_inputs"][i]) is None:
                        untracked += 1
    return {"gold_recoveries": gold_rec, "nongold_fallback_verifications": nongold,
            "untracked_fallback_accepted": untracked, "statements_with_fallback_verification": sorted(set(statements))}


def paid_fallback_runs(entry, policy=FB) -> int:
    """Fallback runs a live pass executes: raw not strong on a blank example, in the executed window."""
    if "fallback_inputs" not in entry:
        return 0
    first = [r for r in entry["candidates"] if r["j"] < policy.exec_depth]
    window = first
    if not any(candidate_tier(r, entry, policy) == "strong" for r in first):
        window = [r for r in entry["candidates"] if r["j"] < policy.deepen_to]
    count = 0
    for record in window:
        for i, example in enumerate(entry["stdin_examples"][: policy.stdin_examples]):
            if str(i) not in entry["fallback_inputs"]:
                continue
            runs = record.get("stdin_runs") or []
            if i < len(runs) and stdin_verdict(runs[i], example["expected"], entry["rules"], policy) != "strong":
                count += 1
    return count


def displaced_rank1(entries, gold, before, after) -> list[str]:
    """Rule 6: statements where a fallback-only non-gold verification pushes a rank-1 gold down."""
    hits = []
    for entry, b, a in zip(entries, before, after):
        g = gold[entry["qid"]]
        if b and b[0] == g and a[0] != g:
            by_doc = {r["doc"]: r for r in entry["candidates"]}
            if a[0] in by_doc and fallback_only(by_doc[a[0]], entry):
                hits.append(entry["qid"])
    return hits


def live_config(on: bool):
    from app.pipeline.execution_rerank import ExecutionConfig

    # candidate_v2.yaml as the live scored path builds it (timeout_mode cpu falls back to wall on Windows)
    return ExecutionConfig(enabled=True, max_candidates=20, timeout_seconds=2.0, numeric_tolerance=1e-6,
                           call_style=True, deepen_to=50, skip_interactive=True, timeout_mode="cpu",
                           workers=6, verified_bonus=10.0, blank_line_fallback=on)


def s4_live(treated, base_by_qid, queries, corpus, gold) -> dict:
    """S4: live top-10 with the switch on equals the replay; also off-vs-off and added wall-clock."""
    from app.pipeline.execution_rerank import apply_execution_rerank, spawn_failures

    rows = []
    for entry in treated:
        if "fallback_inputs" not in entry:
            continue
        docs, scores = entry["pool"], entry["scores"]
        out = {"qid": entry["qid"]}
        for on in (False, True):
            t = time.perf_counter()
            new, _ = apply_execution_rerank(queries[entry["qid"]], [corpus[d] for d in docs], scores, live_config(on))
            out[f"secs_{'on' if on else 'off'}"] = round(time.perf_counter() - t, 3)
            live = [docs[j] for j in sorted(range(len(docs)), key=lambda j: (-new[j], j))]
            replayed = rerank(entry if on else base_by_qid[entry["qid"]], FB if on else V2)
            out[f"match_{'on' if on else 'off'}"] = live[:10] == replayed[:10]
            out[f"live_ndcg_{'on' if on else 'off'}"] = metrics(live, gold[entry["qid"]])[0]
        rows.append(out)
    added = np.array([r["secs_on"] - r["secs_off"] for r in rows])
    return {"n": len(rows), "match_on": sum(r["match_on"] for r in rows), "match_off": sum(r["match_off"] for r in rows),
            "mismatch_on": [r["qid"] for r in rows if not r["match_on"]],
            "mismatch_off": [r["qid"] for r in rows if not r["match_off"]],
            "added_seconds_p50": round(float(np.percentile(added, 50)), 3),
            "added_seconds_p95": round(float(np.percentile(added, 95)), 3),
            "spawn_failures": spawn_failures(), "rows": rows}


def s3_rerun(base_entries, corpus) -> dict:
    """S3: 200 recorded raw runs on affected examples re-executed; >= 198 verdicts must match."""
    from app.pipeline.execution_rerank import ExecutionConfig, execute_candidate
    from scripts.record_outcomes import _stdin_record

    pool, timeouts = [], []
    for entry in base_entries:
        for i in affected(entry):
            example = entry["stdin_examples"][i]
            for record in entry["candidates"]:
                runs = record.get("stdin_runs") or []
                if i < len(runs):
                    item = (entry, record, i, example)
                    (timeouts if runs[i]["status"] == "timeout" else pool).append(item)
    rng = random.Random(SEED)
    sample = timeouts + rng.sample(pool, max(0, 200 - len(timeouts)))
    config = ExecutionConfig(enabled=True, timeout_seconds=2.0, call_style=True, extended_stdin=True, max_examples=5)
    same = 0
    for entry, record, i, example in sample:
        new = _stdin_record(execute_candidate(corpus[record["doc"]], example["stdin"], config, track_input=True))
        old_v = stdin_verdict(record["stdin_runs"][i], example["expected"], entry["rules"], V2)
        new_v = stdin_verdict(new, example["expected"], entry["rules"], V2)
        same += old_v == new_v
    return {"n": len(sample), "recorded_timeouts_included": len(timeouts), "verdicts_identical": same,
            "pass": same >= len(sample) - 2}


def evaluate(split: str, sidecar: Path, placebo: Path | None, retry: Path | None, live: bool) -> dict:
    store, qrels_path, query_path, corpus_path = SPLITS[split]
    header, base = load_store(store)
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    gold = {q: next(iter(v)) for q, v in qrels.items()}
    base = [e for e in base if e["qid"] in gold]
    stdin = np.array([e["family"] == STDIN for e in base])

    b_nd, b_mrr, b_orders = ranks(base, V2, gold)
    result: dict = {"split": split, "store": header}
    if split in KNOWN:
        s1 = (round(float(b_nd.mean()), 5), round(float(b_nd[stdin].mean()), 5))
        result.update(S1_reproduced=list(s1), S1_expected=list(KNOWN[split]), S1_pass=s1 == KNOWN[split])

    treated = copy.deepcopy(base)
    result["sidecar"] = attach_fallback(treated, sidecar)
    t_nd, t_mrr, t_orders = ranks(treated, FB, gold)
    result["primary_stdin_ndcg10"] = compare(b_nd[stdin], t_nd[stdin])
    result["stdin_mrr10"] = compare(b_mrr[stdin], t_mrr[stdin])
    result["all_family_ndcg10"] = compare(b_nd, t_nd)
    result["all_family_mrr10"] = compare(b_mrr, t_mrr)
    diff = t_nd - b_nd
    result["changed_statements"] = {e["qid"]: round(float(d), 4) for e, d in zip(base, diff) if d != 0}
    result["mechanism"] = mechanism(treated, gold)
    result["rule6_rank1_gold_displaced"] = displaced_rank1(treated, gold, b_orders, t_orders)
    affected_entries = [e for e in treated if "fallback_inputs" in e]
    runs = [paid_fallback_runs(e) for e in affected_entries]
    result["cost"] = {"affected_statements": len(affected_entries), "fallback_runs_total": int(sum(runs)),
                      "fallback_runs_per_affected_mean": round(float(np.mean(runs)), 2) if runs else 0.0}

    # sensitivity: untracked fallback runs rejected
    s_nd, _, _ = ranks(treated, replace(FB, fallback_untracked_ok=False), gold)
    result["sensitivity_untracked_rejected_stdin"] = compare(b_nd[stdin], s_nd[stdin])
    # descriptive: the frozen (submitted) rule with the same fallback
    f_nd, _, _ = ranks(base, FROZEN, gold)
    ft_nd, _, _ = ranks(treated, replace(FROZEN, blank_fallback=True), gold)
    result["descriptive_frozen_rule_stdin"] = compare(f_nd[stdin], ft_nd[stdin])
    # meaningful blank lines (statements that talk about blank/empty lines)
    queries = texts(query_path)
    idx = {e["qid"]: k for k, e in enumerate(base)}
    result["meaningful_blank_line_statements"] = [
        {"qid": e["qid"], "baseline": round(float(b_nd[idx[e["qid"]]]), 4), "treated": round(float(t_nd[idx[e["qid"]]]), 4)}
        for e in affected_entries if MEANINGFUL.search(queries.get(e["qid"], ""))]

    for name, path in (("negative_control", placebo), ("retry_control", retry)):
        if path is None:
            continue
        control = copy.deepcopy(base)
        result[f"{name}_sidecar"] = attach_fallback(control, path)
        c_nd, _, c_orders = ranks(control, FB, gold)
        result[f"{name}_stdin"] = compare(b_nd[stdin], c_nd[stdin])
        result[f"{name}_mechanism"] = mechanism(control, gold)
        result[f"{name}_moved"] = {e["qid"]: round(float(c - b), 4) for e, b, c in zip(base, b_nd, c_nd) if c != b}

    if live:
        corpus = texts(corpus_path)
        result["S3"] = s3_rerun(base, corpus)
        result["S4"] = s4_live(treated, {e["qid"]: e for e in base}, queries, corpus, gold)

    if split not in KNOWN:
        return result
    p = result["primary_stdin_ndcg10"]
    rules = {
        "1_delta_ge_0.003": p["delta"] >= 0.003,
        "2_ci_excludes_0": p["ci95"][0] > 0,
        "3_sign_flip_p_lt_0.05": p["sign_flip_p"] < 0.05,
        "4_at_least_6_changed": p["up"] + p["down"] >= 6,
        "5_all_family_delta_ge_0": result["all_family_ndcg10"]["delta"] >= 0,
        "6_at_most_1_rank1_displacement": len(result["rule6_rank1_gold_displaced"]) <= 1,
    }
    result["decision_rules"] = rules
    neg = result.get("negative_control_stdin")
    result["negative_control_fails_as_required"] = (
        None if neg is None else result["negative_control_mechanism"]["gold_recoveries"] <= 1 and neg["delta"] <= 0)
    sanity = {"S1": result["S1_pass"]}
    if live:
        sanity["S3"] = result["S3"]["pass"]
        s4 = result["S4"]
        sanity["S4"] = s4["match_on"] >= s4["n"] - 1 and s4["spawn_failures"] == 0
        # a tolerated mismatch on a changed statement: the decision must also hold scored live
        crit = [q for q in s4["mismatch_on"] if q in result["changed_statements"]]
        if crit:
            live_nd = t_nd.copy()
            for row in s4["rows"]:
                if row["qid"] in crit:
                    live_nd[idx[row["qid"]]] = row["live_ndcg_on"]
            result["primary_with_live_scores_for_mismatches"] = compare(b_nd[stdin], live_nd[stdin])
            lp = result["primary_with_live_scores_for_mismatches"]
            rules["S4_live_rescore_holds"] = (lp["delta"] >= 0.003 and lp["ci95"][0] > 0 and lp["sign_flip_p"] < 0.05)
    result["sanity"] = sanity
    result["adopt"] = bool(all(sanity.values()) and all(rules.values())
                           and result["negative_control_fails_as_required"] is not False)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(SPLITS), required=True)
    parser.add_argument("--live", action="store_true", help="also run S3 and S4 (executes code)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    store = SPLITS[args.split][0]
    side = lambda suffix: store.with_name(store.name.replace(".jsonl.gz", f"{suffix}.jsonl.gz"))  # noqa: E731
    placebo, retry = side("_blankfb_placebo"), side("_blankfb_retry")
    result = evaluate(args.split, side("_blankfb"), placebo if placebo.exists() else None,
                      retry if retry.exists() else None, args.live)
    text = json.dumps(result, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
