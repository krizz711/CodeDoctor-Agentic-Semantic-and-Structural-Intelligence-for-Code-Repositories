"""Read-only: where does the 1.7B call-style(+deep50) pipeline lose NDCG@10 on DEV/TUNE?"""
import json, math, sys
from collections import Counter, defaultdict
from pathlib import Path
REPO = Path(r"C:\Users\Asus\manish\CodeDoctor")
sys.path.insert(0, str(REPO))
from scripts.replay_outcomes import Policy, load_store, rerank, candidate_tier, metrics, _shape  # noqa
from app.pipeline.execution_rerank import _normalize_output  # noqa

gold = {q: next(iter(v)) for q, v in json.loads((REPO / "artifacts/apps/qrels_train.json").read_text(encoding="utf-8")).items()}
pol = Policy("call-style+deep50", numeric_tolerance=1e-6, call_style=True, deepen_to=50)

def why_gold_unverified(entry, rec):
    if rec is None:
        return "gold>exec_window"
    runs = rec.get("stdin_runs") or []
    if entry["legacy_examples"] == 0 and not entry["call_examples"]:
        return "no_examples_parsed"
    if entry["call_examples"] and rec.get("function_style"):
        r = rec.get("call_run")
        if not r or r["status"] != "completed":
            return "call:crash/timeout"
        return "call:mismatch"
    if not runs:
        return "no_runs(style_mismatch?)"
    kinds = []
    for run, ex in list(zip(runs, entry["stdin_examples"]))[:2]:
        if run["status"] != "completed":
            kinds.append("timeout/" + str(run["status"])); continue
        if run["rc"] != 0:
            kinds.append("crash:" + str(run.get("exc"))); continue
        out, want = run["out"], _normalize_output(ex["expected"])
        if out == want: kinds.append("ok"); continue
        if out.lower() == want.lower(): kinds.append("case"); continue
        if sorted(out.split()) == sorted(want.split()): kinds.append("multiset"); continue
        if _shape(out) == _shape(want): kinds.append("same_shape"); continue
        if out.split() == want.split(): kinds.append("ws"); continue
        kinds.append("mismatch")
    bad = [k for k in kinds if k != "ok"]
    return bad[0] if bad else "ok?"

for split in ["dev", "tune"]:
    header, entries = load_store(REPO / f"data/outcomes/f2llm17b_all_{split}_deep50.jsonl.gz")
    loss = defaultdict(float); cnt = Counter(); fam_loss = defaultdict(float); rules_c = Counter()
    total = 0.0
    for e in entries:
        g = gold[e["qid"]]
        rk = rerank(e, pol)
        nd = metrics(rk, g)[0]
        total += nd
        if nd == 1.0:
            continue
        lost = 1.0 - nd
        recs = {r["doc"]: r for r in e["candidates"]}
        if g not in e["pool"]:
            cls = "A_gold_not_in_top50"
        else:
            grec = recs.get(g)
            gt = candidate_tier(grec, e, pol) if grec else "none"
            top = rk[0]
            ttier = candidate_tier(recs[top], e, pol) if top in recs else "none"
            if gt == "strong":
                cls = "B_gold_verified_but_other_verified_above"
            elif ttier == "strong":
                cls = "C_other_verified_gold_not:" + why_gold_unverified(e, grec)
            else:
                cls = "D_nothing_verified:" + why_gold_unverified(e, grec)
            if not cls.startswith("B"):
                r = e["rules"]
                rules_c[(cls.split(":")[0], "any_answer" if r.get("any_answer") else "", "case_ins" if r.get("case_insensitive") else "")] += 1
        loss[cls] += lost; cnt[cls] += 1
        fam_loss[e["family"]] += lost
    n = len(entries)
    print(f"== {split}: n={n} NDCG@10={total/n:.5f}  total loss={1-total/n:.5f}")
    for k in sorted(loss, key=lambda k: -loss[k]):
        print(f"  {k:60s} stmts={cnt[k]:4d}  ndcg_loss={loss[k]/n:.5f}")
    print("  by family:", {k: round(v / n, 5) for k, v in fam_loss.items()})
    print("  rule flags among C/D:", dict(rules_c))
