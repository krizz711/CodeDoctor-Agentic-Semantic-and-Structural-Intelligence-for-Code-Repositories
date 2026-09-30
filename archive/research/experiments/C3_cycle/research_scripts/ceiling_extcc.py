"""Read-only ceiling check for a within-tie consensus rule on real near-misses
(CodeContests EXT-dev, 108 statements; recorded stores; no lockbox rows)."""
import json, math, sys
from pathlib import Path
REPO = Path(r"C:\Users\Asus\manish\CodeDoctor")
sys.path.insert(0, str(REPO))
from scripts.replay_outcomes import Policy, candidate_tier, load_store, rerank  # noqa: E402

qrels = json.loads((REPO / "artifacts/ext_cc/qrels.json").read_text(encoding="utf-8"))
dev_ids = set(json.loads((REPO / "artifacts/ext_cc/splits.json").read_text(encoding="utf-8"))["ext_dev"])
pol = Policy("callstyle", call_style=True)
def ndcg(rank, gold):
    return 1 / math.log2(rank.index(gold) + 2) if gold in rank[:10] else 0.0
out = {}
for model in ["f2llm17b", "f2llm06b"]:
    _, entries = load_store(REPO / f"data/outcomes/{model}_extcc_dev.jsonl.gz")
    entries = [e for e in entries if e["qid"] in dev_ids]
    n = len(entries); base = ceil = 0.0
    tie_cases = gold_first = wrong_first_verified = gold_unverified_wrong_first = 0
    ties_sizes = []
    for e in entries:
        gold = next(iter(qrels[e["qid"]]))
        r = rerank(e, pol)
        base += ndcg(r, gold)
        tiers = {rec["doc"]: candidate_tier(rec, e, pol) for rec in e["candidates"] if rec["j"] < pol.exec_depth}
        strong = [d for d in r if tiers.get(d) == "strong"]
        if r[0] == gold:
            gold_first += 1; ceil += 1.0; continue
        if tiers.get(r[0]) == "strong" and r[0].startswith("ccx"):
            wrong_first_verified += 1
        if gold in strong and r[0] != gold:
            tie_cases += 1; ties_sizes.append(len(strong)); ceil += 1.0
        else:
            ceil += ndcg(r, gold)
            if r[0].startswith("ccx") and tiers.get(gold) != "strong":
                gold_unverified_wrong_first += 1
    out[model] = {"n": n, "ndcg_callstyle": round(base / n, 4),
                  "gold_rank1": gold_first,
                  "rank1_is_verified_incorrect_submission": wrong_first_verified,
                  "reachable_by_tie_break (gold verified, not rank1)": tie_cases,
                  "verified_set_size_in_those": ties_sizes,
                  "ceiling_ndcg_if_all_ties_resolved": round(ceil / n, 4),
                  "max_gain": round((ceil - base) / n, 4),
                  "incorrect_first_and_gold_unverified": gold_unverified_wrong_first}
print(json.dumps(out, indent=1))
