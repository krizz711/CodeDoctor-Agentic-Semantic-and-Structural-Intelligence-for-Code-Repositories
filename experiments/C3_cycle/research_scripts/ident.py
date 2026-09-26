import json, sys, re
from collections import Counter, defaultdict
from pathlib import Path
REPO = Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0, str(REPO))
from scripts.replay_outcomes import Policy, load_store, rerank, metrics, candidate_tier
from scripts.dev_eval import load_jsonl_texts
gold = {q: next(iter(v)) for q, v in json.loads((REPO/"artifacts/apps/qrels_train.json").read_text(encoding="utf-8")).items()}
cids, ctexts = load_jsonl_texts(REPO/"artifacts/apps/texts_corpus.jsonl"); ctext = dict(zip(cids, ctexts))
qids, qtexts = load_jsonl_texts(REPO/"artifacts/apps/texts_queries_train.jsonl"); qtext = dict(zip(qids, qtexts))
pol = Policy("c", numeric_tolerance=1e-6, call_style=True, deepen_to=50)
DEF = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", re.M)
IDENT = re.compile(r"[A-Za-z_]\w{2,}")
def defs(t): return {n for n in DEF.findall(t) if not n.startswith("__")}
for split in ["dev","tune"]:
    h, es = load_store(REPO/f"data/outcomes/f2llm17b_all_{split}_deep50.jsonl.gz")
    fam = defaultdict(lambda: Counter())
    stats = Counter()
    for e in es:
        g = gold[e["qid"]]; rk = rerank(e, pol); nd = metrics(rk, g)[0]
        f = e["family"]; fam[f]["n"] += 1; fam[f]["ndcg"] += nd
        noex = len(e["candidates"]) == 0
        fam[f]["noex"] += noex; fam[f]["noex_loss"] += (1-nd)*noex; fam[f]["loss"] += 1-nd
        if g not in e["pool"]: fam[f]["A_loss"] += 1-nd
        if not noex: continue
        # identifier-match analysis on no-example statements
        qt = qtext[e["qid"]]; qid_tokens = set(IDENT.findall(qt))
        top10 = rk[:10]
        match = [bool(defs(ctext[d]) & qid_tokens) for d in e["pool"]]
        gm = bool(defs(ctext[g]) & qid_tokens) if g in ctext else False
        stats["noex"] += 1; stats["gold_defname_in_query"] += gm
        stats["top1_defname_in_query"] += match[0]
        if nd < 1 and g in e["pool"]:
            stats["miss"] += 1; stats["miss_gold_match"] += gm
            stats["miss_top1_match"] += match[0]
            # simulated: stable-partition matching docs first within top-50
            order = [d for d, m in zip(e["pool"], match) if m] + [d for d, m in zip(e["pool"], match) if not m]
            stats["miss_fixed_by_partition"] += (order[0] == g)
        if nd == 1:
            order = [d for d, m in zip(e["pool"], match) if m] + [d for d, m in zip(e["pool"], match) if not m]
            stats["hit_broken_by_partition"] += (order[0] != g)
    print("==", split)
    for f, c in fam.items():
        n = c["n"]; print(f"  {f:18s} n={n:4d} ndcg={c['ndcg']/n:.4f} noex={c['noex']:4d} loss(total/2500)={c['loss']/2500:.5f} noex_loss={c['noex_loss']/2500:.5f} recallA_loss={c['A_loss']/2500:.5f}")
    print("  identifier:", dict(stats))
