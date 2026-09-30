import json, sys
from collections import Counter
from pathlib import Path
REPO = Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0, str(REPO))
from scripts.replay_outcomes import Policy, load_store, rerank, metrics, candidate_tier
q=json.loads((REPO/"artifacts/ext_cc/qrels.json").read_text(encoding="utf-8"))
gold={k:next(iter(v)) for k,v in q.items()}
pol = Policy("c", numeric_tolerance=1e-6, call_style=True, deepen_to=50)
def interleaved(s):
    lines = s.rstrip("\n").split("\n")
    return len(lines) >= 3 and all(l.strip()=="" for l in lines[1::2]) and all(l.strip() for l in lines[0::2])
h, es = load_store(REPO/"data/outcomes/f2llm17b_extcc_dev_deep50.jsonl.gz")
c=Counter(); exc=Counter()
for e in es:
    c["n"]+=1
    if not e["stdin_examples"]: c["noex"]+=1; continue
    s=e["stdin_examples"][0]["stdin"]; c["interleaved"]+=interleaved(s); c["any_blank"]+=("\n\n" in s.strip("\n"))
    g=gold.get(e["qid"]); recs={r["doc"]:r for r in e["candidates"]}
    if g in recs:
        runs=recs[g].get("stdin_runs") or []
        if runs and runs[0]["status"]=="completed" and runs[0]["rc"]!=0: c["gold_crash"]+=1; exc[runs[0]["exc"]]+=1
        if candidate_tier(recs[g],e,pol)=="strong": c["gold_verified"]+=1
        c["gold_executed"]+=1
print(dict(c), dict(exc))
