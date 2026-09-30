import json, sys
from collections import Counter
from pathlib import Path
REPO=Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0,str(REPO))
from scripts.replay_outcomes import load_store, Policy, candidate_tier, stdin_verdict
from app.pipeline.execution_rerank import _normalize_output
gold={q:next(iter(v)) for q,v in json.loads((REPO/"artifacts/apps/qrels_train.json").read_text(encoding="utf-8")).items()}
h,es=load_store(REPO/"data/outcomes/f2llm17b_all_tune_deep50.jsonl.gz")
print("max j in store", max(r["j"] for e in es for r in e["candidates"]), "header", {k:h.get(k) for k in h if k!="_header"})
def interleaved(s):
    lines=s.rstrip("\n").split("\n")
    return len(lines)>=3 and all(l.strip()=="" for l in lines[1::2]) and all(l.strip() for l in lines[0::2])
tg=[e for e in es if e["legacy_examples"]>0 and e["stdin_examples"] and interleaved(e["stdin_examples"][0]["stdin"])]
pol=Policy("c",numeric_tolerance=1e-6,call_style=True,deepen_to=50)
c=Counter()
for e in tg:
    g=gold[e["qid"]]; pos=e["pool"].index(g) if g in e["pool"] else None
    c["gold_pos_"+("none" if pos is None else "<20" if pos<20 else "<50")]+=1
    rec=[r for r in e["candidates"] if r["doc"]==g]
    if rec:
        r=rec[0]; runs=r.get("stdin_runs") or []
        k=runs[0]["status"]+":"+str(runs[0]["rc"])+":"+str(runs[0]["exc"]) if runs else "norun"
        c["gold_run0 "+k]+=1
        c["gold_tier_"+candidate_tier(r,e,pol)]+=1
print(dict(c))
