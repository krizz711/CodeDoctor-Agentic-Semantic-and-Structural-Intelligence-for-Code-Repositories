import json, sys
from pathlib import Path
REPO = Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0, str(REPO))
from scripts.replay_outcomes import Policy, load_store, rerank, metrics, candidate_tier
gold = {q: next(iter(v)) for q, v in json.loads((REPO/"artifacts/apps/qrels_train.json").read_text(encoding="utf-8")).items()}
inv = {v:k for k,v in gold.items()}
pol = Policy("c", numeric_tolerance=1e-6, call_style=True, deepen_to=50)
for split in ["dev","tune"]:
    h, es = load_store(REPO/f"data/outcomes/f2llm17b_all_{split}_deep50.jsonl.gz")
    for e in es:
        if e["family"]!="competitive_stdin": continue
        g=gold[e["qid"]]; rk=rerank(e,pol); nd=metrics(rk,g)[0]
        if nd==1 or g not in e["pool"] or not e["candidates"]: continue
        recs={r["doc"]:r for r in e["candidates"]}
        if g not in recs: continue
        gt=candidate_tier(recs[g],e,pol)
        runs=recs[g].get("stdin_runs") or []
        if gt!="strong" and runs and any(r["rc"]!=0 for r in runs[:2] if r["status"]=="completed"):
            r0=[r for r in runs[:2] if r["status"]=="completed" and r["rc"]!=0][0]
            print(split,"CRASH",e["qid"],"exc",r0["exc"],"read",r0["read"],"len_stdin",len(e["stdin_examples"][0]["stdin"]), "n_ex",len(e["stdin_examples"]),"stdin[:80]",repr(e["stdin_examples"][0]["stdin"][:80]), "detail", str(r0.get("detail"))[:120])
        if gt=="strong":
            above=[d for d in rk[:rk.index(g)] if d in recs and candidate_tier(recs[d],e,pol)=="strong"]
            ex=e["stdin_examples"][0]["expected"]
            print(split,"B",e["qid"],"goldrank_final",rk.index(g)+1,"above",[(d,inv.get(d,"test/none")) for d in above],"n_ex",len(e["stdin_examples"]),"exp",repr(ex[:40]))
