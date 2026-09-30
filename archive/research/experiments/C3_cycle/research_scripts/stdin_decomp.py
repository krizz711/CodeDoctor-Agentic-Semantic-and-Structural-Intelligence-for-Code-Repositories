import json, sys
from collections import Counter, defaultdict
from pathlib import Path
REPO = Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0, str(REPO))
from scripts.replay_outcomes import Policy, load_store, rerank, metrics, candidate_tier, _shape
from app.pipeline.execution_rerank import _normalize_output
gold = {q: next(iter(v)) for q, v in json.loads((REPO/"artifacts/apps/qrels_train.json").read_text(encoding="utf-8")).items()}
pol = Policy("c", numeric_tolerance=1e-6, call_style=True, deepen_to=50)
for split in ["dev","tune"]:
    h, es = load_store(REPO/f"data/outcomes/f2llm17b_all_{split}_deep50.jsonl.gz")
    es = [e for e in es if e["family"]=="competitive_stdin"]; n=len(es)
    loss=defaultdict(float); cnt=Counter(); tot=0; dense=0
    Bdetail=Counter()
    for e in es:
        g=gold[e["qid"]]; rk=rerank(e,pol); nd=metrics(rk,g)[0]; tot+=nd; dense+=metrics(e["pool"],g)[0]
        if nd==1: continue
        recs={r["doc"]:r for r in e["candidates"]}
        if g not in e["pool"]: k="A_recall(gold>50)"
        elif not e["candidates"]: k="D_no_examples"
        else:
            gt=candidate_tier(recs[g],e,pol) if g in recs else "none"
            ver=[d for d in rk[:5] if d in recs and candidate_tier(recs[d],e,pol)=="strong"]
            if gt=="strong": k="B_gold_verified_other_verified_above"; Bdetail[len(ver)]+=1
            elif ver: k="C_wrong_verified_gold_not"
            else: k="D_nothing_verified_gold_unverified"
            if gt!="strong" and g in recs:
                runs=recs[g].get("stdin_runs") or []
                sub=[]
                for run,ex in list(zip(runs,e["stdin_examples"]))[:2]:
                    if run["status"]!="completed": sub.append("timeout")
                    elif run["rc"]!=0: sub.append("crash")
                    else:
                        o,w=run["out"],_normalize_output(ex["expected"])
                        sub.append("ok" if o==w else "case" if o.lower()==w.lower() else "multiset" if sorted(o.split())==sorted(w.split()) else "shape" if _shape(o)==_shape(w) else "mismatch")
                k+=":"+"/".join(sub)
            elif gt!="strong": k+=":gold_not_executed"
        loss[k]+=1-nd; cnt[k]+=1
    print(f"== {split} stdin n={n} dense={dense/n:.5f} pipeline={tot/n:.5f} loss={1-tot/n:.5f}")
    for k in sorted(loss,key=lambda k:-loss[k]): print(f"  {k:62s} {cnt[k]:3d}  {loss[k]/n:.5f}")
    print("  B: #verified in top5 ->", dict(Bdetail))
