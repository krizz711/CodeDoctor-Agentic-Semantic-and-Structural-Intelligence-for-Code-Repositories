import json, sys
import numpy as np
from pathlib import Path
REPO = Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0, str(REPO))
from scripts.replay_outcomes import Policy, load_store, rerank, metrics, candidate_tier
from scripts.dev_eval import load_jsonl_texts
art=REPO/"artifacts/apps"
qids,qt=load_jsonl_texts(art/"texts_queries_train.jsonl"); qi={q:i for i,q in enumerate(qids)}
Q=np.load(art/"emb_f2llm17b_qtrain.npy").astype(np.float32)
gold = {q: next(iter(v)) for q, v in json.loads((art/"qrels_train.json").read_text(encoding="utf-8")).items()}
inv={v:k for k,v in gold.items()}
pol = Policy("c", numeric_tolerance=1e-6, call_style=True, deepen_to=50)
cos=[]; nontrain=0; tot=0
for split in ["dev","tune"]:
    h, es = load_store(REPO/f"data/outcomes/f2llm17b_all_{split}_deep50.jsonl.gz")
    for e in es:
        g=gold[e["qid"]]; rk=rerank(e,pol)
        if metrics(rk,g)[0]==1 or g not in rk: continue
        recs={r["doc"]:r for r in e["candidates"]}
        if g not in recs or candidate_tier(recs[g],e,pol)!="strong": continue
        top=rk[0]; tot+=1
        if top in inv: cos.append(float(Q[qi[e["qid"]]]@Q[qi[inv[top]]]))
        else: nontrain+=1
cos=np.array(cos)
print("B cases",tot,"top doc owned by another train query:",len(cos),"(test/unowned:",nontrain,")")
print("query-query cosine quantiles", np.round(np.quantile(cos,[0,.25,.5,.75,1]),3), "share >=0.95:", round((cos>=0.95).mean(),3))
