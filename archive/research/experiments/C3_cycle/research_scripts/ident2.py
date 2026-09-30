"""TUNE-only: rare-identifier match (query mentions a def-name that few corpus docs define) as a bonus, no-example statements only."""
import json, sys, re
import numpy as np
from collections import Counter
from pathlib import Path
REPO = Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0, str(REPO))
from scripts.replay_outcomes import Policy, load_store, rerank, metrics, paired_bootstrap
from scripts.dev_eval import load_jsonl_texts
gold = {q: next(iter(v)) for q, v in json.loads((REPO/"artifacts/apps/qrels_train.json").read_text(encoding="utf-8")).items()}
cids, ctexts = load_jsonl_texts(REPO/"artifacts/apps/texts_corpus.jsonl"); ctext=dict(zip(cids,ctexts))
qids,qt=load_jsonl_texts(REPO/"artifacts/apps/texts_queries_train.jsonl"); Q=dict(zip(qids,qt))
DEF=re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", re.M)
dn={d:{n for n in DEF.findall(t) if not n.startswith("_")} for d,t in ctext.items()}
df=Counter(n for s in dn.values() for n in s)
MENT=re.compile(r"`([A-Za-z_]\w*)`|\b([A-Za-z_]\w*)\s*\(")
pol = Policy("c", numeric_tolerance=1e-6, call_style=True, deepen_to=50)
h, es = load_store(REPO/"data/outcomes/f2llm17b_all_tune_deep50.jsonl.gz")
for maxdf in [3,10]:
  for bonus in [0.02,0.05,1.0]:
    a=[];b=[]
    for e in es:
        g=gold[e["qid"]]; base=rerank(e,pol); a.append(metrics(base,g)[0])
        if e["candidates"]: b.append(a[-1]); continue
        names={x or y for x,y in MENT.findall(Q[e["qid"]])}
        names={n for n in names if len(n)>=4 and 0<df.get(n,0)<=maxdf}
        if not names: b.append(a[-1]); continue
        sc={d:s+(bonus if dn[d]&names else 0) for d,s in zip(e["pool"],e["scores"])}
        new=sorted(e["pool"],key=lambda d:-sc[d]); b.append(metrics(new,g)[0])
    a=np.array(a);b=np.array(b); m,lo,hi,p=paired_bootstrap(a,b)
    print(f"maxdf={maxdf} bonus={bonus} d={m:+.5f} [{lo:+.5f},{hi:+.5f}] up={(b>a).sum()} down={(b<a).sum()}")
