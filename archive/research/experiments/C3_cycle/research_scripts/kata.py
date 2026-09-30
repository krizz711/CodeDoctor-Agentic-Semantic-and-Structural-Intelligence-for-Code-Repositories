import json, sys, re, random
from collections import Counter
from pathlib import Path
REPO = Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0, str(REPO))
from scripts.replay_outcomes import Policy, load_store, rerank, metrics
from scripts.dev_eval import load_jsonl_texts
gold = {q: next(iter(v)) for q, v in json.loads((REPO/"artifacts/apps/qrels_train.json").read_text(encoding="utf-8")).items()}
qids,qt=load_jsonl_texts(REPO/"artifacts/apps/texts_queries_train.jsonl"); Q=dict(zip(qids,qt))
pol = Policy("c", numeric_tolerance=1e-6, call_style=True, deepen_to=50)
PAT={"call_arrow":re.compile(r"\w+\([^()\n]{0,80}\)\s*(==|=>|-->|->|returns?|should return|#\s*=>|#\s*returns?|\bis\b)"),
     "doctest":re.compile(r">>>"), "example_word":re.compile(r"(?i)\bexamples?\b"), "input_output":re.compile(r"(?i)\binput\b.*\n?.*\boutput\b")}
h, es = load_store(REPO/"data/outcomes/f2llm17b_all_tune_deep50.jsonl.gz")
c=Counter(); lossc=Counter(); samples=[]
for e in es:
    if e["candidates"]: continue
    t=Q[e["qid"]]; nd=metrics(rerank(e,pol),gold[e["qid"]])[0]
    c["noex"]+=1; lossc["noex"]+=1-nd
    hits=[k for k,p in PAT.items() if p.search(t)]
    for k in hits: c[k]+=1; lossc[k]+=1-nd
    if "call_arrow" in hits and nd<1 and len(samples)<4:
        m=PAT["call_arrow"].search(t); samples.append((e["qid"], e["family"], t[max(0,m.start()-60):m.end()+60].replace("\n"," | ")))
print(dict(c)); print({k:round(v/2500,5) for k,v in lossc.items()})
for s in samples: print(s)
