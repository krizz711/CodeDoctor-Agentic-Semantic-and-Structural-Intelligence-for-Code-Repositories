import json, sys
from pathlib import Path
REPO=Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0,str(REPO))
from scripts.replay_outcomes import load_store
from scripts.dev_eval import load_jsonl_texts
qr=json.loads((REPO/"artifacts/apps/qrels_train.json").read_text(encoding="utf-8"))
gold={q:next(iter(v)) for q,v in qr.items()}; owner={}
for q,v in qr.items():
    for d in v: owner.setdefault(d,[]).append(q)
qids,qt=load_jsonl_texts(REPO/"artifacts/apps/texts_queries_train.jsonl"); Q=dict(zip(qids,qt))
h,es=load_store(REPO/"data/outcomes/f2llm17b_all_tune_deep50.jsonl.gz"); E={e["qid"]:e for e in es}
ng=json.load(open(sys.argv[1]))["nongold"]
for q,j,d in ng:
    e=E[q]; gj=e["pool"].index(gold[q])
    o=owner.get(d,["(train-less: test-split doc?)"])
    print(f"{q} gold@{gj} nongold {d}@{j} owner={o}")
    print("   Q :",repr(Q[q][:110]))
    for oq in o[:1]:
        if oq in Q: print("   OQ:",repr(Q[oq][:110]))
