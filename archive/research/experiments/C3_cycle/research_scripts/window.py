import json, sys
from collections import Counter
from pathlib import Path
REPO = Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0, str(REPO))
from scripts.replay_outcomes import Policy, load_store, rerank, metrics
gold = {q: next(iter(v)) for q, v in json.loads((REPO/"artifacts/apps/qrels_train.json").read_text(encoding="utf-8")).items()}
pol = Policy("c", numeric_tolerance=1e-6, call_style=True, deepen_to=50)
for split in ["dev","tune"]:
    h, es = load_store(REPO/f"data/outcomes/f2llm17b_all_{split}_deep50.jsonl.gz")
    c = Counter(); fams=Counter(); ranks=[]; ex=Counter(); nexamples=Counter()
    for e in es:
        g = gold[e["qid"]]; nd = metrics(rerank(e,pol), g)[0]
        if nd==1.0 or g not in e["pool"]: continue
        if any(r["doc"]==g for r in e["candidates"]): continue
        c["n"]+=1; fams[e["family"]]+=1; ranks.append(e["pool"].index(g)+1)
        ex[(e["legacy_examples"]>0, len(e["stdin_examples"]), len(e["call_examples"]), len(e["candidates"]))]+=1
    import numpy as np
    print(split, c, fams, "gold dense rank median", np.median(ranks), "hist", np.histogram(ranks,[1,2,3,6,11,21,51])[0])
    print("  (legacy_examples>0, #stdin_ex, #call_ex, #cand_records) top:", ex.most_common(8))
    # overall how many statements have zero candidate records
    z = Counter((e["family"], len(e["candidates"])==0) for e in es)
    print("  zero-record statements by family:", {k:v for k,v in z.items() if k[1]})
