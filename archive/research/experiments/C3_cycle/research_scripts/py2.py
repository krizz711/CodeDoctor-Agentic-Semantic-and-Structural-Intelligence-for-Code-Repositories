import json, sys, warnings
from collections import Counter
from pathlib import Path
REPO = Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0, str(REPO))
from scripts.dev_eval import load_jsonl_texts
warnings.simplefilter("ignore")
cids, ctexts = load_jsonl_texts(REPO/"artifacts/apps/texts_corpus.jsonl")
msgs=Counter(); ex=[]
for d,t in zip(cids,ctexts):
    try: compile(t, "<doc>", "exec", dont_inherit=True)
    except SyntaxError as e:
        m=str(e.msg)[:60]; msgs[m]+=1
        if len(ex)<3 and "print" in m.lower(): ex.append((d, e.lineno, t.splitlines()[e.lineno-1][:80] if e.lineno and e.lineno<=len(t.splitlines()) else ""))
print(msgs.most_common(15)); print(ex)
