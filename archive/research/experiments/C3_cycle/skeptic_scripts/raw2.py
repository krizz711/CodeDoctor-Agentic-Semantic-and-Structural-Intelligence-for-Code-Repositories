import json, sys, re
from pathlib import Path
REPO=Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0,str(REPO))
from scripts.replay_outcomes import load_store
from scripts.dev_eval import load_jsonl_texts
h,es=load_store(REPO/"data/outcomes/f2llm17b_all_tune_deep50.jsonl.gz")
qids,qt=load_jsonl_texts(REPO/"artifacts/apps/texts_queries_train.jsonl"); Q=dict(zip(qids,qt))
def interleaved(s):
    lines=s.rstrip("\n").split("\n")
    return len(lines)>=3 and all(l.strip()=="" for l in lines[1::2]) and all(l.strip() for l in lines[0::2])
def collapse(s): return "\n".join(l for l in s.split("\n") if l.strip())+"\n"
def norm_in(x): return "\n".join(x)+"\n" if isinstance(x,list) else x
tg=[e for e in es if e["legacy_examples"]>0 and e["stdin_examples"] and interleaved(e["stdin_examples"][0]["stdin"])]
want={Q[e["qid"]]:e for e in tg}; apps={}
with open(REPO/"artifacts/oracle/apps_train.jsonl",encoding="utf-8") as f:
    for line in f:
        r=json.loads(line)
        if r["question"] in want: apps[r["question"]]=r
raw_eq=col_eq=inter=0; types=set()
for e in tg:
    t=Q[e["qid"]]; ex=e["stdin_examples"][0]["stdin"]
    io=json.loads(apps[t]["input_output"]); ins=io["inputs"]; types|={type(x).__name__ for x in ins}
    s=[norm_in(x) for x in ins]
    raw_eq+=any(x.strip()==ex.strip() for x in s); col_eq+=any(collapse(x).strip()==collapse(ex).strip() for x in s)
    inter+=any(interleaved(x) for x in s)
print("input types",types,"official==raw sample",raw_eq,"official==collapsed sample",col_eq,"official interleaved",inter,"of",len(tg))
for e in tg[:6]:
    t=Q[e["qid"]]; m=re.search(r"-{3,}\s*Sample Input[^\n]*\n", t) or re.search(r"Sample Input|Input\s*\n",t)
    print(e["qid"],"RAW-SAMPLE:",repr(t[m.start():m.start()+140]) if m else None)
    print("    raw has \n\n inside sample:", "\n\n" in t[m.end():m.end()+60] if m else None)
