"""TUNE only: raw statement text vs parsed stdin vs APPS official test inputs for interleaved-blank statements."""
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
tg=[e for e in es if e["legacy_examples"]>0 and e["stdin_examples"] and interleaved(e["stdin_examples"][0]["stdin"])]
want={Q[e["qid"]]:e for e in tg}
apps={}
with open(REPO/"artifacts/oracle/apps_train.jsonl",encoding="utf-8") as f:
    for line in f:
        r=json.loads(line)
        if r["question"] in want: apps[r["question"]]=r
print("affected",len(tg),"matched to apps_train by exact text",len(apps))
stats={"official_has_interleaved":0,"official_has_anyblank":0,"sample_in_official_raw":0,"sample_in_official_collapsed":0,"n_official_inputs":0}
rows=[]
for e in tg:
    t=Q[e["qid"]]; ex=e["stdin_examples"][0]["stdin"]
    r=apps.get(t)
    io=json.loads(r["input_output"]) if r and r["input_output"] else {}
    ins=io.get("inputs",[])
    ins_s=[x if isinstance(x,str) else json.dumps(x) for x in ins]
    stats["n_official_inputs"]+=len(ins_s)
    inter=any(interleaved(x) for x in ins_s); ab=any(any(l.strip()=="" for l in x.rstrip("\n").split("\n")) for x in ins_s)
    raw_hit=any(x.strip()==ex.strip() for x in ins_s); col_hit=any(collapse(x).strip()==collapse(ex).strip() for x in ins_s)
    stats["official_has_interleaved"]+=inter; stats["official_has_anyblank"]+=ab
    stats["sample_in_official_raw"]+=raw_hit; stats["sample_in_official_collapsed"]+=col_hit
    i=t.find(ex.split("\n")[0])
    rows.append((e["qid"], r["url"] if r else None, r["difficulty"] if r else None, len(ins_s), inter, ab, raw_hit, col_hit))
print(stats)
for x in rows: print(x)
print("\n==== 5 raw examples ====")
for e in tg[:5]:
    t=Q[e["qid"]]; ex=e["stdin_examples"][0]["stdin"]
    k=t.find("Input")
    m=re.search(r"-{3,}\s*Examples?", t)
    s=m.start() if m else max(0,len(t)-500)
    print(e["qid"], "RAW:", repr(t[s:s+260]))
    print("   PARSED:", repr(ex[:120]))
    r=apps.get(t)
    if r:
        ins=json.loads(r["input_output"])["inputs"]; print("   OFFICIAL[0]:", repr((ins[0] if isinstance(ins[0],str) else str(ins[0]))[:120]))
