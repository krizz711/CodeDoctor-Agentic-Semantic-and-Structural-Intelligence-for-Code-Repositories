"""TUNE only: prevalence of blank lines in parsed stdin examples; raw statement check."""
import json, sys, re
from collections import Counter
from pathlib import Path
REPO=Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0,str(REPO))
from scripts.replay_outcomes import load_store
from scripts.dev_eval import load_jsonl_texts
h,es=load_store(REPO/"data/outcomes/f2llm17b_all_tune_deep50.jsonl.gz")
print("header",{k:h[k] for k in ("model","split","family","seed","exec_depth","timeout","workers")})
qids,qt=load_jsonl_texts(REPO/"artifacts/apps/texts_queries_train.jsonl"); Q=dict(zip(qids,qt))
def interleaved(s):
    lines=s.rstrip("\n").split("\n")
    return len(lines)>=3 and all(l.strip()=="" for l in lines[1::2]) and all(l.strip() for l in lines[0::2])
def anyblank(s): return any(l.strip()=="" for l in s.rstrip("\n").split("\n"))
c=Counter()
for e in es:
    c["all"]+=1
    if e["stdin_examples"]: c["has_stdin_examples"]+=1
    if e["legacy_examples"]>0: c["legacy>0"]+=1
    if e["legacy_examples"]>0 and e["stdin_examples"]:
        c["stdin_stmts(legacy>0)"]+=1
        ex=e["stdin_examples"]
        if interleaved(ex[0]["stdin"]): c["interleaved_ex0"]+=1
        if any(interleaved(x["stdin"]) for x in ex[:2]): c["interleaved_any_first2"]+=1
        if any(anyblank(x["stdin"]) for x in ex[:2]): c["anyblank_first2"]+=1
        if any(anyblank(x["stdin"]) and not interleaved(x["stdin"]) for x in ex[:2]): c["blank_not_interleaved_first2"]+=1
        if any(interleaved(x["expected"]) or anyblank(x["expected"]) for x in ex[:2]): c["blank_in_expected"]+=1
    fam=e["family"]; c["fam:"+fam]+=1
print(dict(c))
