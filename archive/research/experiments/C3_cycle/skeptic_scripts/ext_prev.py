import json, sys
from collections import Counter
from pathlib import Path
REPO=Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0,str(REPO))
from scripts.replay_outcomes import load_store
def interleaved(s):
    lines=s.rstrip("\n").split("\n")
    return len(lines)>=3 and all(l.strip()=="" for l in lines[1::2]) and all(l.strip() for l in lines[0::2])
def anyblank(s): return any(l.strip()=="" for l in s.rstrip("\n").split("\n"))
h,es=load_store(REPO/"data/outcomes/f2llm17b_extcc_dev_deep50.jsonl.gz")
c=Counter()
for e in es:
    if e["stdin_examples"]:
        c["stdin"]+=1; ex=[x["stdin"] for x in e["stdin_examples"][:2]]
        c["interleaved"]+=any(interleaved(x) for x in ex); c["anyblank"]+=any(anyblank(x) for x in ex)
print("EXT-dev (store header split:",h["split"],")",dict(c),"n=",len(es))
