"""Where do blank lines in the official (judge) inputs matter?  TUNE-level mapping + train-wide descriptive count."""
import json, sys, re
from collections import Counter
from pathlib import Path
REPO=Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0,str(REPO))
from scripts.replay_outcomes import load_store
from scripts.dev_eval import load_jsonl_texts
qids,qt=load_jsonl_texts(REPO/"artifacts/apps/texts_queries_train.jsonl"); Q=dict(zip(qids,qt))
h,es=load_store(REPO/"data/outcomes/f2llm17b_all_tune_deep50.jsonl.gz")
def interleaved(s):
    lines=s.rstrip("\n").split("\n")
    return len(lines)>=3 and all(l.strip()=="" for l in lines[1::2]) and all(l.strip() for l in lines[0::2])
def anyblank(s): return any(l.strip()=="" for l in s.rstrip("\n").split("\n"))
def collapse(s): return "\n".join(l for l in s.split("\n") if l.strip())+"\n"
def norm_in(x):
    if isinstance(x,list) and all(isinstance(y,str) for y in x): return "\n".join(x)+"\n"
    return x if isinstance(x,str) else json.dumps(x)
text2e={Q[e["qid"]]:e for e in es}
c=Counter(); tune_rows=[]
with open(REPO/"artifacts/oracle/apps_train.jsonl",encoding="utf-8") as f:
    for line in f:
        r=json.loads(line)
        try: io=json.loads(r["input_output"]) if r["input_output"] else {}
        except Exception: io={}
        ins=[norm_in(x) for x in io.get("inputs",[])]
        if not ins or io.get("fn_name"): continue
        c["train_stdin_problems"]+=1
        ob=any(anyblank(x) for x in ins); oi=any(interleaved(x) for x in ins)
        c["train_official_anyblank"]+=ob; c["train_official_interleaved"]+=oi
        e=text2e.get(r["question"])
        if e and e["legacy_examples"]>0 and e["stdin_examples"]:
            c["tune_stdin"]+=1
            ex=[x["stdin"] for x in e["stdin_examples"][:2]]
            pb=any(anyblank(x) for x in ex); pi=any(interleaved(x) for x in ex)
            c[f"tune parsed_blank={pb} parsed_inter={pi} official_blank={ob}"]+=1
            if pb and ob: tune_rows.append((e["qid"],r["url"],[x[:80] for x in ex][:1],[x[:80] for x in ins][:2]))
            if pb and not pi:
                hit=any(collapse(x).strip()==collapse(ex[0]).strip() for x in ins)
                c[f"tune blank-not-interleaved: official==collapsed_sample {hit}"]+=1
for k,v in sorted(c.items()): print(k,v)
print("TUNE statements whose parsed sample AND official inputs both contain blank lines:")
for r in tune_rows: print(r)
