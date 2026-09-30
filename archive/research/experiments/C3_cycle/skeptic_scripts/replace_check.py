"""TUNE: would collapse-INSTEAD-of-raw lose any raw-strong verification? (cheaper design)."""
import json, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO=Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0,str(REPO))
from scripts.replay_outcomes import Policy, load_store, stdin_verdict
from scripts.dev_eval import load_jsonl_texts
from scripts.record_outcomes import _stdin_record
from app.pipeline.execution_rerank import ExecutionConfig, execute_candidate
gold={q:next(iter(v)) for q,v in json.loads((REPO/"artifacts/apps/qrels_train.json").read_text(encoding="utf-8")).items()}
cids,ct=load_jsonl_texts(REPO/"artifacts/apps/texts_corpus.jsonl"); C=dict(zip(cids,ct))
cfg=ExecutionConfig(enabled=True,timeout_seconds=2.0)
def interleaved(s):
    lines=s.rstrip("\n").split("\n")
    return len(lines)>=3 and all(l.strip()=="" for l in lines[1::2]) and all(l.strip() for l in lines[0::2])
def anyblank(s): return any(l.strip()=="" for l in s.rstrip("\n").split("\n"))
def collapse(s): return "\n".join(l for l in s.split("\n") if l.strip())+"\n"
h,es=load_store(REPO/"data/outcomes/f2llm17b_all_tune_deep50.jsonl.gz")
pol=Policy("p",numeric_tolerance=1e-6)
for name,det in [("interleaved",interleaved),("anyblank",anyblank)]:
    jobs=[]
    for e in es:
        if not(e["legacy_examples"]>0 and e["stdin_examples"]): continue
        for r in e["candidates"]:
            for k,(run,ex) in enumerate(list(zip(r.get("stdin_runs") or [],e["stdin_examples"]))[:2]):
                if det(ex["stdin"]) and stdin_verdict(run,ex["expected"],e["rules"],pol)=="strong":
                    jobs.append((e,r,ex))
    def f(j):
        e,r,ex=j; rec=_stdin_record(execute_candidate(C[r["doc"]],collapse(ex["stdin"]),cfg))
        return e["qid"],r["doc"]==gold[e["qid"]],stdin_verdict(rec,ex["expected"],e["rules"],pol)
    with ThreadPoolExecutor(4) as p: res=list(p.map(f,jobs))
    c=Counter((g,v) for _,g,v in res)
    print(name,"raw-strong runs:",len(jobs),"collapsed verdicts (is_gold,verdict):",dict(c),"lost in:",sorted({q for q,g,v in res if v!="strong"}))
