"""TUNE only. Arm A replayed properly: fallback run on collapsed input only when the raw run is not 'strong',
spliced into the store record, then scored with the unchanged replay policy (call-style+deep50 and legacy+numeric).
Workers=4. Also a negative control: collapsed outputs compared against a *different* statement's expected output."""
import json, sys, time, copy, random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO=Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0,str(REPO))
from scripts.replay_outcomes import Policy, load_store, rerank, metrics, candidate_tier, stdin_verdict
from scripts.dev_eval import load_jsonl_texts
from scripts.record_outcomes import _stdin_record
from app.pipeline.execution_rerank import ExecutionConfig, execute_candidate
gold={q:next(iter(v)) for q,v in json.loads((REPO/"artifacts/apps/qrels_train.json").read_text(encoding="utf-8")).items()}
cids,ct=load_jsonl_texts(REPO/"artifacts/apps/texts_corpus.jsonl"); C=dict(zip(cids,ct))
cfg=ExecutionConfig(enabled=True,timeout_seconds=2.0,call_style=True,extended_stdin=True,max_examples=5)
MODE=sys.argv[1] if len(sys.argv)>1 else "interleaved"
def interleaved(s):
    lines=s.rstrip("\n").split("\n")
    return len(lines)>=3 and all(l.strip()=="" for l in lines[1::2]) and all(l.strip() for l in lines[0::2])
def anyblank(s): return any(l.strip()=="" for l in s.rstrip("\n").split("\n"))
def _collapse(s): return "\n".join(l for l in s.split("\n") if l.strip())+"\n"
def _placebo(s):
    L=[l for l in s.split("\n") if l.strip()]
    return "\n".join(L[:1]+L[1:][::-1])+"\n"  # collapsed, then lines after the first reversed
PLACEBO=len(sys.argv)>2 and sys.argv[2]=="placebo"
collapse=_placebo if PLACEBO else _collapse
if PLACEBO: MODE_SUFFIX="_placebo"
det={"interleaved":interleaved,"anyblank":anyblank}[MODE]
h,es=load_store(REPO/"data/outcomes/f2llm17b_all_tune_deep50.jsonl.gz")
pols=[Policy("legacy+numeric1e-6",numeric_tolerance=1e-6), Policy("call-style+deep50",numeric_tolerance=1e-6,call_style=True,deepen_to=50)]
tg=[e for e in es if e["legacy_examples"]>0 and e["stdin_examples"] and any(det(x["stdin"]) for x in e["stdin_examples"][:2])]
jobs=[]
for e in tg:
    for r in e["candidates"]:
        runs=r.get("stdin_runs") or []
        for k,(run,ex) in enumerate(list(zip(runs,e["stdin_examples"]))[:2]):
            if det(ex["stdin"]) and stdin_verdict(run,ex["expected"],e["rules"],pols[0])!="strong":
                jobs.append((e["qid"],r["j"],r["doc"],k,collapse(ex["stdin"])))
t=time.time()
def run(job):
    q,j,d,k,s=job
    return job, _stdin_record(execute_candidate(C[d], s, cfg, track_input=True))
with ThreadPoolExecutor(4) as pool: res=list(pool.map(run,jobs))
secs=time.time()-t
fb={(q,j,k):rec for (q,j,d,k,s),rec in res}
fbin={(q,j,k):s for (q,j,d,k,s),rec in res}
c=Counter(); before={p.name:0.0 for p in pols}; after={p.name:0.0 for p in pols}
nongold_docs=[]
neg=Counter()
exp_by_q={e["qid"]:e["stdin_examples"] for e in tg}
qs=[e["qid"] for e in tg]
for e in tg:
    g=gold[e["qid"]]; e2=copy.deepcopy(e)
    for r in e2["candidates"]:
        for k in range(2):
            if (e["qid"],r["j"],k) in fb:
                r["stdin_runs"][k]=fb[(e["qid"],r["j"],k)]
    for p in pols:
        b_=metrics(rerank(e,p),g)[0]; a_=metrics(rerank(e2,p),g)[0]; before[p.name]+=b_; after[p.name]+=a_
        c[f"{p.name}:up"]+=a_>b_; c[f"{p.name}:down"]+=a_<b_
        if a_!=b_ and p.name=="call-style+deep50": print("CHANGED",e["qid"],"gold@",e["pool"].index(g),f"{b_:.3f}->{a_:.3f}", "new top3", rerank(e2,p)[:3], "gold", g)
    # variant: accept a fallback run only if it consumed every non-blank input char (tracked) -- else keep raw run
    e3=copy.deepcopy(e)
    for r in e3["candidates"]:
        for k in range(2):
            key=(e["qid"],r["j"],k)
            if key in fb:
                rec=fb[key]; s_in=fbin[key]
                if rec.get("tracked") and (rec.get("read") or 0)>=len(s_in.rstrip()):
                    r["stdin_runs"][k]=rec
                elif rec.get("tracked") is False: c["fallback_untracked"]+=1; r["stdin_runs"][k]=rec
    pc=pols[1]; b3=metrics(rerank(e,pc),g)[0]; a3=metrics(rerank(e3,pc),g)[0]
    c["consume:sumdelta_x1000"]+=round((a3-b3)*1000); c["consume:up"]+=a3>b3; c["consume:down"]+=a3<b3
    for r0,r3 in zip(e["candidates"],e3["candidates"]):
        if candidate_tier(r0,e,pc)!="strong" and candidate_tier(r3,e3,pc)=="strong": c["consume:new_strong_"+("gold" if r3["doc"]==g else "nongold")]+=1
    p=pols[1]
    for r0,r1 in zip(e["candidates"],e2["candidates"]):
        t0,t1=candidate_tier(r0,e,p),candidate_tier(r1,e2,p)
        role="gold" if r1["doc"]==g else "nongold"
        if t0!="strong" and t1=="strong":
            c[f"new_strong_{role}"]+=1
            if role=="nongold": nongold_docs.append((e["qid"],r1["j"],r1["doc"])); c["nongold_j<20"]+=r1["j"]<20
        if t0=="strong" and t1!="strong": c["LOST_strong"]+=1
        if role=="gold": c[f"gold_{t0}->{t1}"]+=1
    # negative control: fallback outputs vs another statement's expected
    other=exp_by_q[random.Random(hash(e["qid"])%1000).choice([x for x in qs if x!=e["qid"]])]
    for r1 in e2["candidates"]:
        runs=r1.get("stdin_runs") or []
        if runs and runs[0]["status"]=="completed" and runs[0]["rc"]==0 and other and other[0]["expected"]:
            from app.pipeline.execution_rerank import _normalize_output
            neg["runs"]+=1; neg["match_other_expected"]+= runs[0]["out"]==_normalize_output(other[0]["expected"])
c["stmts_with_new_nongold"]=len({q for q,_,_ in nongold_docs})
n=len(tg)
print(f"mode={MODE} statements={n} fallback_runs={len(jobs)} secs={secs:.1f}")
print(dict(c)); print("negative control (shuffled expected):",dict(neg))
for p in pols: print(f"{p.name}: NDCG@10 on these {n}: {before[p.name]/n:.4f} -> {after[p.name]/n:.4f}; per 777 TUNE stdin: {(after[p.name]-before[p.name])/777:+.5f}; per 2500 TUNE: {(after[p.name]-before[p.name])/2500:+.5f}")
print("nongold new strong:",nongold_docs)
json.dump({"nongold":nongold_docs},open(Path(__file__).with_name(f"nongold_{MODE}.json"),"w"))
