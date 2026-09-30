"""TUNE only: Arm B prevalence and live effect. Re-run (file stdin + exit/quit bound in builtins via -c wrapper) every
recorded stdin run that did not verify, for candidates whose code uses the idioms; score with unchanged replay."""
import json, sys, re, os, subprocess, tempfile, copy, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO=Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0,str(REPO))
from scripts.replay_outcomes import Policy, load_store, rerank, metrics, candidate_tier, stdin_verdict
from scripts.dev_eval import load_jsonl_texts
from app.pipeline.execution_rerank import _normalize_output
gold={q:next(iter(v)) for q,v in json.loads((REPO/"artifacts/apps/qrels_train.json").read_text(encoding="utf-8")).items()}
cids,ct=load_jsonl_texts(REPO/"artifacts/apps/texts_corpus.jsonl"); C=dict(zip(cids,ct))
EXIT=re.compile(r"(?<![\w.])(?:exit|quit)\s*\("); FST=re.compile(r"fstat\s*\(\s*0|os\.read\s*\(\s*0")
c0=Counter()
for d,t in C.items():
    c0["corpus_docs"]+=1; c0["bare_exit_quit"]+=bool(EXIT.search(t)); c0["fstat_or_osread0"]+=bool(FST.search(t)); c0["__future__"]+="from __future__" in t
print("corpus-wide idiom counts:",dict(c0))
WRAP=("import sys,builtins\nbuiltins.exit=builtins.quit=sys.exit\np=sys.argv[1]\nsys.argv=[p]\n"
      "g={'__name__':'__main__','__file__':p,'__builtins__':builtins}\nexec(compile(open(p,'rb').read(),p,'exec'),g)\n")
def run(code, stdin):
    with tempfile.TemporaryDirectory() as d:
        p=os.path.join(d,"candidate.py"); open(p,"w",encoding="utf-8").write(code)
        w=os.path.join(d,"_w.py"); open(w,"w",encoding="utf-8").write(WRAP)
        f=os.path.join(d,"in.txt"); open(f,"wb").write(stdin.encode("utf-8"))
        env={"PATH":os.environ.get("PATH",""),"SYSTEMROOT":os.environ.get("SYSTEMROOT",""),"PYTHONIOENCODING":"utf-8"}
        try:
            with open(f,"rb") as fh: r=subprocess.run([sys.executable,"-I","-S",w,p],stdin=fh,capture_output=True,timeout=2.0,cwd=d,env=env)
        except subprocess.TimeoutExpired: return {"status":"timeout","rc":None,"out":""}
        return {"status":"completed","rc":r.returncode,"out":_normalize_output(r.stdout.decode("utf-8","replace"))[:4000]}
h,es=load_store(REPO/"data/outcomes/f2llm17b_all_tune_deep50.jsonl.gz")
pol=Policy("call-style+deep50",numeric_tolerance=1e-6,call_style=True,deepen_to=50)
jobs=[]
for e in es:
    if not(e["legacy_examples"]>0 and e["stdin_examples"]): continue
    for r in e["candidates"]:
        t=C[r["doc"]]
        if not(EXIT.search(t) or FST.search(t)): continue
        for k,(run_,ex) in enumerate(list(zip(r.get("stdin_runs") or [],e["stdin_examples"]))[:2]):
            if stdin_verdict(run_,ex["expected"],e["rules"],pol)!="strong": jobs.append((e["qid"],r["j"],r["doc"],k,ex["stdin"]))
t0=time.time()
with ThreadPoolExecutor(4) as pool: res=list(pool.map(lambda j:(j,run(C[j[2]],j[4])),jobs))
fb={(q,j,k):rec for (q,j,d,k,s),rec in res}
c=Counter(); before=after=0.0; changed=0
for e in es:
    if not(e["legacy_examples"]>0 and e["stdin_examples"]): continue
    g=gold[e["qid"]]; e2=copy.deepcopy(e); touched=False
    for r in e2["candidates"]:
        for k in range(2):
            if (e["qid"],r["j"],k) in fb:
                old=r["stdin_runs"][k]; new=dict(old); new.update(fb[(e["qid"],r["j"],k)]); r["stdin_runs"][k]=new; touched=True
    if not touched: continue
    b=metrics(rerank(e,pol),g)[0]; a=metrics(rerank(e2,pol),g)[0]; before+=b; after+=a; changed+=(a!=b)
    for r0,r1 in zip(e["candidates"],e2["candidates"]):
        t0_,t1_=candidate_tier(r0,e,pol),candidate_tier(r1,e2,pol)
        role="gold" if r1["doc"]==g else "nongold"
        if t0_!="strong" and t1_=="strong": c["new_strong_"+role]+=1
        if t0_=="strong" and t1_!="strong": c["lost_"+role]+=1
print(f"TUNE stdin: re-runs={len(jobs)} secs={time.time()-t0:.0f} {dict(c)} statements with changed NDCG={changed} sumdelta={after-before:+.3f} per777={(after-before)/777:+.5f}")
print("rerun outcomes:",Counter((r["status"],r["rc"]) for _,r in res))
print("wrapper sanity (bare exit prog, file stdin):", run("n=int(input())\nprint(n)\nexit()\n","7\n"), run("import os\nprint(len(os.read(0,os.fstat(0).st_size)))\n","7\n"))
g_in=[(q,j,d) for (q,j,d,k,s),r in res if d==gold[q]]
print("gold candidates among reruns:",len(g_in), [(q,j,d,r["rc"]) for (q,j,d,k,s),r in res if d==gold[q]][:10])
