"""TUNE-only: among >=2 verified stdin candidates, does survival on structure-preserving perturbed inputs separate gold from the coincidental verifier?"""
import json, sys, random, re, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO = Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0, str(REPO))
from scripts.replay_outcomes import Policy, load_store, rerank, metrics, candidate_tier
from scripts.dev_eval import load_jsonl_texts
from app.pipeline.execution_rerank import ExecutionConfig, execute_candidate
gold = {q: next(iter(v)) for q, v in json.loads((REPO/"artifacts/apps/qrels_train.json").read_text(encoding="utf-8")).items()}
cids, ctexts = load_jsonl_texts(REPO/"artifacts/apps/texts_corpus.jsonl"); ctext=dict(zip(cids,ctexts))
pol = Policy("c", numeric_tolerance=1e-6, call_style=True, deepen_to=50)
cfg = ExecutionConfig(enabled=True, timeout_seconds=2.0)
def perturb(s, rng):
    out=[]
    for line in s.split("\n"):
        toks=line.split(" ")
        if len([t for t in toks if t])>=2:
            nt=[]
            for t in toks:
                if re.fullmatch(r"-?\d+", t) and abs(int(t))>1: v=int(t); nt.append(str(rng.randint(1, max(abs(v),10))*(1 if v>0 else -1)))
                elif re.fullmatch(r"[A-Za-z]{2,}", t): nt.append("".join(rng.sample(t,len(t))))
                else: nt.append(t)
            out.append(" ".join(nt))
        elif re.fullmatch(r"[A-Za-z.#*]{2,}", line.strip()):
            t=line.strip(); out.append("".join(rng.sample(t,len(t))))
        else: out.append(line)
    return "\n".join(out)
h, es = load_store(REPO/"data/outcomes/f2llm17b_all_tune_deep50.jsonl.gz")
cases=[]
for e in es:
    if e["family"]!="competitive_stdin": continue
    recs={r["doc"]:r for r in e["candidates"]}
    ver=[r["doc"] for r in sorted(e["candidates"],key=lambda r:r["j"]) if r["j"]<20 and candidate_tier(r,e,pol)=="strong"]
    if len(ver)>=2 and gold[e["qid"]] in ver: cases.append((e,ver))
jobs=[]
for e,ver in cases:
    rng=random.Random(e["qid"])
    ins=[perturb(e["stdin_examples"][0]["stdin"], rng) for _ in range(5)]
    for d in ver: jobs.append((e["qid"],d,ins))
def run(job):
    q,d,ins=job; ok=0
    for s in ins:
        o=execute_candidate(ctext[d], s, cfg); ok+= (o.status=="completed" and o.returncode==0 and bool(o.stdout.strip()))
    return q,d,ok
t=time.time()
with ThreadPoolExecutor(5) as p: res=list(p.map(run,jobs))
surv={}; [surv.setdefault(q,{}).__setitem__(d,ok) for q,d,ok in res]
c=Counter(); before=after=0
for e,ver in cases:
    g=gold[e["qid"]]; s=surv[e["qid"]]; base=rerank(e,pol)
    c["cases"]+=1; c["gold_is_top_verified"]+= (ver[0]==g)
    best=max(s.values()); c["gold_max_survival"]+= (s[g]==best); c["gold_unique_max"]+= (s[g]==best and sum(v==best for v in s.values())==1)
    c["gold_survival_sum"]+=s[g]; c["other_survival_sum"]+=sum(v for d,v in s.items() if d!=g); c["others"]+=len(s)-1
    # stable re-order of verified tier by survival desc
    vo=sorted(ver,key=lambda d:-s[d]); new=vo+[d for d in base if d not in set(ver)]
    before+=metrics(base,g)[0]; after+=metrics(new,g)[0]
print(dict(c), f"secs={time.time()-t:.1f}")
print(f"ndcg on these cases: before {before/len(cases):.4f} after {after/len(cases):.4f}; gold mean survival {c['gold_survival_sum']/c['cases']:.2f}/5, others {c['other_survival_sum']/max(1,c['others']):.2f}/5")
