"""TUNE-only mechanism check: does collapsing interleaved blank lines in the sample input let golds verify,
and how many non-gold candidates verify too? DEV is not touched (kept for a pre-registered decision)."""
import json, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO = Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0, str(REPO))
from scripts.replay_outcomes import Policy, load_store, rerank, metrics, candidate_tier
from scripts.dev_eval import load_jsonl_texts
from app.pipeline.execution_rerank import ExecutionConfig, execute_candidate, compare_output
gold = {q: next(iter(v)) for q, v in json.loads((REPO/"artifacts/apps/qrels_train.json").read_text(encoding="utf-8")).items()}
cids, ctexts = load_jsonl_texts(REPO/"artifacts/apps/texts_corpus.jsonl"); ctext=dict(zip(cids,ctexts))
pol = Policy("c", numeric_tolerance=1e-6, call_style=True, deepen_to=50)
cfg = ExecutionConfig(enabled=True, timeout_seconds=2.0, numeric_tolerance=1e-6)
def interleaved(s):
    lines = s.rstrip("\n").split("\n")
    return len(lines) >= 3 and all(l.strip()=="" for l in lines[1::2]) and all(l.strip() for l in lines[0::2])
def collapse(s): return "\n".join(l for l in s.split("\n") if l.strip()) + "\n"
h, es = load_store(REPO/"data/outcomes/f2llm17b_all_tune_deep50.jsonl.gz")
targets=[e for e in es if e["stdin_examples"] and e["legacy_examples"]>0 and interleaved(e["stdin_examples"][0]["stdin"])]
jobs=[]
for e in targets:
    exs=e["stdin_examples"][:2]
    for r in e["candidates"]:
        if r["j"]<20 and r.get("stdin_runs"):
            jobs.append((e,r))
def run(job):
    e,r=job
    ok=True
    for ex in e["stdin_examples"][:2]:
        o=execute_candidate(ctext[r["doc"]], collapse(ex["stdin"]), cfg)
        if not o.ok or compare_output(o.stdout, ex["expected"], cfg) is None: ok=False; break
    return e["qid"], r["j"], r["doc"], ok
t=time.time()
with ThreadPoolExecutor(5) as pool: res=list(pool.map(run, jobs))
ver={}
for qid,j,d,ok in res: ver.setdefault(qid,{})[j]=ok
c=Counter(); before=after=0.0
for e in targets:
    g=gold[e["qid"]]; v=ver.get(e["qid"],{})
    base=rerank(e,pol); before+=metrics(base,g)[0]
    # new ranking: collapsed-verified in dense order first, else baseline
    vj=sorted(j for j,ok in v.items() if ok)
    old_strong=[j for j in range(len(e["pool"])) if any(r["j"]==j and r["j"]<20 and candidate_tier(r,e,pol)=="strong" for r in e["candidates"])]
    prom=sorted(set(vj)|set(old_strong))
    new=[e["pool"][j] for j in prom]+[d for k,d in enumerate(e["pool"]) if k not in set(prom)] if prom else base
    after+=metrics(new,g)[0]
    gj=e["pool"].index(g) if g in e["pool"] else None
    c["stmts"]+=1; c["gold_in_top20"]+= (gj is not None and gj<20)
    c["gold_verifies_collapsed"]+= bool(gj is not None and v.get(gj))
    c["nongold_verify"]+= sum(1 for j,ok in v.items() if ok and j!=gj)
    c["stmts_with_nongold_verify"]+= any(ok and j!=gj for j,ok in v.items())
    c["cands_run"]+=len(v)
print(dict(c), f"secs={time.time()-t:.1f}")
print(f"TUNE interleaved statements: NDCG@10 before={before/len(targets):.4f} after={after/len(targets):.4f}; on the 825 TUNE stdin statements: +{(after-before)/825:.4f}; on all 2500 TUNE: +{(after-before)/2500:.4f}")
