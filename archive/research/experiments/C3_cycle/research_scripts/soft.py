"""TUNE-only replay: evidence for statements where nothing verifies. (a) shape weak tier ungated, (b) crash demotion."""
import json, sys
import numpy as np
from pathlib import Path
REPO = Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0, str(REPO))
from scripts.replay_outcomes import Policy, load_store, rerank, metrics, candidate_tier, _shape, paired_bootstrap
from app.pipeline.execution_rerank import _normalize_output
gold = {q: next(iter(v)) for q, v in json.loads((REPO/"artifacts/apps/qrels_train.json").read_text(encoding="utf-8")).items()}
pol = Policy("c", numeric_tolerance=1e-6, call_style=True, deepen_to=50)
def status(r, e):
    runs=(r.get("stdin_runs") or [])[:2]
    if not runs: return "na"
    if any(x["status"]=="completed" and x["rc"]!=0 for x in runs): return "crash"
    if any(x["status"]!="completed" for x in runs): return "timeout"
    if all(_shape(x["out"])==_shape(_normalize_output(ex["expected"])) and x["out"] for x,ex in zip(runs,e["stdin_examples"])): return "shape"
    return "mismatch"
def variant(e, mode):
    base=rerank(e,pol)
    recs={r["doc"]:r for r in e["candidates"] if r["j"]<20}
    if any(candidate_tier(r,e,pol)=="strong" for r in e["candidates"]): return base
    if not recs or e["legacy_examples"]==0: return base
    st={d:status(r,e) for d,r in recs.items()}
    if mode=="shape_rank2":
        sh=[d for d in base if st.get(d)=="shape" and d!=base[0]]
        return [base[0]]+sh+[d for d in base[1:] if d not in set(sh)]
    if mode=="crash_demote_top20":
        top=base[:20]; ok=[d for d in top if st.get(d)!="crash"]; cr=[d for d in top if st.get(d)=="crash"]
        return ok+cr+base[20:]
    if mode=="crash_demote_keep_top1":
        top=base[:20]; ok=[d for d in top[1:] if st.get(d)!="crash"]; cr=[d for d in top[1:] if st.get(d)=="crash"]
        return [top[0]]+ok+cr+base[20:]
h, es = load_store(REPO/"data/outcomes/f2llm17b_all_tune_deep50.jsonl.gz")
for fam in [None,"competitive_stdin"]:
    E=[e for e in es if fam is None or e["family"]==fam]
    a=np.array([metrics(rerank(e,pol),gold[e["qid"]])[0] for e in E])
    for mode in ["shape_rank2","crash_demote_top20","crash_demote_keep_top1"]:
        b=np.array([metrics(variant(e,mode),gold[e["qid"]])[0] for e in E])
        m,lo,hi,p=paired_bootstrap(a,b)
        print(fam or "all", mode, f"base={a.mean():.5f} d={m:+.5f} [{lo:+.5f},{hi:+.5f}] changed={int((a!=b).sum())} up={int((b>a).sum())} down={int((b<a).sum())}")
