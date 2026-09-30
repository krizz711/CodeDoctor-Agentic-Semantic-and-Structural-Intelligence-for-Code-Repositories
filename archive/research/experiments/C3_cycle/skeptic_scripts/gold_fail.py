import json, sys
from pathlib import Path
REPO=Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0,str(REPO))
from scripts.replay_outcomes import load_store
from scripts.dev_eval import load_jsonl_texts
from app.pipeline.execution_rerank import ExecutionConfig, execute_candidate, _normalize_output
gold={q:next(iter(v)) for q,v in json.loads((REPO/"artifacts/apps/qrels_train.json").read_text(encoding="utf-8")).items()}
cids,ct=load_jsonl_texts(REPO/"artifacts/apps/texts_corpus.jsonl"); C=dict(zip(cids,ct))
cfg=ExecutionConfig(enabled=True,timeout_seconds=2.0)
def anyblank(s): return any(l.strip()=="" for l in s.rstrip("\n").split("\n"))
def collapse(s): return "\n".join(l for l in s.split("\n") if l.strip())+"\n"
h,es=load_store(REPO/"data/outcomes/f2llm17b_all_tune_deep50.jsonl.gz")
for e in es:
    if not(e["legacy_examples"]>0 and e["stdin_examples"] and any(anyblank(x["stdin"]) for x in e["stdin_examples"][:2])): continue
    g=gold[e["qid"]]; ok=True; det=[]
    for ex in e["stdin_examples"][:2]:
        o=execute_candidate(C[g], ex["stdin"], cfg)
        if o.ok and _normalize_output(o.stdout)==_normalize_output(ex["expected"]): det.append("raw-ok"); continue
        o=execute_candidate(C[g], collapse(ex["stdin"]), cfg)
        if o.ok and _normalize_output(o.stdout)==_normalize_output(ex["expected"]): det.append("col-ok"); continue
        ok=False; det.append(f"FAIL {o.status} rc={o.returncode} exc={o.exception} out={_normalize_output(o.stdout)[:40]!r} want={_normalize_output(ex['expected'])[:40]!r} err={o.stderr_tail.strip().splitlines()[-1][:70] if o.stderr_tail.strip() else ''}")
    if not ok: print(e["qid"], det)
