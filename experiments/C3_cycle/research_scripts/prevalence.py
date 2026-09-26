import json, sys, re, warnings
from collections import Counter
from pathlib import Path
REPO = Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0, str(REPO))
from scripts.replay_outcomes import Policy, load_store, rerank, metrics, candidate_tier
from scripts.dev_eval import load_jsonl_texts
warnings.simplefilter("ignore")
gold = {q: next(iter(v)) for q, v in json.loads((REPO/"artifacts/apps/qrels_train.json").read_text(encoding="utf-8")).items()}
pol = Policy("c", numeric_tolerance=1e-6, call_style=True, deepen_to=50)
def interleaved(s):
    lines = s.rstrip("\n").split("\n")
    return len(lines) >= 3 and all(l.strip()=="" for l in lines[1::2]) and all(l.strip() for l in lines[0::2])
for split in ["dev","tune"]:
    h, es = load_store(REPO/f"data/outcomes/f2llm17b_all_{split}_deep50.jsonl.gz")
    c = Counter()
    for e in es:
        if not e["stdin_examples"] or e["legacy_examples"]==0: continue
        c["stdin_stmts_with_examples"]+=1
        s = e["stdin_examples"][0]["stdin"]
        if "\n\n" in s.strip("\n"): c["has_blank_line"]+=1
        if interleaved(s): 
            c["interleaved"]+=1
            g=gold[e["qid"]]; recs={r["doc"]:r for r in e["candidates"]}
            nd=metrics(rerank(e,pol),g)[0]; c["interleaved_ndcg_sum"]+=nd
            ver=[r for r in e["candidates"] if r["j"]<20 and candidate_tier(r,e,pol)=="strong"]
            c["interleaved_any_verified"]+= bool(ver)
            if g in recs:
                runs=recs[g].get("stdin_runs") or []
                if runs and runs[0]["status"]=="completed" and runs[0]["rc"]!=0: c["interleaved_gold_crash"]+=1
                elif runs and candidate_tier(recs[g],e,pol)=="strong": c["interleaved_gold_verified"]+=1
            # fraction of candidates that crash on interleaved inputs
            for r in e["candidates"][:20]:
                rr=(r.get("stdin_runs") or [None])[0]
                if rr: c["cand_runs"]+=1; c["cand_crash"]+= (rr["status"]=="completed" and rr["rc"]!=0)
    print(split, dict(c))
# corpus: Python-3 parse failures
cids, ctexts = load_jsonl_texts(REPO/"artifacts/apps/texts_corpus.jsonl")
train_golds=set(gold.values())
fail=Counter(); py2=Counter()
P2=re.compile(r"^\s*print\s+[^\s(=]|raw_input\s*\(|xrange\s*\(|^\s*exec\s+\"", re.M)
for d,t in zip(cids,ctexts):
    try: compile(t, "<doc>", "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as ex:
        k = "train_gold" if d in train_golds else "other"
        fail[k]+=1
        if P2.search(t): py2[k]+=1
    except Exception as ex: fail["other_err"]+=1
print("corpus docs", len(cids), "py3 compile failures", dict(fail), "of which py2 markers", dict(py2))
