"""TUNE-only replay: trim trailing prose from expected sample output (fallback-only relaxation)."""
import json, sys, re, dataclasses
import numpy as np
from collections import Counter
from pathlib import Path
REPO = Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0, str(REPO))
import scripts.replay_outcomes as ro
from scripts.replay_outcomes import Policy, load_store, rerank, metrics, candidate_tier, paired_bootstrap
from app.pipeline.execution_rerank import _normalize_output
gold = {q: next(iter(v)) for q, v in json.loads((REPO/"artifacts/apps/qrels_train.json").read_text(encoding="utf-8")).items()}
pol = Policy("c", numeric_tolerance=1e-6, call_style=True, deepen_to=50)
PROSE=re.compile(r"(?i)^(note|explanation|by|here|in the first|in the second|for the|the answer|this is|because)\b|\[a-z]+|[A-Za-z]{3,}(\s+[A-Za-z,.'()]+){4,}")
def trim(exp):
    lines=_normalize_output(exp).split("\n")
    k=len(lines)
    while k>1 and PROSE.search(lines[k-1]): k-=1
    # require that every removed line is prose, and the kept block is non-empty
    return "\n".join(lines[:k]) if k<len(lines) else None
def trimmed_entry(e):
    ex2=[]; changed=False
    for ex in e["stdin_examples"]:
        t=trim(ex["expected"])
        if t is not None: changed=True; ex2.append({**ex,"expected":t})
        else: ex2.append(ex)
    return ({**e,"stdin_examples":ex2} if changed else None)
split=sys.argv[1] if len(sys.argv)>1 else "tune"
assert split=="tune"
h, es = load_store(REPO/f"data/outcomes/f2llm17b_all_{split}_deep50.jsonl.gz")
c=Counter(); a=[]; b=[]; fams=[]
for e in es:
    g=gold[e["qid"]]; base=rerank(e,pol); nb=metrics(base,g)[0]
    t=trimmed_entry(e) if e["legacy_examples"]>0 else None
    if t is None: a.append(nb); b.append(nb); fams.append(e["family"]); continue
    c["stmts_trimmed"]+=1
    # fallback-only: raw strong stays strong; trimmed can add strong
    raw_strong={r["j"] for r in e["candidates"] if candidate_tier(r,e,pol)=="strong"}
    new=rerank(t,pol)
    tr_strong={r["j"] for r in t["candidates"] if candidate_tier(r,t,pol)=="strong"}
    gj=e["pool"].index(g) if g in e["pool"] else -1
    c["gold_newly_verified"]+= (gj in tr_strong and gj not in raw_strong)
    c["nongold_newly_verified"]+= len({j for j in tr_strong-raw_strong if j!=gj})
    c["raw_strong_lost"]+= len(raw_strong-tr_strong)
    a.append(nb); b.append(metrics(new,g)[0]); fams.append(e["family"])
a=np.array(a); b=np.array(b); fams=np.array(fams)
print(dict(c))
for name,mask in [("all",np.ones(len(a),bool)),("stdin",fams=="competitive_stdin")]:
    m,lo,hi,p=paired_bootstrap(a[mask],b[mask]); print(name, f"d={m:+.5f} [{lo:+.5f},{hi:+.5f}] up={(b[mask]>a[mask]).sum()} down={(b[mask]<a[mask]).sum()}")
