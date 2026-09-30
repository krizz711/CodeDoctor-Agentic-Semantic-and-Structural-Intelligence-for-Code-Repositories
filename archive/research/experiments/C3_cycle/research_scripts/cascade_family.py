"""Read-only: 0.6B->1.7B cascade escalation rate and delta, split by query family (stdin vs rest)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(r"C:\Users\Asus\manish\CodeDoctor")
sys.path.insert(0, str(ROOT))
from scripts.replay_outcomes import Policy, load_store, metrics  # noqa: E402
from scripts.simulate_cascade import _ranked  # noqa: E402
from app.pipeline.query_analysis import analyze_query  # noqa: E402

A = ROOT / "artifacts" / "apps"
with open(A / "texts_queries_train.jsonl", encoding="utf-8") as h:
    fam = {r["id"]: analyze_query(r["text"]).family.value for r in map(json.loads, h)}
gold = {q: next(iter(v)) for q, v in json.loads((A / "qrels_train.json").read_text(encoding="utf-8")).items()}
policy = Policy("call-style", numeric_tolerance=1e-6, call_style=True)
for half in ["dev", "tune"]:
    _, small = load_store(ROOT / f"data/outcomes/f2llm06b_all_{half}_r2.jsonl.gz")
    _, large = load_store(ROOT / f"data/outcomes/f2llm17b_all_{half}.jsonl.gz")
    L = {e["qid"]: e for e in large}
    out = {}
    for s in small:
        q = s["qid"]; l = L[q]
        ev = {r["doc"]: r for r in l["candidates"]}; ev.update({r["doc"]: r for r in s["candidates"]})
        lr, _, _ = _ranked(l, ev, policy, 20)
        sr, strong, _ = _ranked(s, ev, policy, 20)
        top1_strong = strong and sr[0] in {r for r in [sr[0]]}
        f = "stdin" if fam[q] == "competitive_stdin" else "other"
        nl = metrics(lr, gold[q])[0]
        # stop rule A: any strong verification; stop rule B: 0.6B dense rank-1 itself verifies
        ns = metrics(sr, gold[q])[0]
        dense_top1_verified = strong and s["pool"][0] == sr[0]
        for rule, stop in (("any_strong", strong), ("dense_top1_verified", dense_top1_verified)):
            o = out.setdefault((f, rule), {"n": 0, "esc": 0, "d": []})
            o["n"] += 1
            o["esc"] += (not stop)
            o["d"].append((ns if stop else nl) - nl)
    for (f, rule), o in sorted(out.items()):
        d = np.array(o["d"]); esc = o["esc"] / o["n"]
        rng = np.random.default_rng(7); m = np.sort(d[rng.integers(0, len(d), (2000, len(d)))].mean(1))
        cost = 2.79 + esc * 7.83
        print(f"{half} {f:5s} {rule:20s} n={o['n']} escalate={esc:.3f} delta={d.mean():+.5f} "
              f"[{m[50]:+.5f},{m[1949]:+.5f}] changed={(d != 0).sum()} est_s/query={cost:.2f} vs 7.83 "
              f"({1 - cost / 7.83:+.1%} saving; estimate from Kaggle 2-core means)")
