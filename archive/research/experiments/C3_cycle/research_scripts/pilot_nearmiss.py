"""Read-only pilot (TUNE half, stdin family, APPS *train* only).

Question: among single-edit mutants that PASS the statement's worked examples,
(1) how many are real near-misses (fail APPS official tests) vs equivalent;
(2) do type-aware mutated sample inputs expose them (outputs differ from gold);
(3) can MBR-exec (agreement across versions, no oracle output) pick the gold.
Official tests are used only as the judge / as a ceiling input source.
"""
from __future__ import annotations

import ast
import json
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(r"C:\Users\Asus\manish\CodeDoctor")
sys.path.insert(0, str(REPO))
sys.set_int_max_str_digits(0)

from app.pipeline.execution_rerank import (  # noqa: E402
    ExecutionConfig, _normalize_output, compare_output, execute_candidate, parse_worked_examples,
)
from app.pipeline.query_analysis import QueryFamily, analyze_query  # noqa: E402
from app.pipeline.splits import make_development_split  # noqa: E402
from scripts.build_version_benchmark import _mutate, _sites  # noqa: E402
from scripts.dev_eval import load_jsonl_texts  # noqa: E402

N_STATEMENTS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SITES = 30
N_OFFICIAL = 8
N_GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 12
SEED = 7
CFG = ExecutionConfig(enabled=True, timeout_seconds=2.0)
POOL = ThreadPoolExecutor(max_workers=6)
RUNS = Counter()


def run(code, stdin):
    RUNS["n"] += 1
    o = execute_candidate(code, stdin, CFG)
    if not o.ok:
        return ("FAIL", o.status)
    return ("OK", _normalize_output(o.stdout))


def passes(code, pairs):
    for i, e in pairs:
        s = run(code, i)
        if s[0] != "OK" or compare_output(s[1], e, CFG) is None:
            return False
    return True


INT = re.compile(r"-?\d+")


def mutate_input(text: str, rng: random.Random) -> str:
    """Type-aware, format-preserving: perturb integer tokens only (EvalPlus-style)."""
    toks = list(INT.finditer(text))
    if not toks:
        return text
    out, last = [], 0
    k = max(1, len(toks) // 4)
    chosen = set(rng.sample(range(len(toks)), min(k, len(toks))))
    for j, m in enumerate(toks):
        out.append(text[last:m.start()])
        v = int(m.group())
        if j in chosen:
            op = rng.choice(["+1", "-1", "x2", "0", "1", "swap"])
            if op == "+1": v += 1
            elif op == "-1": v = v - 1 if v > 0 else v + 1
            elif op == "x2": v *= 2
            elif op == "0": v = 0 if v >= 0 else v
            elif op == "1": v = 1
            elif op == "swap":
                other = int(toks[rng.randrange(len(toks))].group())
                v = other
        out.append(str(v))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def main():
    t0 = time.time()
    A = REPO / "artifacts" / "apps"
    qids, qtexts = load_jsonl_texts(A / "texts_queries_train.jsonl")
    cids, ctexts = load_jsonl_texts(A / "texts_corpus.jsonl")
    text_of = dict(zip(cids, ctexts))
    qrels = json.loads((A / "qrels_train.json").read_text(encoding="utf-8"))
    by_text = {}
    with (REPO / "artifacts" / "oracle" / "apps_train.jsonl").open(encoding="utf-8") as h:
        for line in h:
            r = json.loads(line)
            by_text.setdefault(" ".join(r["question"].split()), r)
    split = make_development_split(qtexts, match_family=None, seed=20260919)
    rows = list(split.tune)
    rng = random.Random(SEED)
    rng.shuffle(rows)
    items = []
    for row in rows:
        t = qtexts[row]
        if analyze_query(t).family is not QueryFamily.COMPETITIVE_STDIN:
            continue
        ex = parse_worked_examples(t, limit=2)
        r = by_text.get(" ".join(t.split()))
        if not ex or r is None or not r["input_output"]:
            continue
        io = json.loads(r["input_output"])
        ins, outs = io.get("inputs", []), io.get("outputs", [])
        if io.get("fn_name") or not ins:
            continue
        off = [(i, o) for i, o in zip(ins, outs) if isinstance(i, str) and isinstance(o, str) and len(i) < 20000][:N_OFFICIAL]
        if len(off) < 2:
            continue
        gold = next(iter(qrels[qids[row]]))
        try:
            tree = ast.parse(text_of[gold]); norm = ast.unparse(tree)
        except (SyntaxError, ValueError, RecursionError):
            continue
        items.append((qids[row], tree, norm, [(e.stdin, e.expected_stdout) for e in ex], off))
        if len(items) >= int(N_STATEMENTS * 1.25):
            break

    def study(item):
        qid, tree, norm, samples, off = item
        if not passes(norm, samples) or not passes(norm, off):
            return None
        local = random.Random(f"{SEED}-{qid}")
        sites = _sites(tree); local.shuffle(sites)
        seen = {norm}; counts = Counter(); sample_pass = []
        for site in sites[:SITES]:
            code = _mutate(tree, site, local)
            if code is None or code in seen:
                continue
            seen.add(code)
            ok = passes(code, samples)
            counts["mutants"] += 1
            if ok:
                sample_pass.append(code)
        near, equiv = [], []
        for code in sample_pass:
            (equiv if passes(code, off) else near).append(code)
        counts["sample_pass"] = len(sample_pass); counts["near"] = len(near); counts["equiv"] = len(equiv)
        res = {"qid": qid, "counts": dict(counts)}
        if not near:
            return res
        near = near[:3]
        versions = [norm] + near  # index 0 = gold
        grng = random.Random(f"gen-{qid}")
        gen = []
        for j in range(N_GEN):
            base = samples[j % len(samples)][0]
            gen.append(mutate_input(base, grng))
        gen = list(dict.fromkeys(g for g in gen if g not in {s for s, _ in samples}))
        off_in = [i for i, _ in off]
        sig_gen = [[run(v, g) for g in gen] for v in versions]
        sig_off = [[run(v, g) for g in off_in] for v in versions]
        res.update(gen=len(gen), sig_gen=sig_gen, sig_off=sig_off, k=len(versions))
        return res

    results = [r for r in POOL.map(study, items) if r is not None][:N_STATEMENTS]
    elapsed = time.time() - t0

    # --- analysis
    tot = Counter()
    for r in results:
        tot.update(r["counts"])
    def expose(sig):
        # per near-miss mutant: differs from gold on >= 1 input
        g = sig[0]
        return [any(a != b for a, b in zip(g, m)) for m in sig[1:]]
    def mbr(sig, valid_only):
        k = len(sig); n = len(sig[0])
        cols = range(n)
        if valid_only:  # keep inputs where a majority of versions complete (no gold knowledge)
            cols = [c for c in cols if sum(sig[v][c][0] == "OK" for v in range(k)) * 2 > k]
        score = [0.0] * k
        for c in cols:
            for v in range(k):
                score[v] += sum(sig[v][c] == sig[u][c] for u in range(k) if u != v)
                # crash asymmetry: a version that fails where others ran is penalised
        best = max(score)
        tied = [v for v in range(k) if score[v] == best]
        return (1.0 / len(tied)) if 0 in tied else 0.0, len(cols)
    def crash_rule(sig):
        # pick versions with fewest FAIL on inputs; then chance
        k = len(sig)
        fails = [sum(s[0] != "OK" for s in sig[v]) for v in range(k)]
        best = min(fails); tied = [v for v in range(k) if fails[v] == best]
        return (1.0 / len(tied)) if 0 in tied else 0.0
    def plurality(sig):
        k = len(sig); n = len(sig[0]); pts = [0] * k
        for c in range(n):
            if sum(sig[v][c][0] == "OK" for v in range(k)) * 2 <= k:
                continue
            cnt = Counter(tuple(sig[v][c]) for v in range(k))
            top = max(cnt.values())
            if top == k:
                continue
            for v in range(k):
                if cnt[tuple(sig[v][c])] == top:
                    pts[v] += 1
        best = max(pts); tied = [v for v in range(k) if pts[v] == best]
        return (1.0 / len(tied)) if 0 in tied else 0.0
    lin = [r for r in results if "sig_gen" in r]
    ex_gen = [x for r in lin for x in expose(r["sig_gen"])]
    ex_off = [x for r in lin for x in expose(r["sig_off"])]
    by_k = Counter(r["k"] for r in lin)
    chance = sum(1.0 / r["k"] for r in lin) / max(1, len(lin))
    mb_gen = [mbr(r["sig_gen"], True)[0] for r in lin]
    mb_off = [mbr(r["sig_off"], True)[0] for r in lin]
    mb_both = [mbr([a + b for a, b in zip(r["sig_gen"], r["sig_off"])], True)[0] for r in lin]
    cr_gen = [crash_rule(r["sig_gen"]) for r in lin]
    lin3 = [r for r in lin if r["k"] >= 3]
    out = {
        "statements_studied": len(results), "runs": RUNS["n"], "seconds": round(elapsed, 1),
        "mutants_total": tot["mutants"], "sample_pass": tot["sample_pass"],
        "sample_pass_rate": tot["sample_pass"] / max(1, tot["mutants"]),
        "near_miss": tot["near"], "equivalent_on_official": tot["equiv"],
        "statements_with_near_miss": len(lin), "lineage_sizes": dict(by_k),
        "near_miss_exposed_by_generated_inputs": sum(ex_gen) / max(1, len(ex_gen)),
        "near_miss_exposed_by_official_inputs": sum(ex_off) / max(1, len(ex_off)),
        "n_near_miss_in_lineages": len(ex_gen),
        "gold_first_chance": chance,
        "gold_first_mbr_generated": sum(mb_gen) / max(1, len(lin)),
        "gold_first_mbr_official_inputs": sum(mb_off) / max(1, len(lin)),
        "gold_first_mbr_both": sum(mb_both) / max(1, len(lin)),
        "gold_first_fewest_crashes_generated": sum(cr_gen) / max(1, len(lin)),
        "gold_first_plurality_generated": sum(plurality(r["sig_gen"]) for r in lin) / max(1, len(lin)),
        "gold_first_plurality_official": sum(plurality(r["sig_off"]) for r in lin) / max(1, len(lin)),
        "k>=3: plurality_generated": sum(plurality(r["sig_gen"]) for r in lin3) / max(1, len(lin3)),
        "k>=3: plurality_official": sum(plurality(r["sig_off"]) for r in lin3) / max(1, len(lin3)),
        "k>=3: n": len(lin3),
        "k>=3: chance": sum(1.0 / r["k"] for r in lin3) / max(1, len(lin3)),
        "k>=3: mbr_generated": sum(mbr(r["sig_gen"], True)[0] for r in lin3) / max(1, len(lin3)),
        "k>=3: mbr_official": sum(mbr(r["sig_off"], True)[0] for r in lin3) / max(1, len(lin3)),
        "gen_inputs_per_statement_mean": sum(r["gen"] for r in lin) / max(1, len(lin)),
    }
    here = Path(__file__).parent
    (here / "pilot_nearmiss_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (here / "pilot_nearmiss_raw.json").write_text(json.dumps(results), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
