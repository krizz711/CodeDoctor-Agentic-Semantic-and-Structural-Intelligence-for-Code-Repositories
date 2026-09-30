"""P0: why execution adds nothing on CC-onesol (pre-registration: experiments/P0_ceiling_anomaly/prereg.md).

Dense ranks are recomputed from the stored 1.7B vectors; no encoder runs, and no APPS
test-split file is opened.  M3 and M4 run the frozen execution rule live.

usage: python scripts/p0_headroom.py [--workers 6] [--m3 200] [--out result.json]
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.pipeline.execution_rerank import ExecutionConfig, apply_execution_rerank  # noqa: E402

APPS, EXT = ROOT / "artifacts" / "apps", ROOT / "artifacts" / "ext_cc"
SEED = 20260926
CC_POOL = 217  # same-source documents a CC-onesol query competes with (gold + 216)


def unit(path: Path) -> np.ndarray:
    m = np.load(path, allow_pickle=False).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def texts(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as h:
        return {r["id"]: r["text"] for r in map(json.loads, h)}


def ndcg(rank: int) -> float:
    return 1 / math.log2(rank + 1) if rank <= 10 else 0.0


def summary(ranks: list[int]) -> dict[str, float]:
    n = len(ranks)
    dense = sum(map(ndcg, ranks)) / n
    ceiling = sum(1.0 if r <= 20 else ndcg(r) for r in ranks) / n
    return {"n": n, "ndcg10": round(dense, 5), "r1": round(sum(r == 1 for r in ranks) / n, 4),
            "ceiling20": round(ceiling, 5), "headroom": round(ceiling - dense, 5)}


def boot(diffs: list[float], seed: int = 7, reps: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs)
    means = np.sort(d[rng.integers(0, len(d), (reps, len(d)))].mean(axis=1))
    return round(float(means[int(0.025 * reps)]), 5), round(float(means[int(0.975 * reps) - 1]), 5)


def exec_config(workers: int) -> ExecutionConfig:
    # config/frozen.yaml: depth 20, 2 examples, 2 s wall clock, numeric 1e-6; a bonus
    # larger than any cosine gap, as in the scored path.
    return ExecutionConfig(enabled=True, max_candidates=20, timeout_seconds=2.0, numeric_tolerance=1e-6,
                           workers=workers, verified_bonus=10.0)


def rerank(query: str, docs: list[str], scores: list[float], corpus: dict[str, str], cfg) -> tuple[list[str], dict]:
    top = docs[:20]
    new, audit = apply_execution_rerank(query, [corpus[d] for d in top], list(scores[:20]), cfg)
    verified = [top[j] for j in range(len(top)) if new[j] - scores[j] > 5.0]
    order = [top[j] for j in sorted(range(len(top)), key=lambda j: -new[j])] + docs[20:]
    return order, {"examples": audit.get("examples_found", 0), "verified": verified}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--m3", type=int, default=200)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "P0_ceiling_anomaly" / "result.json")
    args = ap.parse_args()
    res: dict[str, object] = {}
    cfg = exec_config(args.workers)

    # ---- APPS DEV stdin (train split only) ----
    C, Q = unit(APPS / "emb_f2llm17b_corpus.npy"), unit(APPS / "emb_f2llm17b_qtrain.npy")
    cids = json.loads((APPS / "ids_corpus.json").read_text(encoding="utf-8"))
    qids = json.loads((APPS / "ids_queries_train.json").read_text(encoding="utf-8"))
    qrels = json.loads((APPS / "qrels_train.json").read_text(encoding="utf-8"))
    crow, qrow = {d: i for i, d in enumerate(cids)}, {q: i for i, q in enumerate(qids)}
    with gzip.open(ROOT / "data" / "outcomes" / "f2llm17b_all_dev.jsonl.gz", "rt", encoding="utf-8") as h:
        h.readline()
        dev = [e for e in map(json.loads, h) if e["family"] == "competitive_stdin"]
    dev.sort(key=lambda e: int(e["qid"][1:]))
    gold = {e["qid"]: next(iter(qrels[e["qid"]])) for e in dev}
    S = Q[[qrow[e["qid"]] for e in dev]] @ C.T                     # 882 x 8765
    g_idx = np.array([crow[gold[e["qid"]]] for e in dev])
    g_score = S[np.arange(len(dev)), g_idx]
    full_rank = (S > g_score[:, None]).sum(axis=1) + 1
    res["M0_apps_dev_stdin_full"] = summary(full_rank.tolist())

    # M2: gold + 216 random other APPS documents, 5 seeds; negative control: the 216 highest-scoring others.
    m2 = []
    others = np.ones_like(S, dtype=bool)
    others[np.arange(len(dev)), g_idx] = False
    for s in range(5):
        rng = np.random.default_rng(SEED + s)
        ranks = []
        for i in range(len(dev)):
            pick = rng.choice(np.flatnonzero(others[i]), CC_POOL - 1, replace=False)
            ranks.append(int((S[i, pick] > g_score[i]).sum()) + 1)
        m2.append(summary(ranks))
    res["M2_apps_random217"] = {"seeds": m2, "mean_headroom": round(float(np.mean([x["headroom"] for x in m2])), 5),
                                "mean_ndcg10": round(float(np.mean([x["ndcg10"] for x in m2])), 5)}
    neg = []
    for i in range(len(dev)):
        row = np.where(others[i], S[i], -np.inf)
        top = np.argpartition(-row, CC_POOL - 1)[: CC_POOL - 1]
        neg.append(int((S[i, top] > g_score[i]).sum()) + 1)
    res["negative_control_apps_top217"] = summary(neg)

    # ---- CC-onesol (EXT-dev only; the lockbox is never scored) ----
    ec, eq = texts(EXT / "texts_corpus.jsonl"), texts(EXT / "texts_queries.jsonl")
    ecid, eqid = list(ec), list(eq)
    splits = json.loads((EXT / "splits.json").read_text(encoding="utf-8"))
    eqrels = json.loads((EXT / "qrels.json").read_text(encoding="utf-8"))
    EC, EQ = unit(EXT / "emb_f2llm17b_corpus.npy"), unit(EXT / "emb_f2llm17b_queries.npy")
    assert EC.shape[0] == len(ecid) and EQ.shape[0] == len(eqid)
    keep = np.array([not d.startswith("ccx-") for d in ecid])
    kid = [d for d, k in zip(ecid, keep) if k]
    ECk = EC[keep]
    is_ccd = np.array([d.startswith("ccd-") for d in kid])
    dev_q = [q for q in eqid if q in set(splits["ext_dev"])]
    cc = {}
    for q in dev_q:
        s = ECk @ EQ[eqid.index(q)]
        g = next(iter(eqrels[q]))
        gs = s[kid.index(g)]
        order = np.argsort(-s)
        top20 = order[:20]
        cc[q] = {"rank": int((s > gs).sum()) + 1,
                 "rank_among_ccd": int((s[is_ccd] > gs).sum()) + 1,
                 "ccd_in_top20": int(is_ccd[top20].sum()),
                 "docs": [kid[j] for j in order[:50]], "scores": [float(s[j]) for j in order[:50]], "gold": g}
    for label, pre in (("cc_test", "cc-test-"), ("cc_valid", "cc-valid-")):
        sel = [q for q in dev_q if q.startswith(pre)]
        res[f"M0_M5_{label}"] = summary([cc[q]["rank"] for q in sel])
        res[f"M1_{label}"] = {
            "ccd_share_top20": round(sum(cc[q]["ccd_in_top20"] for q in sel) / (20 * len(sel)), 4),
            "ccd_share_corpus": round(float(is_ccd.mean()), 4),
            "gold_r1_among_ccd_only": round(sum(cc[q]["rank_among_ccd"] == 1 for q in sel) / len(sel), 4),
        }

    # M5: statement length (characters).
    apps_q = texts(APPS / "texts_queries_train.jsonl")
    res["M5_median_chars"] = {"apps_dev_stdin": int(np.median([len(apps_q[e["qid"]]) for e in dev])),
                              "cc_ext_dev": int(np.median([len(eq[q]) for q in dev_q]))}

    # M4: autopsy of the cc-test misses (live frozen rule on the CC-onesol top 20).
    autopsy = []
    for q in [q for q in dev_q if q.startswith("cc-test-") and cc[q]["rank"] > 1]:
        x = cc[q]
        order, info = rerank(eq[q], x["docs"], x["scores"], ec, cfg)
        autopsy.append({"qid": q, "dense_rank": x["rank"], "examples_parsed": info["examples"],
                        "gold_verifies": x["gold"] in info["verified"],
                        "others_verifying": [d for d in info["verified"] if d != x["gold"]],
                        "rank_after_execution": order.index(x["gold"]) + 1 if x["gold"] in order else None,
                        "dense_top3": x["docs"][:3]})
    res["M4_autopsy_cc_test"] = autopsy

    # M3: live frozen rule on shrunken pools vs full pools, same statements.
    corpus = texts(APPS / "texts_corpus.jsonl")
    rng = np.random.default_rng(SEED + 100)
    eligible = [i for i, e in enumerate(dev) if e["stdin_examples"]]
    chosen = sorted(rng.choice(eligible, min(args.m3, len(eligible)), replace=False).tolist())
    rows = []
    t0 = time.perf_counter()
    for i in chosen:
        e, g = dev[i], gold[dev[i]["qid"]]
        pick = rng.choice(np.flatnonzero(others[i]), CC_POOL - 1, replace=False)
        pool = np.concatenate([[g_idx[i]], pick])
        pool = pool[np.argsort(-S[i, pool])]
        small_docs, small_scores = [cids[j] for j in pool], [float(S[i, j]) for j in pool]
        full_docs, full_scores = e["pool"], e["scores"]
        text = apps_q[e["qid"]]
        s_exec, _ = rerank(text, small_docs, small_scores, corpus, cfg)
        f_exec, _ = rerank(text, full_docs, full_scores, corpus, cfg)
        rank = lambda order: order.index(g) + 1 if g in order else 999  # noqa: E731
        rows.append({"qid": e["qid"], "small_dense": rank(small_docs), "small_exec": rank(s_exec),
                     "full_dense": rank(full_docs), "full_exec": rank(f_exec)})
    m3 = {"n": len(rows), "wall_s": round(time.perf_counter() - t0, 1), "workers": args.workers}
    for k in ("small", "full"):
        d = [ndcg(r[f"{k}_exec"]) - ndcg(r[f"{k}_dense"]) for r in rows]
        m3[k] = {"dense": round(float(np.mean([ndcg(r[f"{k}_dense"]) for r in rows])), 5),
                 "exec": round(float(np.mean([ndcg(r[f"{k}_exec"]) for r in rows])), 5),
                 "gain": round(float(np.mean(d)), 5), "ci95": boot(d),
                 "headroom": summary([r[f"{k}_dense"] for r in rows])["headroom"],
                 "up": sum(x > 0 for x in d), "down": sum(x < 0 for x in d)}
    res["M3_live_exec"] = m3

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=2), encoding="utf-8")
    tmp.replace(args.out)
    print(json.dumps({k: v for k, v in res.items() if k != "M4_autopsy_cc_test"}, indent=1))
    print("M4", json.dumps(autopsy, indent=1))


if __name__ == "__main__":
    main()
