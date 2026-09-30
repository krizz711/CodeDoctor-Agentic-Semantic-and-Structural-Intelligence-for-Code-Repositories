"""Skeptic probe for Experiment 3 (b)(c), TUNE half only.

1. Load the interactive 0.6B encoder on this laptop's CPU (float32), time the load.
2. Encode, one at a time, the gold documents of a seeded sample of TUNE queries
   (the 'changed snippets' of a hypothetical incremental update). Time each.
3. Compare with the stored GPU vectors (artifacts/apps/emb_f2llm06b_corpus.npy, fp16 on disk).
4. Mixed index: replace those rows with CPU vectors; rank ALL TUNE queries (stored query
   vectors) against pure-GPU vs mixed corpus; report top-10/top-20 identity, gold rank moves,
   dense NDCG@10 delta.
Stop rule: stop encoding after BUDGET_S seconds.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(r"C:\Users\Asus\manish\CodeDoctor")
sys.path.insert(0, str(REPO))
OUT = Path(__file__).with_name("mixed_vectors_result.json")
SEED = 20260926
N_SAMPLE = int(os.environ.get("N_SAMPLE", "150"))
BUDGET_S = float(os.environ.get("BUDGET_S", "360"))

art = REPO / "artifacts" / "apps"
tune_qids = []
with gzip.open(REPO / "data/outcomes/f2llm06b_all_tune.jsonl.gz", "rt", encoding="utf-8") as fh:
    for line in fh:
        row = json.loads(line)
        if row.get("_header"):
            continue
        tune_qids.append(row["qid"])
qrels = json.loads((art / "qrels_train.json").read_text(encoding="utf-8"))
qids_all = json.loads((art / "ids_queries_train.json").read_text(encoding="utf-8"))
cids = json.loads((art / "ids_corpus.json").read_text(encoding="utf-8"))
cpos = {c: i for i, c in enumerate(cids)}
qpos = {q: i for i, q in enumerate(qids_all)}
texts = {}
with open(art / "texts_corpus.jsonl", encoding="utf-8") as fh:
    for line in fh:
        r = json.loads(line)
        texts[r.get("id") or r.get("_id")] = r.get("text")
print(f"TUNE qids: {len(tune_qids)}; corpus texts: {len(texts)}", flush=True)

rng = np.random.default_rng(SEED)
sample = list(rng.choice(sorted(tune_qids), size=min(N_SAMPLE, len(tune_qids)), replace=False))
gold_docs = []
for q in sample:
    for d in qrels[q]:
        if d not in gold_docs:
            gold_docs.append(d)

D = np.load(art / "emb_f2llm06b_corpus.npy").astype(np.float32)
Q = np.load(art / "emb_f2llm06b_qtrain.npy").astype(np.float32)
D /= np.linalg.norm(D, axis=1, keepdims=True)
Q /= np.linalg.norm(Q, axis=1, keepdims=True)

import torch  # noqa: E402
from app.retrieval.encode import CodeEncoder  # noqa: E402

t0 = time.perf_counter()
enc = CodeEncoder(device="cpu", tier="interactive")
_ = enc.model
load_s = time.perf_counter() - t0
print(f"model {enc.model_name}@{enc.revision} dtype={enc.torch_dtype} load {load_s:.1f}s "
      f"torch threads (encode sets) = {os.cpu_count()}", flush=True)

enc_started = time.perf_counter()
per_doc = []
cpu_vecs = {}
for d in gold_docs:
    if time.perf_counter() - enc_started > BUDGET_S:
        print("stop rule: encode budget reached", flush=True)
        break
    t = time.perf_counter()
    v = np.asarray(enc.encode(texts[d], mode="document"), dtype=np.float32)
    dt = time.perf_counter() - t
    ntok = len(enc.model.tokenizer.encode(texts[d], add_special_tokens=False))
    per_doc.append({"doc": d, "seconds": round(dt, 3), "chars": len(texts[d]), "tokens": ntok})
    cpu_vecs[d] = v / np.linalg.norm(v)

encoded = list(cpu_vecs)
cos = np.array([float(cpu_vecs[d] @ D[cpos[d]]) for d in encoded])
secs = np.array([p["seconds"] for p in per_doc])
toks = np.array([p["tokens"] for p in per_doc])
print(f"encoded {len(encoded)} docs: s/doc mean {secs.mean():.3f} median {np.median(secs):.3f} "
      f"p95 {np.percentile(secs,95):.3f}; tokens median {np.median(toks):.0f}", flush=True)
print(f"cosine CPU fp32 vs stored GPU: min {cos.min():.5f} median {np.median(cos):.5f} mean {cos.mean():.5f}",
      flush=True)

# Mixed index
Dm = D.copy()
for d in encoded:
    Dm[cpos[d]] = cpu_vecs[d]
tq = [q for q in tune_qids if q in qpos]
Qt = Q[[qpos[q] for q in tq]]


def rank_all(Dx):
    S = Qt @ Dx.T
    top = np.argsort(-S, axis=1)[:, :50]
    return S, top


S0, T0 = rank_all(D)
S1, T1 = rank_all(Dm)


def ndcg10(top, q):
    gold = {cpos[d] for d, rel in qrels[q].items() if rel > 0}
    dcg = sum(1 / math.log2(i + 2) for i, doc in enumerate(top[:10]) if doc in gold)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(10, len(gold))))
    return dcg / idcg if idcg else 0.0


n0 = np.array([ndcg10(T0[i], q) for i, q in enumerate(tq)])
n1 = np.array([ndcg10(T1[i], q) for i, q in enumerate(tq)])
top10_order_same = np.mean([np.array_equal(T0[i, :10], T1[i, :10]) for i in range(len(tq))])
top10_set_same = np.mean([set(T0[i, :10]) == set(T1[i, :10]) for i in range(len(tq))])
top20_set_same = np.mean([set(T0[i, :20]) == set(T1[i, :20]) for i in range(len(tq))])
touched = [i for i in range(len(tq)) if set(T0[i, :20]) & {cpos[d] for d in encoded}]
t_order_same = np.mean([np.array_equal(T0[i, :10], T1[i, :10]) for i in touched]) if touched else float("nan")
sampled_idx = [tq.index(q) for q in sample if q in tq]
gold_rank_moves = 0
for i in sampled_idx:
    q = tq[i]
    g = [cpos[d] for d in qrels[q] if d in cpu_vecs]
    for gi in g:
        r0 = int(np.sum(S0[i] > S0[i, gi]))
        r1 = int(np.sum(S1[i] > S1[i, gi]))
        gold_rank_moves += int(r0 != r1)
changed_q = int(np.sum(n0 != n1))
# Score margin context: gap between rank-1 and rank-2 dense scores
gaps = np.sort(S0, axis=1)[:, ::-1]
margin = gaps[:, 0] - gaps[:, 1]
score_delta = np.array([abs(float(Qt[i] @ (cpu_vecs[d] - D[cpos[d]])))
                        for i in range(0, len(tq), 5) for d in encoded[:20]])

result = {
    "seed": SEED, "n_sample_queries": len(sample), "n_docs_encoded": len(encoded),
    "model": enc.model_name, "revision": enc.revision, "dtype": enc.torch_dtype,
    "load_seconds": round(load_s, 2),
    "sec_per_doc": {"mean": float(secs.mean()), "median": float(np.median(secs)),
                     "p95": float(np.percentile(secs, 95)), "max": float(secs.max())},
    "tokens": {"median": float(np.median(toks)), "p95": float(np.percentile(toks, 95)), "max": int(toks.max())},
    "cosine_cpu_vs_stored_gpu": {"min": float(cos.min()), "median": float(np.median(cos)),
                                  "p05": float(np.percentile(cos, 5))},
    "tune_queries_ranked": len(tq),
    "top10_order_identical_frac": float(top10_order_same),
    "top10_set_identical_frac": float(top10_set_same),
    "top20_set_identical_frac": float(top20_set_same),
    "queries_whose_top20_contains_a_replaced_doc": len(touched),
    "top10_order_identical_frac_among_those": float(t_order_same),
    "gold_rank_changes_in_sampled_queries": gold_rank_moves,
    "queries_with_ndcg10_change": changed_q,
    "mean_ndcg10_gpu": float(n0.mean()), "mean_ndcg10_mixed": float(n1.mean()),
    "delta_ndcg10": float(n1.mean() - n0.mean()),
    "median_top1_top2_margin": float(np.median(margin)),
    "frac_margin_below_0.002": float(np.mean(margin < 0.002)),
    "abs_score_delta_from_doc_swap": {"median": float(np.median(score_delta)),
                                       "p99": float(np.percentile(score_delta, 99))},
    "per_doc": per_doc,
}
OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k != "per_doc"}, indent=1))
