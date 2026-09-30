"""Read-only: can a cheap query encoder be mapped into the 1.7B document space?

Fit maps on TUNE queries (label-free: target = the 1.7B vector of the same query text),
evaluate dense NDCG@10 on DEV against the stored 1.7B corpus vectors.
No APPS test-split file is opened (only *_qtrain, *_corpus, qrels_train, texts_queries_train).
"""
import json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(r"C:\Users\Asus\manish\CodeDoctor")
sys.path.insert(0, str(ROOT))
from app.pipeline.splits import make_development_split  # noqa: E402
from app.pipeline.query_analysis import analyze_query  # noqa: E402

A = ROOT / "artifacts" / "apps"
t0 = time.time()


def load(name):
    m = np.load(A / name, allow_pickle=False).astype(np.float32)
    return m


def unit(m):
    return m / np.linalg.norm(m, axis=1, keepdims=True)


qids = json.loads((A / "ids_queries_train.json").read_text(encoding="utf-8"))
cids = json.loads((A / "ids_corpus.json").read_text(encoding="utf-8"))
qrels = json.loads((A / "qrels_train.json").read_text(encoding="utf-8"))
cpos = {c: i for i, c in enumerate(cids)}
with open(A / "texts_queries_train.jsonl", encoding="utf-8") as h:
    qt = {r["id"]: r["text"] for r in map(json.loads, h)}
texts = [qt[q] for q in qids]
split = make_development_split(texts, match_family=None, seed=20260919)
gold = np.array([cpos[next(iter(qrels[q]))] for q in qids])
fam = np.array([analyze_query(t).family.value for t in texts])
print(split.describe(), "gold multi:", sum(len(qrels[q]) > 1 for q in qids))

Q = {k: load(f"emb_{k}_qtrain.npy") for k in ["f2llm17b", "f2llm06b", "f2llm160m"]}
D = {k: unit(load(f"emb_{k}_corpus.npy")) for k in ["f2llm17b", "f2llm06b", "f2llm160m"]}
for k in Q:
    print(k, Q[k].shape, D[k].shape)
tune, dev = split.tune, split.dev


def ranks(qv, dv, rows):
    s = unit(qv[rows]) @ dv.T
    g = s[np.arange(len(rows)), gold[rows]]
    return (s > g[:, None]).sum(1) + 1


def scored(S, rows):
    g = S[np.arange(len(rows)), gold[rows]]
    return (S > g[:, None]).sum(1) + 1


def nd(r):
    return np.where(r <= 10, 1 / np.log2(r + 1), 0.0)


def rep(name, r, base=None):
    v = nd(r)
    out = {"ndcg10": round(float(v.mean()), 5), "r1": round(float((r == 1).mean()), 4),
           "r20": round(float((r <= 20).mean()), 4), "r50": round(float((r <= 50).mean()), 4),
           "stdin_ndcg10": round(float(v[fam[dev] == "competitive_stdin"].mean()), 5)}
    if base is not None:
        d = v - nd(base)
        rng = np.random.default_rng(7)
        m = np.sort(d[rng.integers(0, len(d), (2000, len(d)))].mean(1))
        out["delta"] = round(float(d.mean()), 5)
        out["ci95"] = [round(float(m[50]), 5), round(float(m[1949]), 5)]
        out["changed"] = int((d != 0).sum())
    print(f"{name:55s}", out, flush=True)
    return out


res = {}
base17 = ranks(Q["f2llm17b"], D["f2llm17b"], dev)
res["dense_17b"] = rep("dense 1.7B (sanity: DEV 0.92792)", base17)
for k in ["f2llm06b", "f2llm160m"]:
    res[f"dense_{k}"] = rep(f"dense {k} native", ranks(Q[k], D[k], dev), base17)

Y = unit(Q["f2llm17b"])


def ridge(X, Yt, lam):
    Xb = np.hstack([X, np.ones((len(X), 1), np.float32)])
    G = Xb.T @ Xb + lam * np.eye(Xb.shape[1], dtype=np.float32)
    return np.linalg.solve(G, Xb.T @ Yt)


def apply(W, X):
    return np.hstack([X, np.ones((len(X), 1), np.float32)]) @ W


for k in ["f2llm06b", "f2llm160m"]:
    X = unit(Q[k])
    # choose lambda by inner 5-fold on TUNE (cosine to target), no labels used
    best = None
    for lam in [0.01, 0.1, 1.0, 10.0]:
        rng = np.random.default_rng(0)
        idx = rng.permutation(tune)
        folds = np.array_split(idx, 5)
        cs = []
        for f in folds:
            tr = np.setdiff1d(tune, f)
            W = ridge(X[tr], Y[tr], lam)
            P = unit(apply(W, X[f]))
            cs.append(float((P * Y[f]).sum(1).mean()))
        c = float(np.mean(cs))
        print(k, "lam", lam, "inner-cv cos", round(c, 4))
        if best is None or c > best[1]:
            best = (lam, c)
    lam = best[0]
    W = ridge(X[tune], Y[tune], lam)
    P = apply(W, X)
    cosdev = float((unit(P[dev]) * Y[dev]).sum(1).mean())
    print(k, "chosen lam", lam, "DEV cos(mapped, true 1.7B)", round(cosdev, 4))
    rM = ranks(P, D["f2llm17b"], dev)
    res[f"map_{k}_ridge"] = rep(f"{k} -> ridge -> 1.7B docs", rM, base17)
    res[f"map_{k}_ridge"]["lam"] = lam
    res[f"map_{k}_ridge"]["dev_cos"] = round(cosdev, 4)
    # fusion: alpha*native + (1-alpha)*mapped, alpha fit on TUNE by NDCG (labels on TUNE only)
    Snat_t = unit(Q[k][tune]) @ D[k].T
    Smap_t = unit(P[tune]) @ D["f2llm17b"].T
    bestf = None
    for a in np.linspace(0, 1, 11):
        v = nd(scored(a * Snat_t + (1 - a) * Smap_t, tune)).mean()
        if bestf is None or v > bestf[1]:
            bestf = (float(a), float(v))
    a = bestf[0]
    Snat = unit(Q[k][dev]) @ D[k].T
    Smap = unit(P[dev]) @ D["f2llm17b"].T
    rF = scored(a * Snat + (1 - a) * Smap, dev)
    res[f"fuse_{k}"] = rep(f"{k} fuse native+mapped a={a:.1f} (TUNE-fit)", rF, base17)
    res[f"fuse_{k}"]["alpha"] = a
    # negative control: map fit on shuffled pairs must collapse
    perm = np.random.default_rng(1).permutation(tune)
    Wn = ridge(X[tune], Y[perm], lam)
    res[f"negctl_{k}"] = rep(f"{k} NEG CONTROL (shuffled pairs)", ranks(apply(Wn, X), D["f2llm17b"], dev), base17)
    # doc-pair map: fit on corpus docs (k -> 1.7B), apply to queries (cross-modal transfer)
    Wd = ridge(D[k], D["f2llm17b"], lam)
    res[f"docmap_{k}"] = rep(f"{k} map fit on corpus doc pairs", ranks(apply(Wd, X), D["f2llm17b"], dev), base17)
    del Snat_t, Smap_t, Snat, Smap

res["seconds"] = round(time.time() - t0, 1)
out = Path(__file__).with_name("qmap_result.json")
out.write_text(json.dumps(res, indent=1), encoding="utf-8")
print("done", res["seconds"], "s")
