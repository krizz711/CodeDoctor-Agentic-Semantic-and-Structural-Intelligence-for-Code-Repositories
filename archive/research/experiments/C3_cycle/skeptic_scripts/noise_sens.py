"""How much dense NDCG@10 does a query-vector error of a given cosine cost?  TUNE half only.

Perturbs the stored 1.7B TUNE query vectors to an exact target cosine and re-ranks the
full APPS corpus.  Two error shapes: isotropic, and shaped by the covariance of the
stored query vectors (errors propagated through a network live in its active subspace).
This is a model of quantisation error, not a measurement of it.
"""
import gzip, json, sys
from pathlib import Path
import numpy as np
REPO = Path(r"C:/Users/Asus/manish/CodeDoctor")
sys.path.insert(0, str(REPO))
from scripts.dev_eval import load_jsonl_texts
art = REPO / "artifacts/apps"
qids, _ = load_jsonl_texts(art / "texts_queries_train.jsonl")
cids, _ = load_jsonl_texts(art / "texts_corpus.jsonl")
qrels = json.loads((art / "qrels_train.json").read_text(encoding="utf-8"))
with gzip.open(REPO / "data/outcomes/f2llm17b_all_tune.jsonl.gz", "rt", encoding="utf-8") as f:
    tune = [json.loads(l).get("qid") for l in f][1:]
pos = {q: i for i, q in enumerate(qids)}; dpos = {d: i for i, d in enumerate(cids)}
rows = np.array([pos[q] for q in tune])
Q = np.load(art / "emb_f2llm17b_qtrain.npy").astype(np.float32)
D = np.load(art / "emb_f2llm17b_corpus.npy").astype(np.float32)
D /= np.linalg.norm(D, axis=1, keepdims=True)
allQ = Q / np.linalg.norm(Q, axis=1, keepdims=True)
Qt = allQ[rows]
gold = np.array([dpos[next(iter(qrels[q]))] for q in tune])
print("TUNE", len(tune), "corpus", D.shape)

def ranks(Qm):
    S = Qm @ D.T
    g = S[np.arange(len(S)), gold]
    return (S > g[:, None]).sum(1)  # 0-based rank of gold

def ndcg10(r):
    return np.where(r < 10, 1.0 / np.log2(r + 2), 0.0)

r0 = ranks(Qt); n0 = ndcg10(r0)
print(f"baseline dense NDCG@10 {n0.mean():.5f}  top1 {np.mean(r0==0):.4f}")
# covariance shaping from ALL train query vectors (no labels involved)
C = np.cov(allQ.T); w, V = np.linalg.eigh(C); w = np.clip(w, 0, None)
sqrtC = (V * np.sqrt(w)) @ V.T
rng_master = np.random.default_rng(20260926)
def perturb(Qm, c, shape, rng):
    z = rng.standard_normal(Qm.shape).astype(np.float32)
    if shape == "cov":
        z = z @ sqrtC.T.astype(np.float32)
    z -= (z * Qm).sum(1, keepdims=True) * Qm  # orthogonal to q
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return c * Qm + np.sqrt(1 - c * c) * z

out = {}
for shape in ("iso", "cov"):
    for c in (0.9999, 0.999, 0.998, 0.997, 0.995, 0.99):
        ds, changed, top1 = [], [], []
        for s in range(3):
            r = ranks(perturb(Qt, c, shape, np.random.default_rng(1000 * s + int(c * 1e4))))
            n = ndcg10(r)
            ds.append(n.mean() - n0.mean()); changed.append(int((n != n0).sum())); top1.append(int((r == 0).sum() - (r0 == 0).sum()))
        out[f"{shape}_{c}"] = dict(delta=np.round(ds, 5).tolist(), changed=changed, top1=top1)
        print(f"{shape:4s} cos={c:<7} dNDCG@10 mean {np.mean(ds):+.5f} (seeds {np.round(ds,5).tolist()})  statements changed {changed}  top1 change {top1}")
# gold margin: score(gold) - best non-gold, among rank-0 golds
S = Qt @ D.T; g = S[np.arange(len(S)), gold]; S[np.arange(len(S)), gold] = -9
m = g - S.max(1)
print("margin (gold - best other) quantiles:", np.round(np.quantile(m, [0.01, 0.05, 0.1, 0.25, 0.5]), 4).tolist(), " share |margin|<0.005:", round(float(np.mean(np.abs(m) < 0.005)), 4), " <0.01:", round(float(np.mean(np.abs(m) < 0.01)), 4))
json.dump(out, open("noise_sens.json", "w"), indent=1)
