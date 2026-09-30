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


def boot_ci(d, B=2000, seed=7):
    rng = np.random.default_rng(seed); n = len(d)
    idx = rng.integers(0, n, size=(B, n)); m = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))
def signflip(d, B=5000, seed=11):
    rng = np.random.default_rng(seed); obs = abs(d.mean())
    s = rng.choice([-1, 1], size=(B, len(d))); return float(np.mean(np.abs((s * d).mean(1)) >= obs))
for c in (0.999, 0.997, 0.995, 0.99, 0.98):
    for s in range(3):
        rng = np.random.default_rng(500 + s)
        u = rng.standard_normal(Qt.shape[1]).astype(np.float32) @ sqrtC.T.astype(np.float32)
        u /= np.linalg.norm(u)
        Z = np.repeat(u[None, :], len(Qt), 0)
        Z -= (Z * Qt).sum(1, keepdims=True) * Qt; Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        Qp = c * Qt + np.sqrt(1 - c * c) * Z
        n = ndcg10(ranks(Qp)); d = n - n0
        m, lo, hi = boot_ci(d)
        print(f"shared cos={c:<6} seed{s}: dNDCG@10 {m:+.5f} CI [{lo:+.5f},{hi:+.5f}] changed {int((d!=0).sum())} signflip p {signflip(d):.3f}")
# power check at the iso/cov level: CI width for cov 0.997 seed 0
Qp = perturb(Qt, 0.997, "cov", np.random.default_rng(0)); d = ndcg10(ranks(Qp)) - n0
print("cov 0.997 single draw: mean %+.5f CI [%+.5f,%+.5f] changed %d" % (*boot_ci(d), int((d!=0).sum())))
